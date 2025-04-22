import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.functional import interpolate
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d

from dhg import Hypergraph
from dhg.nn import HGNNConv
from dhg.models import HGNN
from einops import rearrange, repeat,reduce
# from timm.models.layers import DropPath
import cv2
import numpy as np
import math
import time
## CNN + LSTM 的基本框架
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def norm_layer(norm, nc):
    # normalization layer 2d
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class BasicConv_2(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv_2, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            # elif pool_type == 'lse':
            #     # LSE pool only
            #     lse_pool = logsumexp_2d(x)
            #     channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
    
class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False, use_cbam=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        # elif norm == 'adain':
        #     self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if norm == 'sn':
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st,padding=padding,bias=self.use_bias)

        if use_cbam:
            self.cbam = CBAM(out_dim, 16, no_spatial=True)
        else:
            self.cbam = None

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            # x = self.conv(self.pad(x))
            x = self.conv(x)
            
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
            if self.cbam:
                x = self.cbam(x)
            if self.activation:
                x = self.activation(x)
        return x

class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=1, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class CNN2d_t2_v_a_three_branch(nn.Module):
    def __init__(self,num_classes=1):
        super(CNN2d_t2_v_a_three_branch, self).__init__()
        ## one stage
        resnet = models.resnet18(pretrained=True)
        self.tumor_conv1_v =  Conv2dBlock(1, 3, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 =  Conv2dBlock(1, 3, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_a =  Conv2dBlock(1, 3, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        
        self.resnet_head_A=nn.Sequential(*list(resnet.children())[:-5])                                           
        self.resnet_head_V=nn.Sequential(*list(resnet.children())[:-5])
        self.resnet_head_T2=nn.Sequential(*list(resnet.children())[:-5])
        # self.pool = nn.AvgPool3d(kernel_size=(1,4,4))   
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM_A  = nn.LSTM(
                input_size=128,
                hidden_size=96,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        self.LSTM_V  = nn.LSTM(
                input_size=5*256,
                hidden_size=32,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        self.LSTM_T2  = nn.LSTM(
                input_size=5*256,
                hidden_size=32,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        
        self.fc1 = nn.Linear(96, 32)
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)

        self.stem_v = Stem(out_dim=640) ## 特征提取器
        

    def forward(self,X_image_mask):
        
        # X_A_mask, X_A_image, X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask = X_image_mask
        
        b,n,c,h,w = X_liver_mask.shape
        
        # X_A_image = X_A_image.view(b*n,c,h,w).to(torch.float32)
        X_t2_tumor = X_tumor.view(b*n,c,h,w).to(torch.float32)
        X_tv_tumor = X_tv_tumor.view(b*n,c,h,w).to(torch.float32)
        

        ## 这个地方不使用通道数不变的原则了，因为避免提问
        X_tv_tumor = Stem(X_tv_tumor)

        # X_A_image = self.tumor_conv1_a(X_A_image)
        X_T2_image = self.tumor_conv1_t2(X_T2_image)
        X_V_image = self.tumor_conv1_v(X_V_image)
        
        # X_A_image = self.resnet_head_A(X_A_image)
        X_T2_image = self.resnet_head_T2(X_T2_image)
        X_V_image = self.resnet_head_V(X_V_image)
        
        # X_A_image = self.pool(X_A_image).squeeze().view(b,n,-1)
        X_T2_image = self.pool(X_T2_image).squeeze().view(b,n,-1)
        X_V_image = self.pool(X_V_image).squeeze().view(b,n,-1)
        
        X_pool = torch.cat([ X_T2_image, X_V_image], dim=-1)
        # X_pool = torch.cat([X_A_image, X_T2_image, X_V_image], dim=-1)
        
        RNN_out, (h_n, h_c) = self.LSTM_A(X_pool, None)
        x = RNN_out[:, -1,]
        x = self.fc1(x)
        
        # x = self.fc_mask(x)
        x = F.relu(x)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        return x

       
        
        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool],dim=-1)
        # x = self.fc1(X_pool)
        x = self.fc1(x)
        
        # x = self.fc_mask(x)
        x = F.relu(x)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        return x


class CNN3d_t2_tv_baseline_model(nn.Module):
    def __init__(self,num_classes=1):
        super(CNN3d_t2_tv_baseline_model, self).__init__()
        ## one stage
     
        self.liver_conv1_v =  Conv2dBlock(1, 16, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv1_t2 =  Conv2dBlock(1, 16, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')

        self.tumor_conv1_v =  Conv2dBlock(1, 16, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 =  Conv2dBlock(1, 16, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')                                                 
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM  = nn.LSTM(
                input_size=1280,
                hidden_size=640,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        self.LSTM_1  = nn.LSTM(
                input_size=32,
                hidden_size=32,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        self.LSTM_2  = nn.LSTM(
                input_size=32,
                hidden_size=32,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        
        self.fc1 = nn.Linear(640, 32)
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.stem_v = Stem(out_dim=640) ## 特征提取器
        self.stem_t2 = Stem(out_dim=640)
        self.hyper2d_tv = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_tv = FFN(640,640*4)
        self.hyper2d_t2 = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_t2 = FFN(640,640*4)
        

    def forward(self,X_image_mask):
        # torch.autograd.set_detect_anomaly(True)
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_t2_liver,X_T2_image,liver_t2_mask,tumor_t2_mask,X_tv_liver,X_V_image,liver_tv_mask,tumor_tv_mask=X_image_mask
        b,f,c,h,w = X_V_image.shape
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image= X_V_mask.to(torch.float32),X_V_image.to(torch.float32), X_T2_mask.to(torch.float32),X_T2_image.to(torch.float32)
        X_tumor_v_1 = []
        # X_liver_t2_1 = []
        X_tumor_t2_1 = []
        X_V_mask = tumor_tv_mask.squeeze()
        X_T2_mask = tumor_t2_mask.squeeze()
        for i in range(f):
            # x_t2_liver_2d = X_t2_liver[:, i,:]
            x_t2_tumor_2d = X_T2_image[:, i,:]
            # x_tv_liver_2d = X_tv_liver[:, i,:]
            x_tv_tumor_2d = X_V_image[:, i,:]
            
            x_tumor_v =self.stem_v(x_tv_tumor_2d)
            x_tumor_t2 =self.stem_t2(x_t2_tumor_2d)
            
            X_tumor_v_1.append(x_tumor_v)
            # X_liver_v_1.append(x_liver_v)
            X_tumor_t2_1.append(x_tumor_t2)
            # X_liver_t2_1.append(x_liver_t2)
            
        # X_liver_v_1 = torch.stack(X_tumor_v_1, dim=1)
        X_tumor_v_1 = torch.stack(X_tumor_v_1, dim=1) 
        # X_liver_t2_1 = torch.stack(X_liver_t2_1, dim=1)
        X_tumor_t2_1 = torch.stack(X_tumor_t2_1, dim=1) 

        b, f, c, h, w = X_tumor_v_1.shape
        tumor_t2_mask = interpolate(X_T2_mask, size=[h, w])
        # liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(X_V_mask, size=[h, w])
        # liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])
        ## HGNN 融合
        _tmp_1 = X_tumor_v_1.clone()
        _tmp_1 = _tmp_1.view(b, c, -1, 1)# (batch_size, num_dims, num_points, 1) num_points = 5*h*w
        hyperedge_matrix, point_hyperedge_index, centers = construct_hyperedges(_tmp_1,num_clusters=25) 
        _tmp_1_1 = self.hyper2d_tv(_tmp_1,hyperedge_matrix, point_hyperedge_index, centers).view(b*f, -1, h, w)
        _tmp_1_1 = self.ffn_tv(_tmp_1_1).view(b,f,c,h,w)
        X_tumor_v_1 = _tmp_1_1.view(b,f,c,h,w) + X_tumor_v_1
        _tmp_2 = X_tumor_t2_1.clone()
        _tmp_2 = _tmp_2.view(b, c, -1, 1)# (batch_size, num_dims, num_points, 1) num_points = 5*h*w
        hyperedge_matrix, point_hyperedge_index, centers = construct_hyperedges(_tmp_2,num_clusters=25) 
        _tmp_2 = self.hyper2d_tv(_tmp_2,hyperedge_matrix, point_hyperedge_index, centers).view(b*f, -1, h, w)
        _tmp_2 = self.ffn_t2(_tmp_2).view(b,f,c,h,w)
        X_tumor_t2_1 =_tmp_2  +X_tumor_t2_1
        tumor_tv_mask_pool = self.pool(tumor_tv_mask).squeeze()
        tumor_t2_mask_pool = self.pool(tumor_t2_mask).squeeze()
        
        # # X_liver_v_gate = self.pool(X_liver_v_gate).squeeze()
        # # X_liver_t2_gate = self.pool(X_liver_t2_gate).squeeze()

        # X_liver_v_pool = self.pool(X_liver_v_4).squeeze()
        X_tumor_v_pool = self.pool(X_tumor_v_1).squeeze()
        # X_liver_t2_pool = self.pool(X_liver_t2_4).squeeze()
        X_tumor_t2_pool = self.pool(X_tumor_t2_1).squeeze()
        
        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        # X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        X_pool = torch.cat([ X_tumor_v_pool,  X_tumor_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        # x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool],dim=-1)

        # x = self.fc1(X_pool)

 
        x = self.fc1(x)
        
        # x = self.fc_mask(x)
        # x = F.relu(x,inplace=True)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        return x


class CNN3d_t2_tv_hgnn(nn.Module):
    def __init__(self,num_classes=1):
        super(CNN3d_t2_tv_hgnn, self).__init__()
        ## one stage
     
        self.liver_conv1_v =  Conv2dBlock(1, 16, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv1_t2 =  Conv2dBlock(1, 16, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')

        self.tumor_conv1_v =  Conv2dBlock(1, 16, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_v =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 =  Conv2dBlock(1, 16, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_t2 =  Conv2dBlock(16, 16, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')                                                 
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM  = nn.LSTM(
                input_size=1280,
                hidden_size=640,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        self.LSTM_1  = nn.LSTM(
                input_size=32,
                hidden_size=32,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        self.LSTM_2  = nn.LSTM(
                input_size=32,
                hidden_size=32,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        
        self.fc1 = nn.Linear(640, 32)
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.stem_v = Stem(out_dim=640) ## 特征提取器
        self.stem_t2 = Stem(out_dim=640)
        self.hyper2d_tv = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_tv = FFN(640,640*4)
        self.hyper2d_t2 = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_t2 = FFN(640,640*4)
        

    def forward(self,X_image_mask):
        # torch.autograd.set_detect_anomaly(True)
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_t2_liver,X_T2_image,liver_t2_mask,tumor_t2_mask,X_tv_liver,X_V_image,liver_tv_mask,tumor_tv_mask=X_image_mask
        b,f,c,h,w = X_V_image.shape
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image= X_V_mask.to(torch.float32),X_V_image.to(torch.float32), X_T2_mask.to(torch.float32),X_T2_image.to(torch.float32)
        X_tumor_v_1 = []
        # X_liver_t2_1 = []
        X_tumor_t2_1 = []
        X_V_mask = tumor_tv_mask.squeeze()
        X_T2_mask = tumor_t2_mask.squeeze()
        for i in range(f):
            # x_t2_liver_2d = X_t2_liver[:, i,:]
            x_t2_tumor_2d = X_T2_image[:, i,:]
            # x_tv_liver_2d = X_tv_liver[:, i,:]
            x_tv_tumor_2d = X_V_image[:, i,:]
            
            x_tumor_v =self.stem_v(x_tv_tumor_2d)
            x_tumor_t2 =self.stem_t2(x_t2_tumor_2d)
            
            X_tumor_v_1.append(x_tumor_v)
            # X_liver_v_1.append(x_liver_v)
            X_tumor_t2_1.append(x_tumor_t2)
            # X_liver_t2_1.append(x_liver_t2)
            
        X_tumor_v_1 = torch.stack(X_tumor_v_1, dim=1) 
        # X_liver_t2_1 = torch.stack(X_liver_t2_1, dim=1)
        X_tumor_t2_1 = torch.stack(X_tumor_t2_1, dim=1) 

        b, f, c, h, w = X_tumor_v_1.shape
        tumor_t2_mask = interpolate(X_T2_mask, size=[h, w])
        # liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(X_V_mask, size=[h, w])
        # liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])
        ## HGNN 融合
        _tmp_1 = X_tumor_v_1.clone()
        _tmp_1 = _tmp_1.view(b, c, -1, 1)# (batch_size, num_dims, num_points, 1) num_points = 5*h*w
        hyperedge_matrix, point_hyperedge_index, centers = construct_hyperedges(_tmp_1,num_clusters=25) 
        _tmp_1_1 = self.hyper2d_tv(_tmp_1,hyperedge_matrix, point_hyperedge_index, centers).view(b*f, -1, h, w)
        _tmp_1_1 = self.ffn_tv(_tmp_1_1).view(b,f,c,h,w)
        X_tumor_v_1 = _tmp_1_1.view(b,f,c,h,w) + X_tumor_v_1
        _tmp_2 = X_tumor_t2_1.clone()
        _tmp_2 = _tmp_2.view(b, c, -1, 1)# (batch_size, num_dims, num_points, 1) num_points = 5*h*w
        hyperedge_matrix, point_hyperedge_index, centers = construct_hyperedges(_tmp_2,num_clusters=25) 
        _tmp_2 = self.hyper2d_tv(_tmp_2,hyperedge_matrix, point_hyperedge_index, centers).view(b*f, -1, h, w)
        _tmp_2 = self.ffn_t2(_tmp_2).view(b,f,c,h,w)
        X_tumor_t2_1 =_tmp_2  +X_tumor_t2_1
        tumor_tv_mask_pool = self.pool(tumor_tv_mask).squeeze()
        tumor_t2_mask_pool = self.pool(tumor_t2_mask).squeeze()
        
        # # X_liver_v_gate = self.pool(X_liver_v_gate).squeeze()
        # # X_liver_t2_gate = self.pool(X_liver_t2_gate).squeeze()

        # X_liver_v_pool = self.pool(X_liver_v_4).squeeze()
        X_tumor_v_pool = self.pool(X_tumor_v_1).squeeze()
        # X_liver_t2_pool = self.pool(X_liver_t2_4).squeeze()
        X_tumor_t2_pool = self.pool(X_tumor_t2_1).squeeze()
        
        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        # X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        X_pool = torch.cat([ X_tumor_v_pool,  X_tumor_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        # x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool],dim=-1)

        # x = self.fc1(X_pool) 
        x = self.fc1(x)
        
        # x = self.fc_mask(x)
        # x = F.relu(x,inplace=True)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        return x

def batched_index_gather(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def batched_scatter(input, dim, index, src):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.scatter(input, dim, index, src)

import random
def mask_find_indices(mask):
    b, f, h, w = mask.shape
    device = mask.device
    # mask =mask.view(b*f,C,h,w)
    mask = mask.squeeze()
    if len(mask.shape)==3:
        mask=mask.unsqueeze(dim=0)
    mask = rearrange(mask, 'b f h w  -> b (f h w) ')
    ##从ｆｈｗ个位置中挑选局部特征
    indices = []
    length = []
    ## 可以设置一个固定的indice长度
    max_length = 0
    for i in range(b):
        indices.append(torch.where(mask[i,:]>mask[i,:].mean()))  ## 根据肿瘤的mask提供索引
        length.append(len(indices[-1][0]))

    max_length = max(length)
    oversampling_indices = []
    ## 过采样
    for i in range(b):
        if length[i]< max_length:
            support = max_length-length[i]
            if length[i]==0:
                ### 随机索引
                oversampling_indices.append(torch.LongTensor(random.sample(range(f*h*w), max_length)).to(device))
            else:    
                remainder = support % length[i]
                multi = int(support/length[i])
                media_result =  indices[i][0]
                for k  in range(multi):
                    media_result=torch.cat((media_result.to(device),indices[i][0].to(device)))
                if multi == 0:
                    # print(immediate[i][0][:2])
                    # print(immediate[i][0])
                    turple = (indices[i][0],indices[i][0][:remainder])
                    oversampling_indices.append(torch.cat(turple).to(device))
                else:
                    oversampling_indices.append(torch.cat((media_result,indices[i][0][:remainder])).to(device))
        else:
            oversampling_indices.append(indices[i][0].to(device))
  

    oversampling_indices = torch.stack(oversampling_indices,dim=0)


    return oversampling_indices

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_new_coordinates(contour, center_x, center_y, ratio,x_max =112,y_max=112):
    paixue = False
    new_coordinates = set()
    for i in range(len(contour)):
        new_x = int(round((contour[i][0][0] - center_x) * ratio) + center_x)
        new_y = int(round((contour[i][0][1] - center_y) * ratio) + center_y)
        if ((new_x+3)<x_max and (new_y+3)<y_max) and ((new_x-3)>0 and (new_y-3)>0) :
            new_coordinates.add((new_x,new_y))
        else:
            new_x = int(round((contour[i][0][0] - center_x) * 0.9) + center_x)
            new_y = int(round((contour[i][0][1] - center_y) * 0.9) + center_y)
            if ((new_x+3)<x_max and (new_y+3)<y_max) and ((new_x-3)>0 and (new_y-3)>0):
                new_coordinates.add((new_x,new_y))
            else:                
                new_coordinates.add((center_x,center_y))
    ##是否需要排序
    if paixue:
        return np.array(sorted(list(new_coordinates),key=lambda x: x[1])) ## 集合转list，然后再根据x坐标排序
    else:
        return np.array(list(new_coordinates))

def sorted_coordinates_sample(sorted_coordinates,desired_samples,yushu=0):
    if len(sorted_coordinates) < desired_samples:
        resampled_coordinates = random.choices(sorted_coordinates, k=desired_samples)
        if len(np.array(resampled_coordinates)) != desired_samples:
            print("错误")
        return np.array(resampled_coordinates)
    else:
        # 计算每个区间应该采样的点数
        interval_size = len(sorted_coordinates) // desired_samples
        resampled_coordinates = [sorted_coordinates[i] for i in range(0, len(sorted_coordinates), interval_size)]
        if len(resampled_coordinates)>=desired_samples:
            if desired_samples % 2==0 :
                if len(np.array(resampled_coordinates[:int(desired_samples/2)]+resampled_coordinates[-int(desired_samples/2):]))!=desired_samples:
                    print("错误")
                return np.array(resampled_coordinates[:int(desired_samples/2)]+resampled_coordinates[-int(desired_samples/2):])
            else:
                if len(np.array(resampled_coordinates[:int(desired_samples/2)+1]+resampled_coordinates[-int(desired_samples/2):]))!=desired_samples:
                    print("错误")
                return np.array(resampled_coordinates[:int(desired_samples/2)+1]+resampled_coordinates[-int(desired_samples/2):])                        
        else:
            if len(np.array(resampled_coordinates))!= desired_samples:
                print("错误")
            
            return np.array(resampled_coordinates)
        
def convert_to_one_dimensional(new_coordinates, f, h, w):
    # 获取坐标数组的行数（坐标点的数量）
    k = new_coordinates.shape[0]
    # 初始化一维坐标数组
    one_dimensional_coordinates = np.zeros((k, 1), dtype=int)
    # 将二维坐标转换为一维坐标
    for i in range(k):
        y, x = new_coordinates[i]  # 获取二维坐标的行索引和列索引
        one_dimensional_coordinates[i] = f*h*w + y * w + x  # 使用公式计算一维坐标
    return one_dimensional_coordinates

def top_feature_select(tumor_tv_mask,top_k=1000,liver_tv_mask=None):
    """
    top_k 节点数
    """
    b, f, h_image, w_image = tumor_tv_mask.shape
    device = tumor_tv_mask.device
    mode = cv2.RETR_EXTERNAL  # 例如，选择提取外部轮廓
    b_new_coordinates_25_one_frame_all = []
    b_new_coordinates_50_one_frame_all = []
    b_new_coordinates_75_one_frame_all = []
    b_new_coordinates_100_one_frame_all = []
    b_new_coordinates_125_one_frame_all = []
    
    
    for i in range(b):
        new_coordinates_25_one_frame_all = []
        new_coordinates_50_one_frame_all = []
        new_coordinates_75_one_frame_all = []
        new_coordinates_100_one_frame_all = [] ## 已经是最外面的轮廓了
        new_coordinates_125_one_frame_all = []
        # if i==3:
        layer_sum = 0
        for j in range(f):
            mask_f = tumor_tv_mask[i,j]
            if mask_f.sum()>0:
                layer_sum =layer_sum + 1                
                liver_mask = False
        if layer_sum != 0:  
            sample_count = math.floor(top_k/layer_sum)
            # yushu = 0
            # if top_k % layer_sum != 0:
            yushu = top_k % layer_sum
                # 使用findContours函数查找轮廓  
        if layer_sum == 0:
            # print("false")
            liver_mask = True
            ## 肿瘤被裁掉了
            for j in range(f):
                mask_f = liver_tv_mask[i,j]
                if mask_f.sum()>0:
                    layer_sum = layer_sum + 1  

            sample_count = math.floor(top_k/layer_sum)
            
            yushu = top_k % layer_sum
        
        ## 最外面的轮廓扩一圈 
        for j in range(f):     
            if liver_mask:  
                mask_f = liver_tv_mask[i,j] 
            else: 
                mask_f = tumor_tv_mask[i,j] 

            if mask_f.sum()>0: 
            
                center_point = []
                new_coordinates_25_one_frame = []
                new_coordinates_50_one_frame = []
                new_coordinates_75_one_frame = []
                new_coordinates_100_one_frame = [] ## 已经是最外面的轮廓了
                new_coordinates_125_one_frame = [] ## 最外面的轮廓扩一圈                          
                contours, _ = cv2.findContours(mask_f.detach().cpu().numpy().astype(np.uint8), mode, cv2.CHAIN_APPROX_SIMPLE)
                for i_counter, contour in enumerate(contours):
                    # 计算边界框
                    ## 调试时使用
                    contour = contours[i_counter]
                    x, y, w, h = cv2.boundingRect(contour)
                    contour_points = contour.squeeze(axis=1)
                    # 计算中心坐标
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    distances = [distance((x, y), (center_x, center_y)) for x, y in contour_points]
                    total_distance = np.sum(distances)
                    center_point.append((center_x,center_y)) #自己加的把中心点的坐标记录
                    new_coordinates = []
                    # for ratio in [0.25, 0.5, 0.75]:
                    new_coordinates_25 = calculate_new_coordinates(contour, center_x, center_y, 0.25)
                    new_coordinates_50 = calculate_new_coordinates(contour, center_x, center_y, 0.5)
                    new_coordinates_75 = calculate_new_coordinates(contour, center_x, center_y, 0.75)
                    new_coordinates_100 = calculate_new_coordinates(contour, center_x, center_y, 1.0)
                    new_coordinates_125 = calculate_new_coordinates(contour, center_x, center_y, 1.25,x_max=(mask_f.shape[0]-10),y_max=(mask_f.shape[1]-10))
                    new_coordinates_25_one_frame.append(new_coordinates_25)
                    new_coordinates_50_one_frame.append(new_coordinates_50)
                    new_coordinates_75_one_frame.append(new_coordinates_75)
                    new_coordinates_100_one_frame.append(new_coordinates_100)
                    new_coordinates_125_one_frame.append(new_coordinates_125)
                    
                    
                ## 每一帧的坐标提取完到一个固定的数之后            
                new_coordinates_25_one_frame=np.concatenate(new_coordinates_25_one_frame)
                new_coordinates_50_one_frame=np.concatenate(new_coordinates_50_one_frame)
                new_coordinates_75_one_frame=np.concatenate(new_coordinates_75_one_frame)
                new_coordinates_100_one_frame=np.concatenate(new_coordinates_100_one_frame)
                new_coordinates_125_one_frame=np.concatenate(new_coordinates_125_one_frame)

                ## 处理余数的情况
                if j!=round(f/2):
                    new_coordinates_25_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_25_one_frame,sample_count),j,h_image,w_image ) ##   sample_count *1           
                    new_coordinates_50_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_50_one_frame,sample_count),j,h_image,w_image )  
                    new_coordinates_75_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_75_one_frame,sample_count),j,h_image,w_image )  
                    new_coordinates_100_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_100_one_frame,sample_count),j,h_image,w_image )  
                    new_coordinates_125_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_125_one_frame,sample_count),j,h_image,w_image )  
                    if len(new_coordinates_25_one_frame_j)!=sample_count:
                        print("false patient")
                    new_coordinates_25_one_frame_all.append(torch.tensor(new_coordinates_25_one_frame_j))
                    new_coordinates_50_one_frame_all.append(torch.tensor(new_coordinates_50_one_frame_j))
                    new_coordinates_75_one_frame_all.append(torch.tensor(new_coordinates_75_one_frame_j))
                    new_coordinates_100_one_frame_all.append(torch.tensor(new_coordinates_100_one_frame_j))
                    new_coordinates_125_one_frame_all.append(torch.tensor(new_coordinates_125_one_frame_j))
                else:
                    new_coordinates_25_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_25_one_frame,(sample_count+yushu)),j,h_image,w_image )                 
                    new_coordinates_50_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_50_one_frame,(sample_count+yushu)),j,h_image,w_image )
                    new_coordinates_75_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_75_one_frame,(sample_count+yushu)),j,h_image,w_image )
                    new_coordinates_100_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_100_one_frame,(sample_count+yushu)),j,h_image,w_image )
                    new_coordinates_125_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_125_one_frame,(sample_count+yushu)),j,h_image,w_image )
                    if len(new_coordinates_25_one_frame_j)<(sample_count+yushu):
                        print("false patient")
                    if len(new_coordinates_50_one_frame_j)<(sample_count+yushu):
                        print("false patient")
                    if len(new_coordinates_75_one_frame_j)<(sample_count+yushu):
                        print("false patient")
                    new_coordinates_25_one_frame_all.append(torch.tensor(new_coordinates_25_one_frame_j))
                    new_coordinates_50_one_frame_all.append(torch.tensor(new_coordinates_50_one_frame_j))
                    new_coordinates_75_one_frame_all.append(torch.tensor(new_coordinates_75_one_frame_j))
                    new_coordinates_100_one_frame_all.append(torch.tensor(new_coordinates_100_one_frame_j))
                    new_coordinates_125_one_frame_all.append(torch.tensor(new_coordinates_125_one_frame_j))
                
        new_coordinates_25_one_frame_all = torch.cat(new_coordinates_25_one_frame_all)
        new_coordinates_50_one_frame_all = torch.cat(new_coordinates_50_one_frame_all)
        new_coordinates_75_one_frame_all = torch.cat(new_coordinates_75_one_frame_all)
        new_coordinates_100_one_frame_all = torch.cat(new_coordinates_100_one_frame_all)
        new_coordinates_125_one_frame_all = torch.cat(new_coordinates_125_one_frame_all)
        if len(new_coordinates_25_one_frame_all)!=top_k:
            # print("错误患者")
            new_coordinates_25_one_frame_all =torch.cat([new_coordinates_25_one_frame_all,new_coordinates_25_one_frame_all[:top_k-len(new_coordinates_25_one_frame_all)]])
        if len(new_coordinates_50_one_frame_all)!=top_k:
            # print("错误患者")
            new_coordinates_50_one_frame_all =torch.cat([new_coordinates_50_one_frame_all,new_coordinates_50_one_frame_all[:top_k-len(new_coordinates_50_one_frame_all)]])
        if len(new_coordinates_75_one_frame_all)!=top_k:
            # print("错误患者")
            new_coordinates_75_one_frame_all =torch.cat([new_coordinates_75_one_frame_all,new_coordinates_75_one_frame_all[:top_k-len(new_coordinates_75_one_frame_all)]])
        if len(new_coordinates_100_one_frame_all)!=top_k:
            # print("错误患者")
            new_coordinates_100_one_frame_all =torch.cat([new_coordinates_100_one_frame_all,new_coordinates_100_one_frame_all[:top_k-len(new_coordinates_100_one_frame_all)]])
        if len(new_coordinates_125_one_frame_all)!=top_k:
            # print("错误患者")
            new_coordinates_125_one_frame_all =torch.cat([new_coordinates_125_one_frame_all,new_coordinates_125_one_frame_all[:top_k-len(new_coordinates_125_one_frame_all)]])
        

        b_new_coordinates_25_one_frame_all.append(new_coordinates_25_one_frame_all)
        b_new_coordinates_50_one_frame_all.append(new_coordinates_50_one_frame_all)
        b_new_coordinates_75_one_frame_all.append(new_coordinates_75_one_frame_all)
        b_new_coordinates_100_one_frame_all.append(new_coordinates_100_one_frame_all)
        b_new_coordinates_125_one_frame_all.append(new_coordinates_125_one_frame_all)
    
    b_new_coordinates_25_one_frame_all=torch.stack(b_new_coordinates_25_one_frame_all,dim=0).squeeze().to(device)
    b_new_coordinates_50_one_frame_all=torch.stack(b_new_coordinates_50_one_frame_all,dim=0).squeeze().to(device)
    b_new_coordinates_75_one_frame_all=torch.stack(b_new_coordinates_75_one_frame_all,dim=0).squeeze().to(device)
    b_new_coordinates_100_one_frame_all=torch.stack(b_new_coordinates_100_one_frame_all,dim=0).squeeze().to(device)
    b_new_coordinates_125_one_frame_all=torch.stack(b_new_coordinates_125_one_frame_all,dim=0).squeeze().to(device)
    return [b_new_coordinates_25_one_frame_all,b_new_coordinates_50_one_frame_all,b_new_coordinates_75_one_frame_all,b_new_coordinates_100_one_frame_all,b_new_coordinates_125_one_frame_all]

def top_feature_select_3_layer(tumor_tv_mask,top_k=1000,liver_tv_mask=None):
    """
    top_k 节点数
    """
    b, f, h_image, w_image = tumor_tv_mask.shape
    device = tumor_tv_mask.device
    mode = cv2.RETR_EXTERNAL  # 例如，选择提取外部轮廓
    b_new_coordinates_25_one_frame_all = []
    b_new_coordinates_50_one_frame_all = []
    b_new_coordinates_75_one_frame_all = []
    
    for i in range(b):
        new_coordinates_25_one_frame_all = []
        new_coordinates_50_one_frame_all = []
        new_coordinates_75_one_frame_all = []

        # if i==3:
        layer_sum = 0
        for j in range(f):
            mask_f = tumor_tv_mask[i,j]
            if mask_f.sum()>0:
                layer_sum =layer_sum + 1                
                liver_mask = False
        if layer_sum != 0:  
            sample_count = math.floor(top_k/layer_sum)
            # yushu = 0
            # if top_k % layer_sum != 0:
            yushu = top_k % layer_sum
                # 使用findContours函数查找轮廓  
        if layer_sum == 0:
            # print("false")
            liver_mask = True
            ## 肿瘤被裁掉了
            for j in range(f):
                mask_f = liver_tv_mask[i,j]
                if mask_f.sum()>0:
                    layer_sum = layer_sum + 1  
            sample_count = math.floor(top_k/layer_sum)            
            yushu = top_k % layer_sum
        
        ## 最外面的轮廓扩一圈 
        for j in range(f):     
            if liver_mask:  
                mask_f = liver_tv_mask[i,j] 
            else: 
                mask_f = tumor_tv_mask[i,j] 

            if mask_f.sum()>0:             
                center_point = []
                new_coordinates_25_one_frame = []
                new_coordinates_50_one_frame = []
                new_coordinates_75_one_frame = []
                      
                contours, _ = cv2.findContours(mask_f.detach().cpu().numpy().astype(np.uint8), mode, cv2.CHAIN_APPROX_SIMPLE)
                for i_counter, contour in enumerate(contours):
                    # 计算边界框
                    ## 调试时使用
                    contour = contours[i_counter]
                    x, y, w, h = cv2.boundingRect(contour)
                    contour_points = contour.squeeze(axis=1)
                    # 计算中心坐标
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    distances = [distance((x, y), (center_x, center_y)) for x, y in contour_points]
                    total_distance = np.sum(distances)
                    center_point.append((center_x,center_y)) #自己加的把中心点的坐标记录
                    new_coordinates = []
                    # for ratio in [0.25, 0.5, 0.75]:
                    new_coordinates_25 = calculate_new_coordinates(contour, center_x, center_y, 0.1)
                    new_coordinates_50 = calculate_new_coordinates(contour, center_x, center_y, 0.4)
                    new_coordinates_75 = calculate_new_coordinates(contour, center_x, center_y, 0.9)
                    
                    new_coordinates_25_one_frame.append(new_coordinates_25)
                    new_coordinates_50_one_frame.append(new_coordinates_50)
                    new_coordinates_75_one_frame.append(new_coordinates_75)
                                        
                ## 每一帧的坐标提取完到一个固定的数之后            
                new_coordinates_25_one_frame=np.concatenate(new_coordinates_25_one_frame)
                new_coordinates_50_one_frame=np.concatenate(new_coordinates_50_one_frame)
                new_coordinates_75_one_frame=np.concatenate(new_coordinates_75_one_frame)

                ## 处理余数的情况
                if j!=round(f/2):
                    new_coordinates_25_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_25_one_frame,sample_count),j,h_image,w_image ) ##   sample_count *1           
                    new_coordinates_50_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_50_one_frame,sample_count),j,h_image,w_image )  
                    new_coordinates_75_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_75_one_frame,sample_count),j,h_image,w_image )  
                    
                    if len(new_coordinates_25_one_frame_j)!=sample_count:
                        print("false patient")
                    new_coordinates_25_one_frame_all.append(torch.tensor(new_coordinates_25_one_frame_j))
                    new_coordinates_50_one_frame_all.append(torch.tensor(new_coordinates_50_one_frame_j))
                    new_coordinates_75_one_frame_all.append(torch.tensor(new_coordinates_75_one_frame_j))
                   
                else:
                    new_coordinates_25_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_25_one_frame,(sample_count+yushu)),j,h_image,w_image )                 
                    new_coordinates_50_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_50_one_frame,(sample_count+yushu)),j,h_image,w_image )
                    new_coordinates_75_one_frame_j = convert_to_one_dimensional(sorted_coordinates_sample(new_coordinates_75_one_frame,(sample_count+yushu)),j,h_image,w_image )
                    
                    if len(new_coordinates_25_one_frame_j)<(sample_count+yushu):
                        print("false patient")
                    if len(new_coordinates_50_one_frame_j)<(sample_count+yushu):
                        print("false patient")
                    if len(new_coordinates_75_one_frame_j)<(sample_count+yushu):
                        print("false patient")
                    new_coordinates_25_one_frame_all.append(torch.tensor(new_coordinates_25_one_frame_j))
                    new_coordinates_50_one_frame_all.append(torch.tensor(new_coordinates_50_one_frame_j))
                    new_coordinates_75_one_frame_all.append(torch.tensor(new_coordinates_75_one_frame_j))
                
        new_coordinates_25_one_frame_all = torch.cat(new_coordinates_25_one_frame_all)
        new_coordinates_50_one_frame_all = torch.cat(new_coordinates_50_one_frame_all)
        new_coordinates_75_one_frame_all = torch.cat(new_coordinates_75_one_frame_all)

        if len(new_coordinates_25_one_frame_all)!=top_k:
            # print("错误患者")
            new_coordinates_25_one_frame_all =torch.cat([new_coordinates_25_one_frame_all,new_coordinates_25_one_frame_all[:top_k-len(new_coordinates_25_one_frame_all)]])
        if len(new_coordinates_50_one_frame_all)!=top_k:
            # print("错误患者")
            new_coordinates_50_one_frame_all =torch.cat([new_coordinates_50_one_frame_all,new_coordinates_50_one_frame_all[:top_k-len(new_coordinates_50_one_frame_all)]])
        if len(new_coordinates_75_one_frame_all)!=top_k:
            # print("错误患者")
            new_coordinates_75_one_frame_all =torch.cat([new_coordinates_75_one_frame_all,new_coordinates_75_one_frame_all[:top_k-len(new_coordinates_75_one_frame_all)]])
        

        b_new_coordinates_25_one_frame_all.append(new_coordinates_25_one_frame_all)
        b_new_coordinates_50_one_frame_all.append(new_coordinates_50_one_frame_all)
        b_new_coordinates_75_one_frame_all.append(new_coordinates_75_one_frame_all)
        
    
    b_new_coordinates_25_one_frame_all=torch.stack(b_new_coordinates_25_one_frame_all,dim=0).squeeze().to(device)
    b_new_coordinates_50_one_frame_all=torch.stack(b_new_coordinates_50_one_frame_all,dim=0).squeeze().to(device)
    b_new_coordinates_75_one_frame_all=torch.stack(b_new_coordinates_75_one_frame_all,dim=0).squeeze().to(device)
    
    return [b_new_coordinates_25_one_frame_all,b_new_coordinates_50_one_frame_all,b_new_coordinates_75_one_frame_all]


def multi_tumor_top_fusion(feature_25_v,feature_25_t2,b,A_25,top_k,num_knn,HGNN25,graph_constructor):
    b,c,n = feature_25_v.shape
    device = feature_25_v.device

    combine_feature_25 = torch.cat([feature_25_v,feature_25_t2],dim=2)    
    edge_index = graph_constructor(combine_feature_25)
    combine_feater_all_25 = HGNN25(combine_feature_25,edge_index)    
    select_feature_25_v= combine_feater_all_25[:, :, :combine_feater_all_25.shape[2]//2]  # 前半部分
    select_feature_20_t2 = combine_feater_all_25[:, :, combine_feater_all_25.shape[2]//2:]  # 后半部分
    return select_feature_25_v.squeeze(),select_feature_20_t2.squeeze()


def multi_tumor_top_fusion_v2(feature_25_v,feature_25_t2,b,A_25,top_k,num_knn,HGNN25,graph_constructor):
    b,c,n = feature_25_v.shape
    device = feature_25_v.device

    combine_feature_25 = torch.cat([feature_25_v,feature_25_t2],dim=2).unsqueeze(-1)    
    edge_index = graph_constructor(combine_feature_25)
    combine_feater_all_25 = HGNN25(combine_feature_25,edge_index)    
    select_feature_25_v= combine_feater_all_25[:, :, :combine_feater_all_25.shape[2]//2]  # 前半部分
    select_feature_20_t2 = combine_feater_all_25[:, :, combine_feater_all_25.shape[2]//2:]  # 后半部分
    return select_feature_25_v.squeeze(),select_feature_20_t2.squeeze()



## 肿瘤内部融合
class hgnn_fusion1(nn.Module):
    ## 在空间上挑选并进行
    def __init__(self,points=1000,in_channels=16,out_channels=16,top_k =1000):
        super(hgnn_fusion1,self).__init__()
        # self.HGNN1 = HGNNConv(in_channels=in_channels, out_channels=out_channels)
        # self.HGNN1 = HypergraphConv2d(in_channels=in_channels, out_channels=out_channels)
        self.layers1 = EdgeConv2d(in_channels=in_channels, out_channels=out_channels)
        self.A =Hypergraph(points)
        self.top_k = top_k
        self.w0 = nn.Parameter(torch.ones(2))
        self.graph_constructor = DenseDilatedKnnGraph(k=top_k, dilation=1, stochastic=False, epsilon=0.0)
    def forward(self, X_tumor_v_1):
        batch_size, frame, channel, h, w = X_tumor_v_1.size()
        device = X_tumor_v_1.device
        # Step 1: Sum along the channel dimension
        summed_features = X_tumor_v_1.sum(dim=2)  # Shape: batch_size * frame * h * w

        # Step 2: Find top k indices along the summed dimension
        _, top_indices = torch.topk(summed_features.view(batch_size, -1), self.top_k, dim=1) ## top_indices b,k

        x_tumor_position = rearrange(X_tumor_v_1, 'b f c h w  -> b c (f h w)')
        selected_features = batched_index_gather(x_tumor_position, dim=2, index=top_indices) ## 

        edge_index = self.graph_constructor(selected_features)
        selected_features_new = self.layers1(selected_features,edge_index)

        ## gcn 聚合 ## batch_size, num_dims, num_points, 1
        # selected_features = selected_features.unsqueeze(-1)
        # hyperedge_matrix, point_hyperedge_index, centers = construct_hyperedges(selected_features,num_clusters=28)
        # selected_features_new = self.HGNN1(selected_features,hyperedge_matrix, point_hyperedge_index, centers)
        """
        x_tumor_all = []
        for i in range(batch_size): ##因为掉包不能进行batchsize批处理
            selected_feature=selected_features[i].view(self.top_k,channel)
            # A = Hypergraph(1000).to(device)
            A = self.A.from_feature_kNN(selected_feature.detach(),k=10)
            # layers_1 = HGNNConv(in_channels=16, out_channels=16).to(device)
            x_tumor_position_i = self.HGNN1(selected_feature,A)
            x_tumor_all.append(x_tumor_position_i)
        x_tumor_all = torch.stack(x_tumor_all,dim=0).to(device).permute(0,2,1)
        x_tumor_position_2 = batched_scatter(x_tumor_position,dim=2,index = top_indices,src = x_tumor_all)
        """
        w1 = torch.exp(self.w0[0])/torch.sum(torch.exp(self.w0))
        w2 = torch.exp(self.w0[1])/torch.sum(torch.exp(self.w0))
        selected_features = w1*selected_features.squeeze() + w2*selected_features_new.squeeze()
        x_tumor_position_2 = batched_scatter(x_tumor_position,dim=2,index = top_indices,src = selected_features)
        # liver_feature_fused = x_liver_position + batched_scatter(x_liver_position, dim=2, index=tumor_indices, src=tumor_select)
        x_tumor = x_tumor_position_2.view(batch_size, frame, channel, h, w )
        return x_tumor
    
## 肿瘤肝脏融合
class hgnn_fusion_liver_tumor(nn.Module):
    ## 在空间上挑选并进行
    def __init__(self,points=2000,in_channels=16,out_channels=16,top_k =1000):
        super(hgnn_fusion_liver_tumor,self).__init__()
        # self.HGNN1 = HGNNConv(in_channels=in_channels, out_channels=out_channels)
        # self.HGNN1 = HypergraphConv2d(in_channels=in_channels, out_channels=out_channels)
        self.A =Hypergraph(points)
        self.top_k = top_k
        self.w0 = nn.Parameter(torch.ones(2))
        self.w1 = nn.Parameter(torch.ones(2))
        # self.nn = nn.Conv3d(in_channels=in_channels,out_channels=out_channels)
        self.layers1 = EdgeConv2d(in_channels=in_channels, out_channels=out_channels)
        self.graph_constructor = DenseDilatedKnnGraph(k=top_k, dilation=1, stochastic=False, epsilon=0.0)
        ## topk操作需要被替换，这个操作很耗时
    def forward(self,x_tumor,x_liver):
        '''
        x_liver: b,f,c,h,w
        x_tumor: b,f,c,h,w
        liver_mask: b,f,h,w
        tumor_mask: b,f,h,w
        ''' 
        b, f, c, h, w = x_liver.shape
        device = x_liver.device
        ##  得到tumor—feature  这次不需要按照mask提取特征
        # tumor_indices = mask_find_indices(tumor_mask)  ## b
        x_tumor_position = rearrange(x_tumor, 'b f c h w  -> b c (f h w)')
        # tumor_feature = batched_index_gather(x_tumor_position, dim=2, index=tumor_indices)
        
        summed_features = x_tumor.sum(dim=2)  # Shape: batch_size * frame * h * w
        _, tumor_top_indices = torch.topk(summed_features.view(b, -1), self.top_k, dim=1) ## top_indices b,k      
        tumor_select = batched_index_gather(x_tumor_position, dim=2, index=tumor_top_indices) ## 

        x_liver_position = rearrange(x_liver, 'b f c h w  -> b c (f h w)')
        liver_summed_features = x_liver.sum(dim=2)  # Shape: batch_size * frame * h * w
        _, liver_top_indices = torch.topk(liver_summed_features.view(b, -1), self.top_k, dim=1) ## top_indices b,k      
        liver_select = batched_index_gather(x_liver_position, dim=2, index=liver_top_indices) ## 

        ## 计算 两种特征的相似度
        liver_tumor_postion_map = torch.matmul(liver_select.permute(0, 2, 1), tumor_select)
        _, liver_sim_index = torch.topk(liver_tumor_postion_map, dim=1, k=1)
        _, tumor_sim_index = torch.topk(liver_tumor_postion_map, dim=2, k=1)
        tumor_sim_index = tumor_sim_index.permute(0, 2, 1)

        liver_select_feature = batched_index_gather(liver_select, dim=2, index=liver_sim_index)
        tumor_select_feature = batched_index_gather(tumor_select, dim=2, index=tumor_sim_index)
        combine_feature = torch.cat([liver_select_feature,tumor_select_feature],dim=2)
        
        combine_feature = combine_feature.unsqueeze(-1)
        # hyperedge_matrix, point_hyperedge_index, centers = construct_hyperedges(combine_feature,num_clusters=28)
        # combine_feature_new = self.HGNN1(combine_feature,hyperedge_matrix, point_hyperedge_index, centers)

        edge_index = self.graph_constructor(combine_feature)
        combine_feature_new = self.layers1(combine_feature,edge_index)
        # combine_feater_all = torch.stack(combine_feature_new,dim=0).to(device).permute(0,2,1)
        liver_select_feature_new = combine_feature_new[:, :, :combine_feature_new.shape[2]//2]  # 前半部分
        tumor_select_feature_new = combine_feature_new[:, :, combine_feature_new.shape[2]//2:]  # 后半部分
        """
        combine_feater_all =[]
        for i in range(b): ##因为掉包不能进行batchsize批处理
            selected_feature=combine_feature[i].view(2*self.top_k,c)
            # A = Hypergraph(1000).to(device)
            A = self.A.from_feature_kNN(selected_feature.detach(),k=10)
            # layers_1 = HGNNConv(in_channels=16, out_channels=16).to(device)
            x_tumor_position_i = self.HGNN1(selected_feature,A)
            combine_feater_all.append(x_tumor_position_i)
        ##  得到liver-feature
        combine_feater_all = torch.stack(combine_feater_all,dim=0).to(device).permute(0,2,1)
        liver_select_feature = combine_feater_all[:,  self.Aconv:, :combine_feater_all.shape[2]//2]  # 前半部分
        tumor_select_feature = combine_feater_all[:, :, combine_feater_all.shape[2]//2:]  # 后半部分
        """
        w1 = torch.exp(self.w0[0])/torch.sum(torch.exp(self.w0))
        w2 = torch.exp(self.w0[1])/torch.sum(torch.exp(self.w0))
        liver_select_feature = w1*liver_select_feature_new.squeeze() + w2*liver_select_feature

        w3 = torch.exp(self.w1[0])/torch.sum(torch.exp(self.w1))
        w4 = torch.exp(self.w1[1])/torch.sum(torch.exp(self.w1))
        tumor_select_feature = w3*tumor_select_feature_new.squeeze() + w4*tumor_select_feature

        x_tumor_position = batched_scatter(x_tumor_position,dim=2,index = tumor_top_indices,src = tumor_select_feature)
        x_liver_position = batched_scatter(x_liver_position,dim=2,index = liver_top_indices,src = liver_select_feature)
        
        x_liver_position = x_liver_position.view(b, f, c, h, w )
        x_tumor_position = x_tumor_position.view(b, f, c, h, w )
        # x_tumor = x_tumor + x_tumor_position
        # x_liver = x_liver + x_liver_position
        
        return x_tumor,x_liver

## 模态之间 空间融合
class hgnn_fusion_space(nn.Module):
    ## 在空间上挑选并进行
    def __init__(self,points=2000,in_channels=16,out_channels=16,top_k =100,num_knn=10):
        super(hgnn_fusion_space,self).__init__()
        self.HGNN25 = EdgeConv2d(in_channels=in_channels, out_channels=out_channels)
        self.HGNN50 = EdgeConv2d(in_channels=in_channels, out_channels=out_channels)
        self.HGNN75 = EdgeConv2d(in_channels=in_channels, out_channels=out_channels)
        self.HGNN100 = EdgeConv2d(in_channels=in_channels, out_channels=out_channels)
        self.HGNN125 = EdgeConv2d(in_channels=in_channels, out_channels=out_channels)
        
        self.A_25 =Hypergraph(points)
        self.A_50 =Hypergraph(points)
        self.A_75 =Hypergraph(points)
        self.A_100 =Hypergraph(points)
        self.A_125 =Hypergraph(points)
        
        self.top_k = top_k
        self.num_knn =num_knn
        self.graph_constructor = DenseDilatedKnnGraph(k=top_k, dilation=1, stochastic=False, epsilon=0.0)


    def forward(self,X_tumor_v,X_tumor_t2,tumor_tv_mask,tumor_t2_mask,liver_tv_mask,liver_t2_mask):
        '''
        X_tumor_v: b,f,c,h,w
        X_tumor_t2: b,f,c,h,w
        tumor_tv_mask: b,f,h,w
        tumor_t2_mask: b,f,h,w
        '''
        b, f, c, h_image, w_image = X_tumor_v.shape
        device = X_tumor_v.device
         ## 寻找轮廓位置
        mode = cv2.RETR_EXTERNAL  # 例如，选择提取外部轮廓
        x_tumor_position_v = rearrange(X_tumor_v, 'b f c h w  -> b c (f h w)')
        x_tumor_position_t2 = rearrange(X_tumor_t2, 'b f c h w  -> b c (f h w)')
        
        ## 首先要对非空的mask进行统计，然后对非空的mask层数进行采样
        """
        b_new_coordinates_cor_all包含
        b_new_coordinates_25_one_frame_all 
        b_new_coordinates_50_one_frame_all 
        b_new_coordinates_75_one_frame_all 
        b_new_coordinates_100_one_frame_all 
        b_new_coordinates_125_one_frame_all 
        """
        b_coordinates_cor_all_v=top_feature_select(tumor_tv_mask,top_k=self.top_k,liver_tv_mask=liver_tv_mask)

        b_coordinates_cor_all_t2=top_feature_select(tumor_t2_mask,top_k=self.top_k,liver_tv_mask=liver_t2_mask)
       
        feature_25_v = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_v[0])
        feature_50_v = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_v[1])
        feature_75_v = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_v[2])
        feature_100_v = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_v[3])
        feature_125_v = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_v[4])
        
        feature_25_t2 = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_t2[0])
        feature_50_t2 = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_t2[1])
        feature_75_t2 = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_t2[2])
        feature_100_t2 = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_t2[3])
        feature_125_t2 = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_t2[4])

        ##  得到tumor—feature  这次不需要按照mask提取特征
        # tumor_indices = mask_find_indices(tumor_mask)  ## b
        select_feature_25_v,select_feature_25_t2=multi_tumor_top_fusion(feature_25_v,feature_25_t2,b,self.A_25,self.top_k,self.num_knn,self.HGNN25,self.graph_constructor)
        x_tumor_position_v = batched_scatter(x_tumor_position_v,dim=2,index = b_coordinates_cor_all_v[0],src = select_feature_25_v)
        x_tumor_position_t2 = batched_scatter(x_tumor_position_t2,dim=2,index = b_coordinates_cor_all_t2[0],src = select_feature_25_t2)

        select_feature_50_v,select_feature_50_t2=multi_tumor_top_fusion(feature_50_v,feature_50_t2,b,self.A_50,self.top_k,self.num_knn,self.HGNN50,self.graph_constructor)
        x_tumor_position_v = batched_scatter(x_tumor_position_v,dim=2,index = b_coordinates_cor_all_v[0],src = select_feature_50_v)
        x_tumor_position_t2 = batched_scatter(x_tumor_position_t2,dim=2,index = b_coordinates_cor_all_t2[0],src = select_feature_50_t2)

        select_feature_75_v,select_feature_75_t2=multi_tumor_top_fusion(feature_75_v,feature_75_t2,b,self.A_75,self.top_k,self.num_knn,self.HGNN75,self.graph_constructor)
        x_tumor_position_v = batched_scatter(x_tumor_position_v,dim=2,index = b_coordinates_cor_all_v[0],src = select_feature_75_v)
        x_tumor_position_t2 = batched_scatter(x_tumor_position_t2,dim=2,index = b_coordinates_cor_all_t2[0],src = select_feature_75_t2)

        select_feature_100_v,select_feature_100_t2=multi_tumor_top_fusion(feature_100_v,feature_100_t2,b,self.A_100,self.top_k,self.num_knn,self.HGNN100,self.graph_constructor)
        x_tumor_position_v = batched_scatter(x_tumor_position_v,dim=2,index = b_coordinates_cor_all_v[0],src = select_feature_100_v)
        x_tumor_position_t2 = batched_scatter(x_tumor_position_t2,dim=2,index = b_coordinates_cor_all_t2[0],src = select_feature_100_t2)

        select_feature_125_v,select_feature_125_t2=multi_tumor_top_fusion(feature_125_v,feature_125_t2,b,self.A_125,self.top_k,self.num_knn,self.HGNN125,self.graph_constructor)
        x_tumor_position_v = batched_scatter(x_tumor_position_v,dim=2,index = b_coordinates_cor_all_v[0],src = select_feature_125_v)
        x_tumor_position_t2 = batched_scatter(x_tumor_position_t2,dim=2,index = b_coordinates_cor_all_t2[0],src = select_feature_125_t2)
              


        
        x_tumor_position_v = x_tumor_position_v.view(b, f, c, h_image, w_image )
        x_tumor_position_t2 = x_tumor_position_t2.view(b, f, c, h_image, w_image )
        # x_tumor = x_tumor + x_tumor_position
        # x_liver = x_liver + x_liver_position
        
        return x_tumor_position_v,x_tumor_position_t2


class hgnn_fusion_space_3_ceng(nn.Module):
    ## 在空间上挑选并进行
    def __init__(self,points=2000,in_channels=16,out_channels=16,top_k =100,num_knn=10):
        super(hgnn_fusion_space_3_ceng,self).__init__()
        self.HGNN25 = EdgeConv2d(in_channels=in_channels, out_channels=out_channels)
        self.HGNN50 = EdgeConv2d(in_channels=in_channels, out_channels=out_channels)
        self.HGNN75 = EdgeConv2d(in_channels=in_channels, out_channels=out_channels)       
                    
        self.top_k = top_k
        self.num_knn =num_knn
        self.graph_constructor = DenseDilatedKnnGraph(k=top_k, dilation=1, stochastic=False, epsilon=0.0)
        self.w0 = nn.Parameter(torch.ones(2))
        self.w1 = nn.Parameter(torch.ones(2))
        ## 后期对比用
        self.A_25 =Hypergraph(points)
        self.A_50 =Hypergraph(points)
        self.A_75 =Hypergraph(points)


    def forward(self,X_tumor_v,X_tumor_t2,tumor_tv_mask,tumor_t2_mask,liver_tv_mask,liver_t2_mask):
        '''
        X_tumor_v: b,f,c,h,w
        X_tumor_t2: b,f,c,h,w
        tumor_tv_mask: b,f,h,w
        tumor_t2_mask: b,f,h,w
        '''
        b, f, c, h_image, w_image = X_tumor_v.shape
        device = X_tumor_v.device
         ## 寻找轮廓位置
        mode = cv2.RETR_EXTERNAL  # 例如，选择提取外部轮廓
        x_tumor_position_v = rearrange(X_tumor_v, 'b f c h w  -> b c (f h w)')
        x_tumor_position_t2 = rearrange(X_tumor_t2, 'b f c h w  -> b c (f h w)')
        
        ## 首先要对非空的mask进行统计，然后对非空的mask层数进行采样
        """
        b_new_coordinates_cor_all包含
        b_new_coordinates_25_one_frame_all 
        b_new_coordinates_50_one_frame_all 
        b_new_coordinates_75_one_frame_all 
        b_new_coordinates_100_one_frame_all 
        b_new_coordinates_125_one_frame_all 
        """
        b_coordinates_cor_all_v=top_feature_select(tumor_tv_mask,top_k=self.top_k,liver_tv_mask=liver_tv_mask)

        b_coordinates_cor_all_t2=top_feature_select(tumor_t2_mask,top_k=self.top_k,liver_tv_mask=liver_t2_mask)
       
        feature_25_v = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_v[0])
        feature_50_v = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_v[1])
        feature_75_v = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_v[2])
        
        
        feature_25_t2 = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_t2[0])
        feature_50_t2 = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_t2[1])
        feature_75_t2 = batched_index_gather(x_tumor_position_v, dim=2, index=b_coordinates_cor_all_t2[2])
        

        ##  得到tumor—feature  这次不需要按照mask提取特征
        # tumor_indices = mask_find_indices(tumor_mask)  ## b
        select_feature_25_v,select_feature_25_t2=multi_tumor_top_fusion(feature_25_v,feature_25_t2,b,self.A_25,self.top_k,self.num_knn,self.HGNN25,self.graph_constructor)
        x_tumor_position_v = batched_scatter(x_tumor_position_v,dim=2,index = b_coordinates_cor_all_v[0],src = select_feature_25_v)
        x_tumor_position_t2 = batched_scatter(x_tumor_position_t2,dim=2,index = b_coordinates_cor_all_t2[0],src = select_feature_25_t2)

        select_feature_50_v,select_feature_50_t2=multi_tumor_top_fusion(feature_50_v,feature_50_t2,b,self.A_50,self.top_k,self.num_knn,self.HGNN50,self.graph_constructor)
        x_tumor_position_v = batched_scatter(x_tumor_position_v,dim=2,index = b_coordinates_cor_all_v[0],src = select_feature_50_v)
        x_tumor_position_t2 = batched_scatter(x_tumor_position_t2,dim=2,index = b_coordinates_cor_all_t2[0],src = select_feature_50_t2)

        select_feature_75_v,select_feature_75_t2=multi_tumor_top_fusion(feature_75_v,feature_75_t2,b,self.A_75,self.top_k,self.num_knn,self.HGNN75,self.graph_constructor)
        x_tumor_position_v = batched_scatter(x_tumor_position_v,dim=2,index = b_coordinates_cor_all_v[0],src = select_feature_75_v)
        x_tumor_position_t2 = batched_scatter(x_tumor_position_t2,dim=2,index = b_coordinates_cor_all_t2[0],src = select_feature_75_t2)

                             
        x_tumor_position_v = x_tumor_position_v.view(b, f, c, h_image, w_image )
        x_tumor_position_t2 = x_tumor_position_t2.view(b, f, c, h_image, w_image )
        w1 = torch.exp(self.w0[0])/torch.sum(torch.exp(self.w0))
        w2 = torch.exp(self.w0[1])/torch.sum(torch.exp(self.w0))
        x_tumor_position_v = w1*X_tumor_v + w2*x_tumor_position_v
        w3 = torch.exp(self.w1[0])/torch.sum(torch.exp(self.w1))
        w4 = torch.exp(self.w1[1])/torch.sum(torch.exp(self.w1))
        x_tumor_position_t2 = w3*X_tumor_t2 + w4*x_tumor_position_t2
        
        return x_tumor_position_v,x_tumor_position_t2


class hgnn_fusion_top_space_liangjie(nn.Module):
    ## 在空间上挑选并进行
    def __init__(self,points=2000,in_channels=16,out_channels=16,top_k =100,num_knn=10):
        super(hgnn_fusion_top_space_liangjie,self).__init__()
        ## 取27个邻居，索引通道数*27
        self.HGNNyijie = EdgeConv2d(in_channels=in_channels*27, out_channels=out_channels*27)
        self.HGNNerjie = EdgeConv2d(in_channels=in_channels*27, out_channels=out_channels*27)                              
        self.top_k = top_k
        self.num_knn =num_knn
        self.graph_constructor = DenseDilatedKnnGraph(k=top_k, dilation=1, stochastic=False, epsilon=0.0)
        self.w0 = nn.Parameter(torch.ones(2))
        self.w1 = nn.Parameter(torch.ones(2))
        ## 后期对比用
        self.A_25 =Hypergraph(points)
        self.A_50 =Hypergraph(points)
        self.A_75 =Hypergraph(points)
        self.kernel_size =(3,3,3)
        self.stride =(3,3,3)
        self.pad = (0,0,0)
        
    def forward(self,X_tumor_v,X_tumor_t2,tumor_tv_mask,tumor_t2_mask):
        '''
        X_tumor_v: b,f,c,h,w
        X_tumor_t2: b,f,c,h,w
        tumor_tv_mask: b,f,h,w
        tumor_t2_mask: b,f,h,w
        '''
        b, f, c, h_image, w_image = X_tumor_v.shape
        device = X_tumor_v.device
         ## 寻找轮廓位置
        # x_tumor_position_v = rearrange(X_tumor_v, 'b f c h w  -> b c (f h w)')
        # x_tumor_position_t2 = rearrange(X_tumor_t2, 'b f c h w  -> b c (f h w)')
        tv_yijie_index,tv_erjie_index = mask_toptezhengtiqu(tumor_tv_mask,self.kernel_size,self.stride,self.top_k,b,f,c,h_image,w_image)        
        b,num_point=tv_yijie_index.shape
        # X_tumor_v = X_tumor_v.reshape(b,c,f,h_image,w_image) ## 先将通道数移动
        X_tumor_v_unfold = X_tumor_v.reshape(b,c,f,h_image,w_image).unfold(2,size=self.kernel_size[0], step=self.stride[0])\
            .unfold(3, size=self.kernel_size[1], step= self.stride[1])\
                .unfold(4, size=self.kernel_size[2], step= self.stride[2])\
                    .reshape(b,c*self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2],-1)        
        b,c_n,kernel_3=X_tumor_v_unfold.shape        
        X_tumor_v_feature_yijie = batched_index_gather(X_tumor_v_unfold, dim=2, index=tv_yijie_index)
        X_tumor_v_feature_erjie = batched_index_gather(X_tumor_v_unfold, dim=2, index=tv_erjie_index)

        t2_yijie_index,t2_erjie_index = mask_toptezhengtiqu(tumor_t2_mask,self.kernel_size,self.stride,self.top_k,b,f,c,h_image,w_image)        
        # X_tumor_t2 = X_tumor_t2.reshape(b,c,f,h_image,w_image) ## 先将通道数移动
        X_tumor_t2_unfold = X_tumor_t2.reshape(b,c,f,h_image,w_image).unfold(2,size=self.kernel_size[0], step=self.stride[0])\
            .unfold(3, size=self.kernel_size[1], step= self.stride[1])\
                .unfold(4, size=self.kernel_size[2], step= self.stride[2])\
                    .reshape(b,c*self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2],-1)
        X_tumor_t2_feature_yijie = batched_index_gather(X_tumor_t2_unfold, dim=2, index=t2_yijie_index)
        X_tumor_t2_feature_erjie = batched_index_gather(X_tumor_t2_unfold, dim=2, index=t2_erjie_index)

        X_tumor_v_feature_yijie,X_tumor_t2_feature_yijie=multi_tumor_top_fusion_v2(X_tumor_v_feature_yijie,X_tumor_t2_feature_yijie,b,self.A_50,self.top_k,self.num_knn,self.HGNNyijie,self.graph_constructor)
        X_tumor_v_feature_erjie,X_tumor_t2_feature_erjie=multi_tumor_top_fusion_v2(X_tumor_v_feature_erjie,X_tumor_t2_feature_erjie,b,self.A_50,self.top_k,self.num_knn,self.HGNNerjie,self.graph_constructor)
       

        X_tumor_v_fold = batched_scatter(X_tumor_v_unfold,dim=2,index = tv_yijie_index,src = X_tumor_v_feature_erjie)
        X_tumor_v_fold = batched_scatter(X_tumor_v_fold,dim=2,index = tv_erjie_index,src = X_tumor_v_feature_yijie)
       
        X_tumor_t2_fold = batched_scatter(X_tumor_t2_unfold,dim=2,index = t2_yijie_index,src = X_tumor_t2_feature_erjie)
        X_tumor_t2_fold = batched_scatter(X_tumor_t2_fold,dim=2,index = t2_erjie_index,src = X_tumor_t2_feature_yijie)
        
        X_tumor_v_fold = X_tumor_v_fold.reshape(b,c,f//self.kernel_size[0],h_image//self.kernel_size[1],w_image//self.kernel_size[2],
                                                    self.kernel_size[0],self.kernel_size[1],self.kernel_size[2])
        # X_tumor_v_fold=torch.nn.functional.fold(X_tumor_v_unfold, (b, f, c, h_image, w_image), kernel_size=self.kernel_size, stride=self.stride)
        X_tumor_v_fold = X_tumor_v_fold.reshape( b, f//self.kernel_size[0]*self.kernel_size[0],
                                                     c, h_image//self.kernel_size[1]*self.kernel_size[1],
                                                       w_image//self.kernel_size[2]*self.kernel_size[2])
        f_pad = f-f//self.kernel_size[0]*self.kernel_size[0]
        h_pad = h_image-h_image//self.kernel_size[1]*self.kernel_size[1]
        w_pad = w_image-w_image//self.kernel_size[2]*self.kernel_size[2]
        if f_pad!=0:
            X_tumor_v_fold = torch.cat((X_tumor_v_fold,X_tumor_v[:,-f_pad:,:,:-h_pad,:-w_pad]),dim=1)
        if h_pad!=0:
            X_tumor_v_fold = torch.cat((X_tumor_v_fold,X_tumor_v[:,:,:,-h_pad:,:-w_pad]),dim=3)
        if w_pad!=0:
            X_tumor_v_fold = torch.cat((X_tumor_v_fold,X_tumor_v[:,:,:,:,-w_pad:]),dim=4)
        
        X_tumor_t2_fold = X_tumor_t2_fold.reshape(b,c,f//self.kernel_size[0],h_image//self.kernel_size[1],w_image//self.kernel_size[2],
                                                    self.kernel_size[0],self.kernel_size[1],self.kernel_size[2])
        # X_tumor_v_fold=torch.nn.functional.fold(X_tumor_v_unfold, (b, f, c, h_image, w_image), kernel_size=self.kernel_size, stride=self.stride)
        X_tumor_t2_fold = X_tumor_t2_fold.reshape( b, f//self.kernel_size[0]*self.kernel_size[0],
                                                     c, h_image//self.kernel_size[1]*self.kernel_size[1],
                                                       w_image//self.kernel_size[2]*self.kernel_size[2])
        f_pad = f-f//self.kernel_size[0]*self.kernel_size[0]
        h_pad = h_image-h_image//self.kernel_size[1]*self.kernel_size[1]
        w_pad = w_image-w_image//self.kernel_size[2]*self.kernel_size[2]
        if f_pad!=0:
            X_tumor_t2_fold = torch.cat((X_tumor_t2_fold,X_tumor_v[:,-f_pad:,:,:-h_pad,:-w_pad]),dim=1)
        if h_pad!=0:
            X_tumor_t2_fold = torch.cat((X_tumor_t2_fold,X_tumor_v[:,:,:,-h_pad:,:-w_pad]),dim=3)
        if w_pad!=0:
            X_tumor_t2_fold = torch.cat((X_tumor_t2_fold,X_tumor_v[:,:,:,:,-w_pad:]),dim=4)
        
                             
        # x_tumor_position_v = x_tumor_position_v.view(b, f, c, h_image, w_image )
        # x_tumor_position_t2 = x_tumor_position_t2.view(b, f, c, h_image, w_image )
        w1 = torch.exp(self.w0[0])/torch.sum(torch.exp(self.w0))
        w2 = torch.exp(self.w0[1])/torch.sum(torch.exp(self.w0))
        x_tumor_position_v = w1*X_tumor_v + w2*X_tumor_v_fold
        w3 = torch.exp(self.w1[0])/torch.sum(torch.exp(self.w1))
        w4 = torch.exp(self.w1[1])/torch.sum(torch.exp(self.w1))
        x_tumor_position_t2 = w3*X_tumor_t2 + w4*X_tumor_t2_fold
        
        return x_tumor_position_v,x_tumor_position_t2

def mask_toptezhengtiqu(tumor_tv_mask,kernel_size,stride,top_k,b,f,c,h_image,w_image):
    tumor_tv_mask_unfold = tumor_tv_mask.unfold(1,size=kernel_size[0], step=stride[0])\
            .unfold(2, size=kernel_size[1], step= stride[1])\
                .unfold(3, size=kernel_size[2], step= stride[2])\
                    .reshape(b,-1,kernel_size[0]*kernel_size[1]*kernel_size[2])
    tumor_tv_mask_unfold =tumor_tv_mask_unfold.sum(dim=2)
    _, tumor_tv_top_indices = torch.topk(tumor_tv_mask_unfold.view(b, -1), top_k, dim=1)
    ## 记录每一行 每一列能分成多少个patch
    num_f = f // kernel_size[0]
    num_h = h_image //kernel_size[1]
    num_w = w_image //kernel_size[2]
    yijie_index = kuozhan_max_yijie(tumor_tv_top_indices,num_f,num_h,num_w)
    erjie_index = kuozhan_max_erjie(tumor_tv_top_indices,num_f,num_h,num_w)
    return yijie_index,erjie_index

def kuozhan_max_yijie(liver_top_indices,num_f,num_h,num_w):
    """
    liver_top_indices:b*1 中心点位置    
    """
    result = []    
    yuejie = num_f*num_h*num_w
    b,k = liver_top_indices.shape
    for i in range(b):
        index = liver_top_indices[i]
        ## 一阶相邻
        index_0   = index+1
        index_180 = index-1
        index_90  = index-num_w
        index_270 = index+num_w
        index_45  = index_90 -1
        index_135 = index_90 + 1
        index_225 = index_270-1
        index_315 = index_270+1
        ## 空间相邻
        yijie = [index,index_0,index_180,index_90,index_270,index_45,index_135,index_225,index_315]
        yijie = max_index_filter(yijie,yuejie)
        yijie_xiaceng = [num_h * num_w + index if index + num_h * num_w < yuejie else index for index in yijie]

        yijie_shangceng = [index-num_h * num_w  if index - num_h * num_w > 0 else index for index in yijie]       
        
        result.append(torch.cat([torch.cat(yijie_shangceng),torch.cat(yijie),torch.cat(yijie_xiaceng)]))
    yijielingju=torch.stack(result,dim=0)
    return yijielingju

def kuozhan_max_erjie(liver_top_indices,num_f,num_h,num_w):
    """
    liver_top_indices:b*1 中心点位置    
    """
    result = []    
    yuejie = num_f*num_h*num_w
    b,k = liver_top_indices.shape
    for i in range(b):
        index = liver_top_indices[i]
        ## 一阶相邻
        index_0   = index+2
        index_180 = index-2
        index_90  = index-2*num_w
        index_270 = index+2*num_w
        index_45  = index_90 - 2
        index_135 = index_90 + 2
        index_225 = index_270 -2
        index_315 = index_270 + 2
        ## 空间相邻
        yijie = [index,index_0,index_180,index_90,index_270,index_45,index_135,index_225,index_315]
        yijie = max_index_filter(yijie,yuejie)
        yijie_xiaceng = [(2*num_h * num_w) + index if index + (2*num_h * num_w) < yuejie else index for index in yijie]

        yijie_shangceng = [index- (2*num_h*num_w)  if index - (2*num_h * num_w) > 0 else index for index in yijie]       
        
        result.append(torch.cat([torch.cat(yijie_shangceng),torch.cat(yijie),torch.cat(yijie_xiaceng)]))
    yijielingju=torch.stack(result,dim=0)

    return yijielingju

def max_index_filter(yijie,max_index):
    processed_indices = []

    # 遍历索引  
    for index in yijie:
        # 如果索引大于等于边界值，取边界值
        if index >= max_index:
            processed_index = torch.tensor(max_index-1,device=index.device,dtype=torch.int64).unsqueeze(dim=0)
        # 如果索引小于0，取0
        elif index < 0:
            processed_index = torch.tensor(0+1,device=index.device,dtype=torch.int64).unsqueeze(dim=0)
        # 否则，保持不变
        else:
            processed_index = index
        # 将处理后的索引值添加到列表中
        processed_indices.append(processed_index)

    # 输出处理后的索引列表
    # print(processed_indices)
    return processed_indices


class CNN3d_t2_tv_hgnn_0414(nn.Module):
    """
    从0414 构建多阶段融合
    """
    def __init__(self,num_classes=1,points=1000, num_channels =128):
        super(CNN3d_t2_tv_hgnn_0414, self).__init__()
        ## one stage     
        self.liver_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_v = Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv1_t2 =  Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')

        self.tumor_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_v =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')                                                 
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM  = nn.LSTM(
                input_size=num_channels*4,
                hidden_size=128,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
              
        self.fc1 = nn.Linear(168, 32)
        self.fc1_5 = nn.Linear(148, 32)
        
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.stem_v = Stem(out_dim=640) ## 特征提取器
        self.stem_t2 = Stem(out_dim=640)
        self.hyper2d_tv = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_tv = FFN(640,640*4)
        self.hyper2d_t2 = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_t2 = FFN(640,640*4)
        ## 换一种方式的超图卷积 注意 这种超图卷积的输入是没有batch维度，所以每一个样本 单独初始化一个超图
        # self.layers2= HGNNConv(hid_channels=16, num_classes, is_last=True)
        ## 肿瘤内部空间融合
        self.tumor_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.tumor_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_tumor_fusion_t2_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 模态间融合
        self.tumor_fusion_v_t2_1 = hgnn_fusion_space_3_ceng(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_fusion_v_t2_1 = hgnn_fusion_space(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        
        self.tumor_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.tumor_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_tumor_fusion_t2_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 模态间融合
        self.tumor_fusion_v_t2_2 = hgnn_fusion_space_3_ceng(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//4)
        self.liver_fusion_v_t2_2 = hgnn_fusion_space_3_ceng(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//4)

        self.tumor_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.tumor_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_tumor_fusion_t2_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 模态间融合
        self.tumor_fusion_v_t2_3 = hgnn_fusion_space(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//8)


        self.tumor_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.tumor_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_tumor_fusion_t2_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 模态间融合
        self.tumor_fusion_v_t2_4 = hgnn_fusion_space(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)



    def forward(self,X_image_mask):
        # torch.autograd.set_detect_anomaly(True)
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_t2_liver,X_t2_tumor,liver_t2_mask,tumor_t2_mask,X_tv_liver,X_tv_tumor,liver_tv_mask,tumor_tv_mask=X_image_mask
        b,f,c,h,w = X_tv_tumor.shape
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image= X_V_mask.to(torch.float32),X_V_image.to(torch.float32), X_T2_mask.to(torch.float32),X_T2_image.to(torch.float32)
        X_tumor_v_1 = []
        # X_liver_t2_1 = []
        X_tumor_t2_1 = []
        tumor_tv_mask = tumor_tv_mask.squeeze()
        tumor_t2_mask = tumor_t2_mask.squeeze()
        liver_t2_mask =liver_t2_mask.squeeze()
        liver_tv_mask =liver_tv_mask.squeeze()
        ## 为了gradcam 更好，先将b f合并
        '''
        for i in range(f):
            x_t2_liver_2d = X_t2_liver[:, i,:]
            x_t2_tumor_2d = X_t2_tumor[:, i,:]
            x_tv_liver_2d = X_tv_liver[:, i,:]
            x_tv_tumor_2d = X_tv_tumor[:, i,:]
            
            x_tumor_v =self.tumor_conv1_v(x_tv_tumor_2d)
            x_tumor_t2 =self.tumor_conv1_t2(x_t2_tumor_2d)
            x_liver_v =self.liver_conv1_v(x_tv_tumor_2d)
            x_liver_t2 =self.liver_conv1_t2(x_t2_tumor_2d)

            X_tumor_v_1.append(x_tumor_v)

            # X_liver_v_1.append(x_liver_v)
            X_tumor_t2_1.append(x_tumor_t2)
            # X_liver_t2_1.append(x_liver_t2)
        
        X_tumor_v_1 = torch.stack(X_tumor_v_1, dim=1) 
        # X_liver_t2_1 = torch.stack(X_liver_t2_1, dim=1)
        X_tumor_t2_1 = torch.stack(X_tumor_t2_1, dim=1) 
        '''
        X_t2_liver =X_t2_liver.view(b*f,c,h,w)
        X_t2_tumor =X_t2_tumor.view(b*f,c,h,w)
        X_tv_tumor =X_tv_tumor.view(b*f,c,h,w)
        X_tv_liver =X_tv_liver.view(b*f,c,h,w)
        X_t2_tumor =self.tumor_conv1_t2(X_t2_tumor)
        X_tv_tumor =self.tumor_conv1_v(X_tv_tumor)
        X_tv_liver =self.liver_conv1_v(X_tv_liver)
        X_t2_liver =self.liver_conv1_t2(X_t2_liver)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor.shape
        X_t2_liver_1 =X_t2_liver.view(b,f,c,h,w)
        X_t2_tumor_1 =X_t2_tumor.view(b,f,c,h,w)
        X_tv_tumor_1 =X_tv_tumor.view(b,f,c,h,w)
        X_tv_liver_1=X_tv_liver.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])
        ## 超图的构建中，怎么去采样是
        # new_X_tumor_v_1 = feature_selection(X_tumor_v_1, k=1000)
        # ## 内部融合
        X_tv_tumor_1 = self.tumor_hgnn_fusion_v_1(X_tv_tumor_1)
        X_t2_tumor_1 = self.tumor_hgnn_fusion_t2_1(X_t2_tumor_1)
        X_tv_liver_1 = self.liver_hgnn_fusion_v_1(X_tv_liver_1)
        X_t2_liver_1 = self.liver_hgnn_fusion_t2_1(X_t2_liver_1)
        ## 肝脏肿瘤融合

        start_time = time.time()
        # 执行代码        
        X_tv_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_v_1(X_tv_tumor_1,X_tv_liver_1)
        X_t2_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_t2_1(X_t2_tumor_1,X_t2_liver_1)
        end_time = time.time()
              ## 模态间融合
        X_tv_liver_1,X_t2_liver_1=self.liver_fusion_v_t2_1(X_tv_liver_1,X_t2_liver_1,tumor_tv_mask,tumor_t2_mask,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_1,X_t2_tumor_1=self.tumor_fusion_v_t2_1(X_tv_tumor_1,X_t2_tumor_1,tumor_tv_mask,tumor_t2_mask,liver_tv_mask,liver_t2_mask)
        
        X_t2_liver_2 =X_t2_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_1.view(b*f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_1.view(b*f,c,h,w)
        X_tv_liver_2 =X_tv_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =self.tumor_conv2_t2(X_t2_tumor_2)
        X_tv_tumor_2 =self.tumor_conv2_v(X_tv_tumor_2)
        X_tv_liver_2 =self.liver_conv2_v(X_tv_liver_2)
        X_t2_liver_2 =self.liver_conv2_t2(X_t2_liver_2)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_2.shape
        X_t2_liver_2 =X_t2_liver_2.view(b,f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_2.view(b,f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_2.view(b,f,c,h,w)
        X_tv_liver_2 =X_tv_liver_2.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_2 = self.tumor_hgnn_fusion_v_2(X_tv_tumor_2)
        X_t2_tumor_2 = self.tumor_hgnn_fusion_t2_2(X_t2_tumor_2)
        X_tv_liver_2 = self.liver_hgnn_fusion_v_2(X_tv_liver_2)
        X_t2_liver_2 = self.liver_hgnn_fusion_t2_2(X_t2_liver_2)
        ## 肝脏肿瘤融合
        X_tv_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_v_2(X_tv_tumor_2,X_tv_liver_2)
        X_t2_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_t2_2(X_t2_tumor_2,X_t2_liver_2)
        ## 模态间融合
        # X_tv_liver_2,X_t2_liver_2=self.liver_fusion_v_t2_2(X_tv_liver_2,X_t2_liver_2,tumor_tv_mask,tumor_t2_mask,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_2,X_t2_tumor_2=self.tumor_fusion_v_t2_2(X_tv_tumor_2,X_t2_tumor_2,tumor_tv_mask,tumor_t2_mask,liver_tv_mask,liver_t2_mask)
        

        X_t2_liver_3 =X_t2_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_2.view(b*f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_2.view(b*f,c,h,w)
        X_tv_liver_3 =X_tv_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =self.tumor_conv3_t2(X_t2_tumor_3)
        X_tv_tumor_3 =self.tumor_conv3_v(X_tv_tumor_3)
        X_tv_liver_3 =self.liver_conv3_v(X_tv_liver_3)
        X_t2_liver_3 =self.liver_conv3_t2(X_t2_liver_3)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_3.shape
        X_t2_liver_3 =X_t2_liver_3.view(b,f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_3.view(b,f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_3.view(b,f,c,h,w)
        X_tv_liver_3 =X_tv_liver_3.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_3 = self.tumor_hgnn_fusion_v_3(X_tv_tumor_3)
        X_t2_tumor_3 = self.tumor_hgnn_fusion_t2_3(X_t2_tumor_3)
        X_tv_liver_3 = self.liver_hgnn_fusion_v_3(X_tv_liver_3)
        X_t2_liver_3 = self.liver_hgnn_fusion_t2_3(X_t2_liver_3)
        ## 肝脏肿瘤融合
        X_tv_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_v_3(X_tv_tumor_3,X_tv_liver_3)
        X_t2_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_t2_3(X_t2_tumor_3,X_t2_liver_3)
        ## 模态间融合
        # X_tv_liver_3,X_t2_liver_3=self.tumor_fusion_v_t2_3(X_tv_liver_3,X_t2_liver_3,tumor_tv_mask,tumor_t2_mask,liver_tv_mask,liver_t2_mask)

        
        X_t2_liver_4 =X_t2_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_3.view(b*f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_3.view(b*f,c,h,w)
        X_tv_liver_4 =X_tv_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =self.tumor_conv4_t2(X_t2_tumor_4)
        X_tv_tumor_4 =self.tumor_conv4_v(X_tv_tumor_4)
        X_tv_liver_4 =self.liver_conv4_v(X_tv_liver_4)
        X_t2_liver_4 =self.liver_conv4_t2(X_t2_liver_4)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_4.shape
        X_t2_liver_4 =X_t2_liver_4.view(b,f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_4.view(b,f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_4.view(b,f,c,h,w)
        X_tv_liver_4 =X_tv_liver_4.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w],mode='nearest')
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w],mode='nearest')
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w],mode='nearest')
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w],mode='nearest')

        X_tv_tumor_4 = self.tumor_hgnn_fusion_v_4(X_tv_tumor_4)
        X_t2_tumor_4 = self.tumor_hgnn_fusion_t2_4(X_t2_tumor_4)
        X_tv_liver_4 = self.liver_hgnn_fusion_v_4(X_tv_liver_4)
        X_t2_liver_4 = self.liver_hgnn_fusion_t2_4(X_t2_liver_4)
        # 肝脏肿瘤融合
        # X_tv_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_v_4(X_tv_tumor_4,X_tv_liver_4)
        # X_t2_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_t2_4(X_t2_tumor_4,X_t2_liver_4)
        # ## 模态间融合
        # X_tv_liver_4,X_t2_liver_4=self.tumor_fusion_v_t2_4(X_tv_liver_4,X_t2_liver_4,tumor_tv_mask,tumor_t2_mask,liver_tv_mask,liver_t2_mask)



        tumor_tv_mask_pool = self.pool(tumor_tv_mask).squeeze()
        tumor_t2_mask_pool = self.pool(tumor_t2_mask).squeeze()
        liver_tv_mask_pool = self.pool(liver_tv_mask).squeeze()
        liver_t2_mask_pool = self.pool(liver_t2_mask).squeeze()
        # # X_liver_v_gate = self.pool(X_liver_v_gate).squeeze()
        # # X_liver_t2_gate = self.pool(X_liver_t2_gate).squeeze()

        X_liver_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_tumor_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_liver_t2_pool = self.pool(X_t2_liver_4).squeeze()
        X_tumor_t2_pool = self.pool(X_t2_liver_4).squeeze()
        
        
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        # X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        X_pool = torch.cat([ X_tumor_v_pool,  X_tumor_t2_pool,X_liver_v_pool,X_liver_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool,liver_t2_mask_pool,liver_tv_mask_pool],dim=-1)

        # x = self.fc1(X_pool) 
        if f ==10:
            x = self.fc1(x)
        elif f ==5:
            x = self.fc1_5(x)
        # x = self.fc_mask(x)
        # x = F.relu(x,inplace=True)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        execution_time = end_time - start_time
        # print(f"代码执行时间为：{execution_time}秒")  
        return x


class CNN3d_t2_tv_hgnn_0414_three_model(nn.Module):
    """
    从0414 构建多阶段融合
    """
    def __init__(self,num_classes=1,points=1000, num_channels =128):
        super(CNN3d_t2_tv_hgnn_0414_three_model, self).__init__()
        ## one stage     
        self.liver_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_v = Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv1_t2 =  Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')

        self.tumor_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_v =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')                                                 
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM  = nn.LSTM(
                input_size=num_channels*4,
                hidden_size=128,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
              
        self.fc1 = nn.Linear(168, 32)
        self.fc1_5 = nn.Linear(148, 32)
        
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.stem_v = Stem(out_dim=640) ## 特征提取器
        self.stem_t2 = Stem(out_dim=640)
        self.hyper2d_tv = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_tv = FFN(640,640*4)
        self.hyper2d_t2 = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_t2 = FFN(640,640*4)
        ## 换一种方式的超图卷积 注意 这种超图卷积的输入是没有batch维度，所以每一个样本 单独初始化一个超图
        # self.layers2= HGNNConv(hid_channels=16, num_classes, is_last=True)
        ## 肿瘤内部空间融合
        self.tumor_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.tumor_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_tumor_fusion_t2_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 模态间融合
        self.tumor_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        self.liver_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        
        self.tumor_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.tumor_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_tumor_fusion_t2_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 模态间融合
        self.tumor_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)
        self.liver_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)

        self.tumor_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.tumor_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_tumor_fusion_t2_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 模态间融合
        self.tumor_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)
        self.liver_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)


        self.tumor_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.tumor_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_tumor_fusion_t2_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 模态间融合
        self.tumor_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        self.liver_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        


    def forward(self,X_image_mask):
        # torch.autograd.set_detect_anomaly(True)
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_t2_liver,X_t2_tumor,liver_t2_mask,tumor_t2_mask,X_tv_liver,X_tv_tumor,liver_tv_mask,tumor_tv_mask=X_image_mask
        b,f,c,h,w = X_tv_tumor.shape
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image= X_V_mask.to(torch.float32),X_V_image.to(torch.float32), X_T2_mask.to(torch.float32),X_T2_image.to(torch.float32)
        X_tumor_v_1 = []
        # X_liver_t2_1 = []
        X_tumor_t2_1 = []
        tumor_tv_mask = tumor_tv_mask.squeeze()
        tumor_t2_mask = tumor_t2_mask.squeeze()
        liver_t2_mask =liver_t2_mask.squeeze()
        liver_tv_mask =liver_tv_mask.squeeze()
        ## 为了gradcam 更好，先将b f合并
        '''
        for i in range(f):
            x_t2_liver_2d = X_t2_liver[:, i,:]
            x_t2_tumor_2d = X_t2_tumor[:, i,:]
            x_tv_liver_2d = X_tv_liver[:, i,:]
            x_tv_tumor_2d = X_tv_tumor[:, i,:]
            
            x_tumor_v =self.tumor_conv1_v(x_tv_tumor_2d)
            x_tumor_t2 =self.tumor_conv1_t2(x_t2_tumor_2d)
            x_liver_v =self.liver_conv1_v(x_tv_tumor_2d)
            x_liver_t2 =self.liver_conv1_t2(x_t2_tumor_2d)

            X_tumor_v_1.append(x_tumor_v)

            # X_liver_v_1.append(x_liver_v)
            X_tumor_t2_1.append(x_tumor_t2)
            # X_liver_t2_1.append(x_liver_t2)
        
        X_tumor_v_1 = torch.stack(X_tumor_v_1, dim=1) 
        # X_liver_t2_1 = torch.stack(X_liver_t2_1, dim=1)
        X_tumor_t2_1 = torch.stack(X_tumor_t2_1, dim=1) 
        '''
        X_t2_liver =X_t2_liver.view(b*f,c,h,w)
        X_t2_tumor =X_t2_tumor.view(b*f,c,h,w)
        X_tv_tumor =X_tv_tumor.view(b*f,c,h,w)
        X_tv_liver =X_tv_liver.view(b*f,c,h,w)
        X_t2_tumor =self.tumor_conv1_t2(X_t2_tumor)
        X_tv_tumor =self.tumor_conv1_v(X_tv_tumor)
        X_tv_liver =self.liver_conv1_v(X_tv_liver)
        X_t2_liver =self.liver_conv1_t2(X_t2_liver)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor.shape
        X_t2_liver_1 =X_t2_liver.view(b,f,c,h,w)
        X_t2_tumor_1 =X_t2_tumor.view(b,f,c,h,w)
        X_tv_tumor_1 =X_tv_tumor.view(b,f,c,h,w)
        X_tv_liver_1=X_tv_liver.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])
        ## 超图的构建中，怎么去采样是
        # new_X_tumor_v_1 = feature_selection(X_tumor_v_1, k=1000)
        # ## 内部融合
        X_tv_tumor_1 = self.tumor_hgnn_fusion_v_1(X_tv_tumor_1)
        X_t2_tumor_1 = self.tumor_hgnn_fusion_t2_1(X_t2_tumor_1)
        X_tv_liver_1 = self.liver_hgnn_fusion_v_1(X_tv_liver_1)
        X_t2_liver_1 = self.liver_hgnn_fusion_t2_1(X_t2_liver_1)
        ## 肝脏肿瘤融合
        start_time = time.time()
        # 执行代码        
        X_tv_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_v_1(X_tv_tumor_1,X_tv_liver_1)
        X_t2_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_t2_1(X_t2_tumor_1,X_t2_liver_1)
        end_time = time.time()
        ## 模态间融合
        X_tv_liver_1,X_t2_liver_1=self.liver_fusion_v_t2_1(X_tv_liver_1,X_t2_liver_1,liver_tv_mask,liver_t2_mask)
        X_tv_tumor_1,X_t2_tumor_1=self.tumor_fusion_v_t2_1(X_tv_tumor_1,X_t2_tumor_1,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_2 =X_t2_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_1.view(b*f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_1.view(b*f,c,h,w)
        X_tv_liver_2 =X_tv_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =self.tumor_conv2_t2(X_t2_tumor_2)
        X_tv_tumor_2 =self.tumor_conv2_v(X_tv_tumor_2)
        X_tv_liver_2 =self.liver_conv2_v(X_tv_liver_2)
        X_t2_liver_2 =self.liver_conv2_t2(X_t2_liver_2)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_2.shape
        X_t2_liver_2 =X_t2_liver_2.view(b,f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_2.view(b,f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_2.view(b,f,c,h,w)
        X_tv_liver_2 =X_tv_liver_2.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_2 = self.tumor_hgnn_fusion_v_2(X_tv_tumor_2)
        X_t2_tumor_2 = self.tumor_hgnn_fusion_t2_2(X_t2_tumor_2)
        X_tv_liver_2 = self.liver_hgnn_fusion_v_2(X_tv_liver_2)
        X_t2_liver_2 = self.liver_hgnn_fusion_t2_2(X_t2_liver_2)
        ## 肝脏肿瘤融合
        X_tv_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_v_2(X_tv_tumor_2,X_tv_liver_2)
        X_t2_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_t2_2(X_t2_tumor_2,X_t2_liver_2)
        ## 模态间融合
        # X_tv_liver_2,X_t2_liver_2=self.liver_fusion_v_t2_2(X_tv_liver_2,X_t2_liver_2,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_2,X_t2_tumor_2=self.tumor_fusion_v_t2_2(X_tv_tumor_2,X_t2_tumor_2,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_3 =X_t2_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_2.view(b*f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_2.view(b*f,c,h,w)
        X_tv_liver_3 =X_tv_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =self.tumor_conv3_t2(X_t2_tumor_3)
        X_tv_tumor_3 =self.tumor_conv3_v(X_tv_tumor_3)
        X_tv_liver_3 =self.liver_conv3_v(X_tv_liver_3)
        X_t2_liver_3 =self.liver_conv3_t2(X_t2_liver_3)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_3.shape
        X_t2_liver_3 =X_t2_liver_3.view(b,f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_3.view(b,f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_3.view(b,f,c,h,w)
        X_tv_liver_3 =X_tv_liver_3.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_3 = self.tumor_hgnn_fusion_v_3(X_tv_tumor_3)
        X_t2_tumor_3 = self.tumor_hgnn_fusion_t2_3(X_t2_tumor_3)
        X_tv_liver_3 = self.liver_hgnn_fusion_v_3(X_tv_liver_3)
        X_t2_liver_3 = self.liver_hgnn_fusion_t2_3(X_t2_liver_3)
        ## 肝脏肿瘤融合
        X_tv_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_v_3(X_tv_tumor_3,X_tv_liver_3)
        X_t2_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_t2_3(X_t2_tumor_3,X_t2_liver_3)
        ## 模态间融合
        # X_tv_liver_3,X_t2_liver_3=self.liver_fusion_v_t2_3(X_tv_liver_3,X_t2_liver_3,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_3,X_t2_tumor_3=self.tumor_fusion_v_t2_3(X_tv_tumor_3,X_t2_tumor_3,tumor_tv_mask,tumor_t2_mask)
       
        
        X_t2_liver_4 =X_t2_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_3.view(b*f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_3.view(b*f,c,h,w)
        X_tv_liver_4 =X_tv_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =self.tumor_conv4_t2(X_t2_tumor_4)
        X_tv_tumor_4 =self.tumor_conv4_v(X_tv_tumor_4)
        X_tv_liver_4 =self.liver_conv4_v(X_tv_liver_4)
        X_t2_liver_4 =self.liver_conv4_t2(X_t2_liver_4)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_4.shape
        X_t2_liver_4 =X_t2_liver_4.view(b,f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_4.view(b,f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_4.view(b,f,c,h,w)
        X_tv_liver_4 =X_tv_liver_4.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w],mode='nearest')
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w],mode='nearest')
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w],mode='nearest')
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w],mode='nearest')

        X_tv_tumor_4 = self.tumor_hgnn_fusion_v_4(X_tv_tumor_4)
        X_t2_tumor_4 = self.tumor_hgnn_fusion_t2_4(X_t2_tumor_4)
        X_tv_liver_4 = self.liver_hgnn_fusion_v_4(X_tv_liver_4)
        X_t2_liver_4 = self.liver_hgnn_fusion_t2_4(X_t2_liver_4)
        # 肝脏肿瘤融合
        X_tv_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_v_4(X_tv_tumor_4,X_tv_liver_4)
        X_t2_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_t2_4(X_t2_tumor_4,X_t2_liver_4)
        # ## 模态间融合
        # X_tv_liver_4,X_t2_liver_4=self.liver_fusion_v_t2_4(X_tv_liver_4,X_t2_liver_4,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_4,X_t2_tumor_4=self.tumor_fusion_v_t2_4(X_tv_tumor_4,X_t2_tumor_4,tumor_tv_mask,tumor_t2_mask)
       


        tumor_tv_mask_pool = self.pool(tumor_tv_mask).squeeze()
        tumor_t2_mask_pool = self.pool(tumor_t2_mask).squeeze()
        liver_tv_mask_pool = self.pool(liver_tv_mask).squeeze()
        liver_t2_mask_pool = self.pool(liver_t2_mask).squeeze()
        # # X_liver_v_gate = self.pool(X_liver_v_gate).squeeze()
        # # X_liver_t2_gate = self.pool(X_liver_t2_gate).squeeze()

        X_liver_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_tumor_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_liver_t2_pool = self.pool(X_t2_liver_4).squeeze()
        X_tumor_t2_pool = self.pool(X_t2_liver_4).squeeze()
        
        
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        # X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        X_pool = torch.cat([ X_tumor_v_pool,  X_tumor_t2_pool,X_liver_v_pool,X_liver_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool,liver_t2_mask_pool,liver_tv_mask_pool],dim=-1)

        # x = self.fc1(X_pool) 
        if f ==10:
            x = self.fc1(x)
        elif f ==5:
            x = self.fc1_5(x)
        # x = self.fc_mask(x)
        # x = F.relu(x,inplace=True)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        execution_time = end_time - start_time
        # print(f"代码执行时间为：{execution_time}秒")  
        return x

class CNN3d_t2_tv_hgnn_0414_baseline(nn.Module):
    """
    从0414 构建多阶段融合
    """
    def __init__(self,num_classes=1,points=1000, num_channels =128):
        super(CNN3d_t2_tv_hgnn_0414_baseline, self).__init__()
        ## one stage     
        self.liver_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_v = Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv1_t2 =  Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')

        self.tumor_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_v =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')                                                 
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM  = nn.LSTM(
                input_size=num_channels*4,
                hidden_size=128,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
              
        self.fc1 = nn.Linear(168, 32)
        self.fc1_5 = nn.Linear(148, 32)
        
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.stem_v = Stem(out_dim=640) ## 特征提取器
        self.stem_t2 = Stem(out_dim=640)
        self.hyper2d_tv = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_tv = FFN(640,640*4)
        self.hyper2d_t2 = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_t2 = FFN(640,640*4)
        ## 换一种方式的超图卷积 注意 这种超图卷积的输入是没有batch维度，所以每一个样本 单独初始化一个超图
        # self.layers2= HGNNConv(hid_channels=16, num_classes, is_last=True)
        ## 肿瘤内部空间融合
        self.tumor_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.tumor_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_tumor_fusion_t2_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 模态间融合
        self.tumor_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        self.liver_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        
        self.tumor_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.tumor_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_tumor_fusion_t2_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 模态间融合
        self.tumor_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)
        self.liver_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)

        self.tumor_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.tumor_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_tumor_fusion_t2_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 模态间融合
        self.tumor_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)
        self.liver_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)


        self.tumor_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.tumor_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_tumor_fusion_t2_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 模态间融合
        self.tumor_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        self.liver_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        


    def forward(self,X_image_mask):
        # torch.autograd.set_detect_anomaly(True)
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_t2_liver,X_t2_tumor,liver_t2_mask,tumor_t2_mask,X_tv_liver,X_tv_tumor,liver_tv_mask,tumor_tv_mask=X_image_mask
        b,f,c,h,w = X_tv_tumor.shape
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image= X_V_mask.to(torch.float32),X_V_image.to(torch.float32), X_T2_mask.to(torch.float32),X_T2_image.to(torch.float32)
        X_tumor_v_1 = []
        # X_liver_t2_1 = []
        X_tumor_t2_1 = []
        tumor_tv_mask = tumor_tv_mask.squeeze()
        tumor_t2_mask = tumor_t2_mask.squeeze()
        liver_t2_mask =liver_t2_mask.squeeze()
        liver_tv_mask =liver_tv_mask.squeeze()
        ## 为了gradcam 更好，先将b f合并
        '''
        for i in range(f):
            x_t2_liver_2d = X_t2_liver[:, i,:]
            x_t2_tumor_2d = X_t2_tumor[:, i,:]
            x_tv_liver_2d = X_tv_liver[:, i,:]
            x_tv_tumor_2d = X_tv_tumor[:, i,:]
            
            x_tumor_v =self.tumor_conv1_v(x_tv_tumor_2d)
            x_tumor_t2 =self.tumor_conv1_t2(x_t2_tumor_2d)
            x_liver_v =self.liver_conv1_v(x_tv_tumor_2d)
            x_liver_t2 =self.liver_conv1_t2(x_t2_tumor_2d)

            X_tumor_v_1.append(x_tumor_v)

            # X_liver_v_1.append(x_liver_v)
            X_tumor_t2_1.append(x_tumor_t2)
            # X_liver_t2_1.append(x_liver_t2)
        
        X_tumor_v_1 = torch.stack(X_tumor_v_1, dim=1) 
        # X_liver_t2_1 = torch.stack(X_liver_t2_1, dim=1)
        X_tumor_t2_1 = torch.stack(X_tumor_t2_1, dim=1) 
        '''
        X_t2_liver =X_t2_liver.view(b*f,c,h,w)
        X_t2_tumor =X_t2_tumor.view(b*f,c,h,w)
        X_tv_tumor =X_tv_tumor.view(b*f,c,h,w)
        X_tv_liver =X_tv_liver.view(b*f,c,h,w)
        X_t2_tumor =self.tumor_conv1_t2(X_t2_tumor)
        X_tv_tumor =self.tumor_conv1_v(X_tv_tumor)
        X_tv_liver =self.liver_conv1_v(X_tv_liver)
        X_t2_liver =self.liver_conv1_t2(X_t2_liver)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor.shape
        X_t2_liver_1 =X_t2_liver.view(b,f,c,h,w)
        X_t2_tumor_1 =X_t2_tumor.view(b,f,c,h,w)
        X_tv_tumor_1 =X_tv_tumor.view(b,f,c,h,w)
        X_tv_liver_1=X_tv_liver.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])
        ## 超图的构建中，怎么去采样是
        # new_X_tumor_v_1 = feature_selection(X_tumor_v_1, k=1000)
        # ## 内部融合
        # X_tv_tumor_1 = self.tumor_hgnn_fusion_v_1(X_tv_tumor_1)
        # X_t2_tumor_1 = self.tumor_hgnn_fusion_t2_1(X_t2_tumor_1)
        # X_tv_liver_1 = self.liver_hgnn_fusion_v_1(X_tv_liver_1)
        # X_t2_liver_1 = self.liver_hgnn_fusion_t2_1(X_t2_liver_1)
        # ## 肝脏肿瘤融合
        # start_time = time.time()
        # # 执行代码        
        # X_tv_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_v_1(X_tv_tumor_1,X_tv_liver_1)
        # X_t2_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_t2_1(X_t2_tumor_1,X_t2_liver_1)
        # end_time = time.time()
        # ## 模态间融合
        # X_tv_liver_1,X_t2_liver_1=self.liver_fusion_v_t2_1(X_tv_liver_1,X_t2_liver_1,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_1,X_t2_tumor_1=self.tumor_fusion_v_t2_1(X_tv_tumor_1,X_t2_tumor_1,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_2 =X_t2_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_1.view(b*f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_1.view(b*f,c,h,w)
        X_tv_liver_2 =X_tv_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =self.tumor_conv2_t2(X_t2_tumor_2)
        X_tv_tumor_2 =self.tumor_conv2_v(X_tv_tumor_2)
        X_tv_liver_2 =self.liver_conv2_v(X_tv_liver_2)
        X_t2_liver_2 =self.liver_conv2_t2(X_t2_liver_2)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_2.shape
        X_t2_liver_2 =X_t2_liver_2.view(b,f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_2.view(b,f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_2.view(b,f,c,h,w)
        X_tv_liver_2 =X_tv_liver_2.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        # X_tv_tumor_2 = self.tumor_hgnn_fusion_v_2(X_tv_tumor_2)
        # X_t2_tumor_2 = self.tumor_hgnn_fusion_t2_2(X_t2_tumor_2)
        # X_tv_liver_2 = self.liver_hgnn_fusion_v_2(X_tv_liver_2)
        # X_t2_liver_2 = self.liver_hgnn_fusion_t2_2(X_t2_liver_2)
        # ## 肝脏肿瘤融合
        # X_tv_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_v_2(X_tv_tumor_2,X_tv_liver_2)
        # X_t2_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_t2_2(X_t2_tumor_2,X_t2_liver_2)
        ## 模态间融合
        # X_tv_liver_2,X_t2_liver_2=self.liver_fusion_v_t2_2(X_tv_liver_2,X_t2_liver_2,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_2,X_t2_tumor_2=self.tumor_fusion_v_t2_2(X_tv_tumor_2,X_t2_tumor_2,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_3 =X_t2_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_2.view(b*f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_2.view(b*f,c,h,w)
        X_tv_liver_3 =X_tv_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =self.tumor_conv3_t2(X_t2_tumor_3)
        X_tv_tumor_3 =self.tumor_conv3_v(X_tv_tumor_3)
        X_tv_liver_3 =self.liver_conv3_v(X_tv_liver_3)
        X_t2_liver_3 =self.liver_conv3_t2(X_t2_liver_3)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_3.shape
        X_t2_liver_3 =X_t2_liver_3.view(b,f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_3.view(b,f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_3.view(b,f,c,h,w)
        X_tv_liver_3 =X_tv_liver_3.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        # X_tv_tumor_3 = self.tumor_hgnn_fusion_v_3(X_tv_tumor_3)
        # X_t2_tumor_3 = self.tumor_hgnn_fusion_t2_3(X_t2_tumor_3)
        # X_tv_liver_3 = self.liver_hgnn_fusion_v_3(X_tv_liver_3)
        # X_t2_liver_3 = self.liver_hgnn_fusion_t2_3(X_t2_liver_3)
        # ## 肝脏肿瘤融合
        # X_tv_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_v_3(X_tv_tumor_3,X_tv_liver_3)
        # X_t2_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_t2_3(X_t2_tumor_3,X_t2_liver_3)
        ## 模态间融合
        # X_tv_liver_3,X_t2_liver_3=self.liver_fusion_v_t2_3(X_tv_liver_3,X_t2_liver_3,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_3,X_t2_tumor_3=self.tumor_fusion_v_t2_3(X_tv_tumor_3,X_t2_tumor_3,tumor_tv_mask,tumor_t2_mask)
       
        
        X_t2_liver_4 =X_t2_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_3.view(b*f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_3.view(b*f,c,h,w)
        X_tv_liver_4 =X_tv_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =self.tumor_conv4_t2(X_t2_tumor_4)
        X_tv_tumor_4 =self.tumor_conv4_v(X_tv_tumor_4)
        X_tv_liver_4 =self.liver_conv4_v(X_tv_liver_4)
        X_t2_liver_4 =self.liver_conv4_t2(X_t2_liver_4)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_4.shape
        X_t2_liver_4 =X_t2_liver_4.view(b,f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_4.view(b,f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_4.view(b,f,c,h,w)
        X_tv_liver_4 =X_tv_liver_4.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w],mode='nearest')
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w],mode='nearest')
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w],mode='nearest')
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w],mode='nearest')

        # X_tv_tumor_4 = self.tumor_hgnn_fusion_v_4(X_tv_tumor_4)
        # X_t2_tumor_4 = self.tumor_hgnn_fusion_t2_4(X_t2_tumor_4)
        # X_tv_liver_4 = self.liver_hgnn_fusion_v_4(X_tv_liver_4)
        # X_t2_liver_4 = self.liver_hgnn_fusion_t2_4(X_t2_liver_4)
        # # 肝脏肿瘤融合
        # X_tv_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_v_4(X_tv_tumor_4,X_tv_liver_4)
        # X_t2_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_t2_4(X_t2_tumor_4,X_t2_liver_4)
        # ## 模态间融合
        # X_tv_liver_4,X_t2_liver_4=self.liver_fusion_v_t2_4(X_tv_liver_4,X_t2_liver_4,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_4,X_t2_tumor_4=self.tumor_fusion_v_t2_4(X_tv_tumor_4,X_t2_tumor_4,tumor_tv_mask,tumor_t2_mask)
       


        tumor_tv_mask_pool = self.pool(tumor_tv_mask).squeeze()
        tumor_t2_mask_pool = self.pool(tumor_t2_mask).squeeze()
        liver_tv_mask_pool = self.pool(liver_tv_mask).squeeze()
        liver_t2_mask_pool = self.pool(liver_t2_mask).squeeze()
        # # X_liver_v_gate = self.pool(X_liver_v_gate).squeeze()
        # # X_liver_t2_gate = self.pool(X_liver_t2_gate).squeeze()

        X_liver_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_tumor_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_liver_t2_pool = self.pool(X_t2_liver_4).squeeze()
        X_tumor_t2_pool = self.pool(X_t2_liver_4).squeeze()
        
        
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        # X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        X_pool = torch.cat([ X_tumor_v_pool,  X_tumor_t2_pool,X_liver_v_pool,X_liver_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool,liver_t2_mask_pool,liver_tv_mask_pool],dim=-1)

        # x = self.fc1(X_pool) 
        if f ==10:
            x = self.fc1(x)
        elif f ==5:
            x = self.fc1_5(x)
        # x = self.fc_mask(x)
        # x = F.relu(x,inplace=True)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        # execution_time = end_time - start_time
        # print(f"代码执行时间为：{execution_time}秒")  
        return x

class CNN3d_t2_tv_hgnn_0414_baseline_neibu(nn.Module):
    """
    从0414 构建多阶段融合
    """
    def __init__(self,num_classes=1,points=1000, num_channels =128):
        super(CNN3d_t2_tv_hgnn_0414_baseline_neibu, self).__init__()
        ## one stage     
        self.liver_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_v = Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv1_t2 =  Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')

        self.tumor_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_v =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')                                                 
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM  = nn.LSTM(
                input_size=num_channels*4,
                hidden_size=128,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
              
        self.fc1 = nn.Linear(168, 32)
        self.fc1_5 = nn.Linear(148, 32)
        
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.stem_v = Stem(out_dim=640) ## 特征提取器
        self.stem_t2 = Stem(out_dim=640)
        self.hyper2d_tv = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_tv = FFN(640,640*4)
        self.hyper2d_t2 = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_t2 = FFN(640,640*4)
        ## 换一种方式的超图卷积 注意 这种超图卷积的输入是没有batch维度，所以每一个样本 单独初始化一个超图
        # self.layers2= HGNNConv(hid_channels=16, num_classes, is_last=True)
        ## 肿瘤内部空间融合
        self.tumor_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.tumor_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_tumor_fusion_t2_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 模态间融合
        self.tumor_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        self.liver_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        
        self.tumor_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.tumor_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_tumor_fusion_t2_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 模态间融合
        self.tumor_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)
        self.liver_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)

        self.tumor_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.tumor_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_tumor_fusion_t2_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 模态间融合
        self.tumor_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)
        self.liver_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)


        self.tumor_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.tumor_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_tumor_fusion_t2_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 模态间融合
        self.tumor_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        self.liver_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        


    def forward(self,X_image_mask):
        # torch.autograd.set_detect_anomaly(True)
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_t2_liver,X_t2_tumor,liver_t2_mask,tumor_t2_mask,X_tv_liver,X_tv_tumor,liver_tv_mask,tumor_tv_mask=X_image_mask
        b,f,c,h,w = X_tv_tumor.shape
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image= X_V_mask.to(torch.float32),X_V_image.to(torch.float32), X_T2_mask.to(torch.float32),X_T2_image.to(torch.float32)
        X_tumor_v_1 = []
        # X_liver_t2_1 = []
        X_tumor_t2_1 = []
        tumor_tv_mask = tumor_tv_mask.squeeze()
        tumor_t2_mask = tumor_t2_mask.squeeze()
        liver_t2_mask =liver_t2_mask.squeeze()
        liver_tv_mask =liver_tv_mask.squeeze()
        ## 为了gradcam 更好，先将b f合并
        '''
        for i in range(f):
            x_t2_liver_2d = X_t2_liver[:, i,:]
            x_t2_tumor_2d = X_t2_tumor[:, i,:]
            x_tv_liver_2d = X_tv_liver[:, i,:]
            x_tv_tumor_2d = X_tv_tumor[:, i,:]
            
            x_tumor_v =self.tumor_conv1_v(x_tv_tumor_2d)
            x_tumor_t2 =self.tumor_conv1_t2(x_t2_tumor_2d)
            x_liver_v =self.liver_conv1_v(x_tv_tumor_2d)
            x_liver_t2 =self.liver_conv1_t2(x_t2_tumor_2d)

            X_tumor_v_1.append(x_tumor_v)

            # X_liver_v_1.append(x_liver_v)
            X_tumor_t2_1.append(x_tumor_t2)
            # X_liver_t2_1.append(x_liver_t2)
        
        X_tumor_v_1 = torch.stack(X_tumor_v_1, dim=1) 
        # X_liver_t2_1 = torch.stack(X_liver_t2_1, dim=1)
        X_tumor_t2_1 = torch.stack(X_tumor_t2_1, dim=1) 
        '''
        X_t2_liver =X_t2_liver.view(b*f,c,h,w)
        X_t2_tumor =X_t2_tumor.view(b*f,c,h,w)
        X_tv_tumor =X_tv_tumor.view(b*f,c,h,w)
        X_tv_liver =X_tv_liver.view(b*f,c,h,w)
        X_t2_tumor =self.tumor_conv1_t2(X_t2_tumor)
        X_tv_tumor =self.tumor_conv1_v(X_tv_tumor)
        X_tv_liver =self.liver_conv1_v(X_tv_liver)
        X_t2_liver =self.liver_conv1_t2(X_t2_liver)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor.shape
        X_t2_liver_1 =X_t2_liver.view(b,f,c,h,w)
        X_t2_tumor_1 =X_t2_tumor.view(b,f,c,h,w)
        X_tv_tumor_1 =X_tv_tumor.view(b,f,c,h,w)
        X_tv_liver_1=X_tv_liver.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])
        ## 超图的构建中，怎么去采样是
        # new_X_tumor_v_1 = feature_selection(X_tumor_v_1, k=1000)
        # ## 内部融合
        X_tv_tumor_1 = self.tumor_hgnn_fusion_v_1(X_tv_tumor_1)
        X_t2_tumor_1 = self.tumor_hgnn_fusion_t2_1(X_t2_tumor_1)
        X_tv_liver_1 = self.liver_hgnn_fusion_v_1(X_tv_liver_1)
        X_t2_liver_1 = self.liver_hgnn_fusion_t2_1(X_t2_liver_1)
        # ## 肝脏肿瘤融合
        # start_time = time.time()
        # # 执行代码        
        # X_tv_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_v_1(X_tv_tumor_1,X_tv_liver_1)
        # X_t2_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_t2_1(X_t2_tumor_1,X_t2_liver_1)
        # end_time = time.time()
        # ## 模态间融合
        # X_tv_liver_1,X_t2_liver_1=self.liver_fusion_v_t2_1(X_tv_liver_1,X_t2_liver_1,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_1,X_t2_tumor_1=self.tumor_fusion_v_t2_1(X_tv_tumor_1,X_t2_tumor_1,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_2 =X_t2_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_1.view(b*f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_1.view(b*f,c,h,w)
        X_tv_liver_2 =X_tv_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =self.tumor_conv2_t2(X_t2_tumor_2)
        X_tv_tumor_2 =self.tumor_conv2_v(X_tv_tumor_2)
        X_tv_liver_2 =self.liver_conv2_v(X_tv_liver_2)
        X_t2_liver_2 =self.liver_conv2_t2(X_t2_liver_2)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_2.shape
        X_t2_liver_2 =X_t2_liver_2.view(b,f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_2.view(b,f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_2.view(b,f,c,h,w)
        X_tv_liver_2 =X_tv_liver_2.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_2 = self.tumor_hgnn_fusion_v_2(X_tv_tumor_2)
        X_t2_tumor_2 = self.tumor_hgnn_fusion_t2_2(X_t2_tumor_2)
        X_tv_liver_2 = self.liver_hgnn_fusion_v_2(X_tv_liver_2)
        X_t2_liver_2 = self.liver_hgnn_fusion_t2_2(X_t2_liver_2)
        # ## 肝脏肿瘤融合
        # X_tv_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_v_2(X_tv_tumor_2,X_tv_liver_2)
        # X_t2_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_t2_2(X_t2_tumor_2,X_t2_liver_2)
        ## 模态间融合
        # X_tv_liver_2,X_t2_liver_2=self.liver_fusion_v_t2_2(X_tv_liver_2,X_t2_liver_2,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_2,X_t2_tumor_2=self.tumor_fusion_v_t2_2(X_tv_tumor_2,X_t2_tumor_2,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_3 =X_t2_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_2.view(b*f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_2.view(b*f,c,h,w)
        X_tv_liver_3 =X_tv_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =self.tumor_conv3_t2(X_t2_tumor_3)
        X_tv_tumor_3 =self.tumor_conv3_v(X_tv_tumor_3)
        X_tv_liver_3 =self.liver_conv3_v(X_tv_liver_3)
        X_t2_liver_3 =self.liver_conv3_t2(X_t2_liver_3)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_3.shape
        X_t2_liver_3 =X_t2_liver_3.view(b,f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_3.view(b,f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_3.view(b,f,c,h,w)
        X_tv_liver_3 =X_tv_liver_3.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_3 = self.tumor_hgnn_fusion_v_3(X_tv_tumor_3)
        X_t2_tumor_3 = self.tumor_hgnn_fusion_t2_3(X_t2_tumor_3)
        X_tv_liver_3 = self.liver_hgnn_fusion_v_3(X_tv_liver_3)
        X_t2_liver_3 = self.liver_hgnn_fusion_t2_3(X_t2_liver_3)
        # ## 肝脏肿瘤融合
        # X_tv_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_v_3(X_tv_tumor_3,X_tv_liver_3)
        # X_t2_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_t2_3(X_t2_tumor_3,X_t2_liver_3)
        ## 模态间融合
        # X_tv_liver_3,X_t2_liver_3=self.liver_fusion_v_t2_3(X_tv_liver_3,X_t2_liver_3,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_3,X_t2_tumor_3=self.tumor_fusion_v_t2_3(X_tv_tumor_3,X_t2_tumor_3,tumor_tv_mask,tumor_t2_mask)
       
        
        X_t2_liver_4 =X_t2_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_3.view(b*f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_3.view(b*f,c,h,w)
        X_tv_liver_4 =X_tv_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =self.tumor_conv4_t2(X_t2_tumor_4)
        X_tv_tumor_4 =self.tumor_conv4_v(X_tv_tumor_4)
        X_tv_liver_4 =self.liver_conv4_v(X_tv_liver_4)
        X_t2_liver_4 =self.liver_conv4_t2(X_t2_liver_4)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_4.shape
        X_t2_liver_4 =X_t2_liver_4.view(b,f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_4.view(b,f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_4.view(b,f,c,h,w)
        X_tv_liver_4 =X_tv_liver_4.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w],mode='nearest')
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w],mode='nearest')
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w],mode='nearest')
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w],mode='nearest')

        X_tv_tumor_4 = self.tumor_hgnn_fusion_v_4(X_tv_tumor_4)
        X_t2_tumor_4 = self.tumor_hgnn_fusion_t2_4(X_t2_tumor_4)
        X_tv_liver_4 = self.liver_hgnn_fusion_v_4(X_tv_liver_4)
        X_t2_liver_4 = self.liver_hgnn_fusion_t2_4(X_t2_liver_4)
        # # 肝脏肿瘤融合
        # X_tv_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_v_4(X_tv_tumor_4,X_tv_liver_4)
        # X_t2_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_t2_4(X_t2_tumor_4,X_t2_liver_4)
        # ## 模态间融合
        # X_tv_liver_4,X_t2_liver_4=self.liver_fusion_v_t2_4(X_tv_liver_4,X_t2_liver_4,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_4,X_t2_tumor_4=self.tumor_fusion_v_t2_4(X_tv_tumor_4,X_t2_tumor_4,tumor_tv_mask,tumor_t2_mask)
       


        tumor_tv_mask_pool = self.pool(tumor_tv_mask).squeeze()
        tumor_t2_mask_pool = self.pool(tumor_t2_mask).squeeze()
        liver_tv_mask_pool = self.pool(liver_tv_mask).squeeze()
        liver_t2_mask_pool = self.pool(liver_t2_mask).squeeze()
        # # X_liver_v_gate = self.pool(X_liver_v_gate).squeeze()
        # # X_liver_t2_gate = self.pool(X_liver_t2_gate).squeeze()

        X_liver_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_tumor_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_liver_t2_pool = self.pool(X_t2_liver_4).squeeze()
        X_tumor_t2_pool = self.pool(X_t2_liver_4).squeeze()
        
        
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        # X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        X_pool = torch.cat([ X_tumor_v_pool,  X_tumor_t2_pool,X_liver_v_pool,X_liver_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool,liver_t2_mask_pool,liver_tv_mask_pool],dim=-1)

        # x = self.fc1(X_pool) 
        if f ==10:
            x = self.fc1(x)
        elif f ==5:
            x = self.fc1_5(x)
        # x = self.fc_mask(x)
        # x = F.relu(x,inplace=True)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        # execution_time = end_time - start_time
        # print(f"代码执行时间为：{execution_time}秒")  
        return x

class CNN3d_t2_tv_hgnn_0414_baseline_liver_tumor(nn.Module):
    """
    从0414 构建多阶段融合
    """
    def __init__(self,num_classes=1,points=1000, num_channels =128):
        super(CNN3d_t2_tv_hgnn_0414_baseline_liver_tumor, self).__init__()
        ## one stage     
        self.liver_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_v = Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv1_t2 =  Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')

        self.tumor_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_v =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')                                                 
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM  = nn.LSTM(
                input_size=num_channels*4,
                hidden_size=128,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
              
        self.fc1 = nn.Linear(168, 32)
        self.fc1_5 = nn.Linear(148, 32)
        
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.stem_v = Stem(out_dim=640) ## 特征提取器
        self.stem_t2 = Stem(out_dim=640)
        self.hyper2d_tv = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_tv = FFN(640,640*4)
        self.hyper2d_t2 = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_t2 = FFN(640,640*4)
        ## 换一种方式的超图卷积 注意 这种超图卷积的输入是没有batch维度，所以每一个样本 单独初始化一个超图
        # self.layers2= HGNNConv(hid_channels=16, num_classes, is_last=True)
        ## 肿瘤内部空间融合
        self.tumor_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.tumor_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_tumor_fusion_t2_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 模态间融合
        self.tumor_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        self.liver_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        
        self.tumor_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.tumor_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_tumor_fusion_t2_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 模态间融合
        self.tumor_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)
        self.liver_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)

        self.tumor_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.tumor_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_tumor_fusion_t2_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 模态间融合
        self.tumor_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)
        self.liver_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)


        self.tumor_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.tumor_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_tumor_fusion_t2_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 模态间融合
        self.tumor_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        self.liver_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        


    def forward(self,X_image_mask):
        # torch.autograd.set_detect_anomaly(True)
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_t2_liver,X_t2_tumor,liver_t2_mask,tumor_t2_mask,X_tv_liver,X_tv_tumor,liver_tv_mask,tumor_tv_mask=X_image_mask
        b,f,c,h,w = X_tv_tumor.shape
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image= X_V_mask.to(torch.float32),X_V_image.to(torch.float32), X_T2_mask.to(torch.float32),X_T2_image.to(torch.float32)
        X_tumor_v_1 = []
        # X_liver_t2_1 = []
        X_tumor_t2_1 = []
        tumor_tv_mask = tumor_tv_mask.squeeze()
        tumor_t2_mask = tumor_t2_mask.squeeze()
        liver_t2_mask =liver_t2_mask.squeeze()
        liver_tv_mask =liver_tv_mask.squeeze()
        ## 为了gradcam 更好，先将b f合并
        '''
        for i in range(f):
            x_t2_liver_2d = X_t2_liver[:, i,:]
            x_t2_tumor_2d = X_t2_tumor[:, i,:]
            x_tv_liver_2d = X_tv_liver[:, i,:]
            x_tv_tumor_2d = X_tv_tumor[:, i,:]
            
            x_tumor_v =self.tumor_conv1_v(x_tv_tumor_2d)
            x_tumor_t2 =self.tumor_conv1_t2(x_t2_tumor_2d)
            x_liver_v =self.liver_conv1_v(x_tv_tumor_2d)
            x_liver_t2 =self.liver_conv1_t2(x_t2_tumor_2d)

            X_tumor_v_1.append(x_tumor_v)

            # X_liver_v_1.append(x_liver_v)
            X_tumor_t2_1.append(x_tumor_t2)
            # X_liver_t2_1.append(x_liver_t2)
        
        X_tumor_v_1 = torch.stack(X_tumor_v_1, dim=1) 
        # X_liver_t2_1 = torch.stack(X_liver_t2_1, dim=1)
        X_tumor_t2_1 = torch.stack(X_tumor_t2_1, dim=1) 
        '''
        X_t2_liver =X_t2_liver.view(b*f,c,h,w)
        X_t2_tumor =X_t2_tumor.view(b*f,c,h,w)
        X_tv_tumor =X_tv_tumor.view(b*f,c,h,w)
        X_tv_liver =X_tv_liver.view(b*f,c,h,w)
        X_t2_tumor =self.tumor_conv1_t2(X_t2_tumor)
        X_tv_tumor =self.tumor_conv1_v(X_tv_tumor)
        X_tv_liver =self.liver_conv1_v(X_tv_liver)
        X_t2_liver =self.liver_conv1_t2(X_t2_liver)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor.shape
        X_t2_liver_1 =X_t2_liver.view(b,f,c,h,w)
        X_t2_tumor_1 =X_t2_tumor.view(b,f,c,h,w)
        X_tv_tumor_1 =X_tv_tumor.view(b,f,c,h,w)
        X_tv_liver_1=X_tv_liver.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])
        ## 超图的构建中，怎么去采样是
        # new_X_tumor_v_1 = feature_selection(X_tumor_v_1, k=1000)
        # ## 内部融合
        # X_tv_tumor_1 = self.tumor_hgnn_fusion_v_1(X_tv_tumor_1)
        # X_t2_tumor_1 = self.tumor_hgnn_fusion_t2_1(X_t2_tumor_1)
        # X_tv_liver_1 = self.liver_hgnn_fusion_v_1(X_tv_liver_1)
        # X_t2_liver_1 = self.liver_hgnn_fusion_t2_1(X_t2_liver_1)
        # ## 肝脏肿瘤融合
        # start_time = time.time()
        # # 执行代码        
        X_tv_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_v_1(X_tv_tumor_1,X_tv_liver_1)
        X_t2_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_t2_1(X_t2_tumor_1,X_t2_liver_1)
        # end_time = time.time()
        # ## 模态间融合
        # X_tv_liver_1,X_t2_liver_1=self.liver_fusion_v_t2_1(X_tv_liver_1,X_t2_liver_1,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_1,X_t2_tumor_1=self.tumor_fusion_v_t2_1(X_tv_tumor_1,X_t2_tumor_1,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_2 =X_t2_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_1.view(b*f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_1.view(b*f,c,h,w)
        X_tv_liver_2 =X_tv_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =self.tumor_conv2_t2(X_t2_tumor_2)
        X_tv_tumor_2 =self.tumor_conv2_v(X_tv_tumor_2)
        X_tv_liver_2 =self.liver_conv2_v(X_tv_liver_2)
        X_t2_liver_2 =self.liver_conv2_t2(X_t2_liver_2)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_2.shape
        X_t2_liver_2 =X_t2_liver_2.view(b,f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_2.view(b,f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_2.view(b,f,c,h,w)
        X_tv_liver_2 =X_tv_liver_2.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        # X_tv_tumor_2 = self.tumor_hgnn_fusion_v_2(X_tv_tumor_2)
        # X_t2_tumor_2 = self.tumor_hgnn_fusion_t2_2(X_t2_tumor_2)
        # X_tv_liver_2 = self.liver_hgnn_fusion_v_2(X_tv_liver_2)
        # X_t2_liver_2 = self.liver_hgnn_fusion_t2_2(X_t2_liver_2)
        # ## 肝脏肿瘤融合
        X_tv_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_v_2(X_tv_tumor_2,X_tv_liver_2)
        X_t2_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_t2_2(X_t2_tumor_2,X_t2_liver_2)
        ## 模态间融合
        # X_tv_liver_2,X_t2_liver_2=self.liver_fusion_v_t2_2(X_tv_liver_2,X_t2_liver_2,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_2,X_t2_tumor_2=self.tumor_fusion_v_t2_2(X_tv_tumor_2,X_t2_tumor_2,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_3 =X_t2_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_2.view(b*f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_2.view(b*f,c,h,w)
        X_tv_liver_3 =X_tv_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =self.tumor_conv3_t2(X_t2_tumor_3)
        X_tv_tumor_3 =self.tumor_conv3_v(X_tv_tumor_3)
        X_tv_liver_3 =self.liver_conv3_v(X_tv_liver_3)
        X_t2_liver_3 =self.liver_conv3_t2(X_t2_liver_3)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_3.shape
        X_t2_liver_3 =X_t2_liver_3.view(b,f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_3.view(b,f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_3.view(b,f,c,h,w)
        X_tv_liver_3 =X_tv_liver_3.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        # X_tv_tumor_3 = self.tumor_hgnn_fusion_v_3(X_tv_tumor_3)
        # X_t2_tumor_3 = self.tumor_hgnn_fusion_t2_3(X_t2_tumor_3)
        # X_tv_liver_3 = self.liver_hgnn_fusion_v_3(X_tv_liver_3)
        # X_t2_liver_3 = self.liver_hgnn_fusion_t2_3(X_t2_liver_3)
        # ## 肝脏肿瘤融合
        X_tv_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_v_3(X_tv_tumor_3,X_tv_liver_3)
        X_t2_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_t2_3(X_t2_tumor_3,X_t2_liver_3)
        ## 模态间融合
        # X_tv_liver_3,X_t2_liver_3=self.liver_fusion_v_t2_3(X_tv_liver_3,X_t2_liver_3,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_3,X_t2_tumor_3=self.tumor_fusion_v_t2_3(X_tv_tumor_3,X_t2_tumor_3,tumor_tv_mask,tumor_t2_mask)
       
        
        X_t2_liver_4 =X_t2_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_3.view(b*f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_3.view(b*f,c,h,w)
        X_tv_liver_4 =X_tv_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =self.tumor_conv4_t2(X_t2_tumor_4)
        X_tv_tumor_4 =self.tumor_conv4_v(X_tv_tumor_4)
        X_tv_liver_4 =self.liver_conv4_v(X_tv_liver_4)
        X_t2_liver_4 =self.liver_conv4_t2(X_t2_liver_4)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_4.shape
        X_t2_liver_4 =X_t2_liver_4.view(b,f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_4.view(b,f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_4.view(b,f,c,h,w)
        X_tv_liver_4 =X_tv_liver_4.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w],mode='nearest')
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w],mode='nearest')
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w],mode='nearest')
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w],mode='nearest')

        # X_tv_tumor_4 = self.tumor_hgnn_fusion_v_4(X_tv_tumor_4)
        # X_t2_tumor_4 = self.tumor_hgnn_fusion_t2_4(X_t2_tumor_4)
        # X_tv_liver_4 = self.liver_hgnn_fusion_v_4(X_tv_liver_4)
        # X_t2_liver_4 = self.liver_hgnn_fusion_t2_4(X_t2_liver_4)
        # # 肝脏肿瘤融合
        X_tv_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_v_4(X_tv_tumor_4,X_tv_liver_4)
        X_t2_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_t2_4(X_t2_tumor_4,X_t2_liver_4)
        # ## 模态间融合
        # X_tv_liver_4,X_t2_liver_4=self.liver_fusion_v_t2_4(X_tv_liver_4,X_t2_liver_4,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_4,X_t2_tumor_4=self.tumor_fusion_v_t2_4(X_tv_tumor_4,X_t2_tumor_4,tumor_tv_mask,tumor_t2_mask)
       


        tumor_tv_mask_pool = self.pool(tumor_tv_mask).squeeze()
        tumor_t2_mask_pool = self.pool(tumor_t2_mask).squeeze()
        liver_tv_mask_pool = self.pool(liver_tv_mask).squeeze()
        liver_t2_mask_pool = self.pool(liver_t2_mask).squeeze()
        # # X_liver_v_gate = self.pool(X_liver_v_gate).squeeze()
        # # X_liver_t2_gate = self.pool(X_liver_t2_gate).squeeze()

        X_liver_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_tumor_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_liver_t2_pool = self.pool(X_t2_liver_4).squeeze()
        X_tumor_t2_pool = self.pool(X_t2_liver_4).squeeze()
        
        
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        # X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        X_pool = torch.cat([ X_tumor_v_pool,  X_tumor_t2_pool,X_liver_v_pool,X_liver_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool,liver_t2_mask_pool,liver_tv_mask_pool],dim=-1)

        # x = self.fc1(X_pool) 
        if f ==10:
            x = self.fc1(x)
        elif f ==5:
            x = self.fc1_5(x)
        # x = self.fc_mask(x)
        # x = F.relu(x,inplace=True)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        # execution_time = end_time - start_time
        # print(f"代码执行时间为：{execution_time}秒")  
        return x

class CNN3d_t2_tv_hgnn_0414_baseline_multi_model_top(nn.Module):
    """
    从0414 构建多阶段融合
    """
    def __init__(self,num_classes=1,points=1000, num_channels =128):
        super(CNN3d_t2_tv_hgnn_0414_baseline_multi_model_top, self).__init__()
        ## one stage     
        self.liver_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_v = Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv1_t2 =  Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')

        self.tumor_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_v =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')                                                 
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM  = nn.LSTM(
                input_size=num_channels*4,
                hidden_size=128,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
              
        self.fc1 = nn.Linear(168, 32)
        self.fc1_5 = nn.Linear(148, 32)
        
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.stem_v = Stem(out_dim=640) ## 特征提取器
        self.stem_t2 = Stem(out_dim=640)
        self.hyper2d_tv = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_tv = FFN(640,640*4)
        self.hyper2d_t2 = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_t2 = FFN(640,640*4)
        ## 换一种方式的超图卷积 注意 这种超图卷积的输入是没有batch维度，所以每一个样本 单独初始化一个超图
        # self.layers2= HGNNConv(hid_channels=16, num_classes, is_last=True)
        ## 肿瘤内部空间融合
        self.tumor_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.tumor_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_tumor_fusion_t2_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 模态间融合
        self.tumor_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        self.liver_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        
        self.tumor_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.tumor_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_tumor_fusion_t2_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 模态间融合
        self.tumor_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)
        self.liver_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)

        self.tumor_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.tumor_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_tumor_fusion_t2_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 模态间融合
        self.tumor_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)
        self.liver_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)


        self.tumor_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.tumor_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_tumor_fusion_t2_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 模态间融合
        self.tumor_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        self.liver_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        


    def forward(self,X_image_mask):
        # torch.autograd.set_detect_anomaly(True)
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_t2_liver,X_t2_tumor,liver_t2_mask,tumor_t2_mask,X_tv_liver,X_tv_tumor,liver_tv_mask,tumor_tv_mask=X_image_mask
        b,f,c,h,w = X_tv_tumor.shape
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image= X_V_mask.to(torch.float32),X_V_image.to(torch.float32), X_T2_mask.to(torch.float32),X_T2_image.to(torch.float32)
        X_tumor_v_1 = []
        # X_liver_t2_1 = []
        X_tumor_t2_1 = []
        tumor_tv_mask = tumor_tv_mask.squeeze()
        tumor_t2_mask = tumor_t2_mask.squeeze()
        liver_t2_mask =liver_t2_mask.squeeze()
        liver_tv_mask =liver_tv_mask.squeeze()
        ## 为了gradcam 更好，先将b f合并
        '''
        for i in range(f):
            x_t2_liver_2d = X_t2_liver[:, i,:]
            x_t2_tumor_2d = X_t2_tumor[:, i,:]
            x_tv_liver_2d = X_tv_liver[:, i,:]
            x_tv_tumor_2d = X_tv_tumor[:, i,:]
            
            x_tumor_v =self.tumor_conv1_v(x_tv_tumor_2d)
            x_tumor_t2 =self.tumor_conv1_t2(x_t2_tumor_2d)
            x_liver_v =self.liver_conv1_v(x_tv_tumor_2d)
            x_liver_t2 =self.liver_conv1_t2(x_t2_tumor_2d)

            X_tumor_v_1.append(x_tumor_v)

            # X_liver_v_1.append(x_liver_v)
            X_tumor_t2_1.append(x_tumor_t2)
            # X_liver_t2_1.append(x_liver_t2)
        
        X_tumor_v_1 = torch.stack(X_tumor_v_1, dim=1) 
        # X_liver_t2_1 = torch.stack(X_liver_t2_1, dim=1)
        X_tumor_t2_1 = torch.stack(X_tumor_t2_1, dim=1) 
        '''
        X_t2_liver =X_t2_liver.view(b*f,c,h,w)
        X_t2_tumor =X_t2_tumor.view(b*f,c,h,w)
        X_tv_tumor =X_tv_tumor.view(b*f,c,h,w)
        X_tv_liver =X_tv_liver.view(b*f,c,h,w)
        X_t2_tumor =self.tumor_conv1_t2(X_t2_tumor)
        X_tv_tumor =self.tumor_conv1_v(X_tv_tumor)
        X_tv_liver =self.liver_conv1_v(X_tv_liver)
        X_t2_liver =self.liver_conv1_t2(X_t2_liver)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor.shape
        X_t2_liver_1 =X_t2_liver.view(b,f,c,h,w)
        X_t2_tumor_1 =X_t2_tumor.view(b,f,c,h,w)
        X_tv_tumor_1 =X_tv_tumor.view(b,f,c,h,w)
        X_tv_liver_1=X_tv_liver.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])
        ## 超图的构建中，怎么去采样是
        # new_X_tumor_v_1 = feature_selection(X_tumor_v_1, k=1000)
        # ## 内部融合
        # X_tv_tumor_1 = self.tumor_hgnn_fusion_v_1(X_tv_tumor_1)
        # X_t2_tumor_1 = self.tumor_hgnn_fusion_t2_1(X_t2_tumor_1)
        # X_tv_liver_1 = self.liver_hgnn_fusion_v_1(X_tv_liver_1)
        # X_t2_liver_1 = self.liver_hgnn_fusion_t2_1(X_t2_liver_1)
        # ## 肝脏肿瘤融合
        # start_time = time.time()
        # # 执行代码        
        # X_tv_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_v_1(X_tv_tumor_1,X_tv_liver_1)
        # X_t2_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_t2_1(X_t2_tumor_1,X_t2_liver_1)
        # end_time = time.time()
        # ## 模态间融合
        X_tv_liver_1,X_t2_liver_1=self.liver_fusion_v_t2_1(X_tv_liver_1,X_t2_liver_1,liver_tv_mask,liver_t2_mask)
        X_tv_tumor_1,X_t2_tumor_1=self.tumor_fusion_v_t2_1(X_tv_tumor_1,X_t2_tumor_1,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_2 =X_t2_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_1.view(b*f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_1.view(b*f,c,h,w)
        X_tv_liver_2 =X_tv_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =self.tumor_conv2_t2(X_t2_tumor_2)
        X_tv_tumor_2 =self.tumor_conv2_v(X_tv_tumor_2)
        X_tv_liver_2 =self.liver_conv2_v(X_tv_liver_2)
        X_t2_liver_2 =self.liver_conv2_t2(X_t2_liver_2)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_2.shape
        X_t2_liver_2 =X_t2_liver_2.view(b,f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_2.view(b,f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_2.view(b,f,c,h,w)
        X_tv_liver_2 =X_tv_liver_2.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        # X_tv_tumor_2 = self.tumor_hgnn_fusion_v_2(X_tv_tumor_2)
        # X_t2_tumor_2 = self.tumor_hgnn_fusion_t2_2(X_t2_tumor_2)
        # X_tv_liver_2 = self.liver_hgnn_fusion_v_2(X_tv_liver_2)
        # X_t2_liver_2 = self.liver_hgnn_fusion_t2_2(X_t2_liver_2)
        # # ## 肝脏肿瘤融合
        # X_tv_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_v_2(X_tv_tumor_2,X_tv_liver_2)
        # X_t2_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_t2_2(X_t2_tumor_2,X_t2_liver_2)
        # 模态间融合
        X_tv_liver_2,X_t2_liver_2=self.liver_fusion_v_t2_2(X_tv_liver_2,X_t2_liver_2,liver_tv_mask,liver_t2_mask)
        X_tv_tumor_2,X_t2_tumor_2=self.tumor_fusion_v_t2_2(X_tv_tumor_2,X_t2_tumor_2,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_3 =X_t2_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_2.view(b*f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_2.view(b*f,c,h,w)
        X_tv_liver_3 =X_tv_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =self.tumor_conv3_t2(X_t2_tumor_3)
        X_tv_tumor_3 =self.tumor_conv3_v(X_tv_tumor_3)
        X_tv_liver_3 =self.liver_conv3_v(X_tv_liver_3)
        X_t2_liver_3 =self.liver_conv3_t2(X_t2_liver_3)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_3.shape
        X_t2_liver_3 =X_t2_liver_3.view(b,f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_3.view(b,f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_3.view(b,f,c,h,w)
        X_tv_liver_3 =X_tv_liver_3.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        # X_tv_tumor_3 = self.tumor_hgnn_fusion_v_3(X_tv_tumor_3)
        # X_t2_tumor_3 = self.tumor_hgnn_fusion_t2_3(X_t2_tumor_3)
        # X_tv_liver_3 = self.liver_hgnn_fusion_v_3(X_tv_liver_3)
        # X_t2_liver_3 = self.liver_hgnn_fusion_t2_3(X_t2_liver_3)
        # # ## 肝脏肿瘤融合
        # X_tv_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_v_3(X_tv_tumor_3,X_tv_liver_3)
        # X_t2_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_t2_3(X_t2_tumor_3,X_t2_liver_3)
        ## 模态间融合
        X_tv_liver_3,X_t2_liver_3=self.liver_fusion_v_t2_3(X_tv_liver_3,X_t2_liver_3,liver_tv_mask,liver_t2_mask)
        X_tv_tumor_3,X_t2_tumor_3=self.tumor_fusion_v_t2_3(X_tv_tumor_3,X_t2_tumor_3,tumor_tv_mask,tumor_t2_mask)
       
        
        X_t2_liver_4 =X_t2_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_3.view(b*f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_3.view(b*f,c,h,w)
        X_tv_liver_4 =X_tv_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =self.tumor_conv4_t2(X_t2_tumor_4)
        X_tv_tumor_4 =self.tumor_conv4_v(X_tv_tumor_4)
        X_tv_liver_4 =self.liver_conv4_v(X_tv_liver_4)
        X_t2_liver_4 =self.liver_conv4_t2(X_t2_liver_4)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_4.shape
        X_t2_liver_4 =X_t2_liver_4.view(b,f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_4.view(b,f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_4.view(b,f,c,h,w)
        X_tv_liver_4 =X_tv_liver_4.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w],mode='nearest')
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w],mode='nearest')
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w],mode='nearest')
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w],mode='nearest')

        # X_tv_tumor_4 = self.tumor_hgnn_fusion_v_4(X_tv_tumor_4)
        # X_t2_tumor_4 = self.tumor_hgnn_fusion_t2_4(X_t2_tumor_4)
        # X_tv_liver_4 = self.liver_hgnn_fusion_v_4(X_tv_liver_4)
        # X_t2_liver_4 = self.liver_hgnn_fusion_t2_4(X_t2_liver_4)
        # # # 肝脏肿瘤融合
        # X_tv_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_v_4(X_tv_tumor_4,X_tv_liver_4)
        # X_t2_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_t2_4(X_t2_tumor_4,X_t2_liver_4)
        # ## 模态间融合
        X_tv_liver_4,X_t2_liver_4=self.liver_fusion_v_t2_4(X_tv_liver_4,X_t2_liver_4,liver_tv_mask,liver_t2_mask)
        X_tv_tumor_4,X_t2_tumor_4=self.tumor_fusion_v_t2_4(X_tv_tumor_4,X_t2_tumor_4,tumor_tv_mask,tumor_t2_mask)
       


        tumor_tv_mask_pool = self.pool(tumor_tv_mask).squeeze()
        tumor_t2_mask_pool = self.pool(tumor_t2_mask).squeeze()
        liver_tv_mask_pool = self.pool(liver_tv_mask).squeeze()
        liver_t2_mask_pool = self.pool(liver_t2_mask).squeeze()
        # # X_liver_v_gate = self.pool(X_liver_v_gate).squeeze()
        # # X_liver_t2_gate = self.pool(X_liver_t2_gate).squeeze()

        X_liver_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_tumor_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_liver_t2_pool = self.pool(X_t2_liver_4).squeeze()
        X_tumor_t2_pool = self.pool(X_t2_liver_4).squeeze()
        
        
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        # X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        X_pool = torch.cat([ X_tumor_v_pool,  X_tumor_t2_pool,X_liver_v_pool,X_liver_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool,liver_t2_mask_pool,liver_tv_mask_pool],dim=-1)

        # x = self.fc1(X_pool) 
        if f ==10:
            x = self.fc1(x)
        elif f ==5:
            x = self.fc1_5(x)
        # x = self.fc_mask(x)
        # x = F.relu(x,inplace=True)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        # execution_time = end_time - start_time
        # print(f"代码执行时间为：{execution_time}秒")  
        return x


class CNN3d_t2_tv_hgnn_0414_baseline_neibu_liver_tumor(nn.Module):
    """
    从0414 构建多阶段融合
    """
    def __init__(self,num_classes=1,points=1000, num_channels =128):
        super(CNN3d_t2_tv_hgnn_0414_baseline_neibu_liver_tumor, self).__init__()
        ## one stage     
        self.liver_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_v = Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv1_t2 =  Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')

        self.tumor_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_v =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')                                                 
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM  = nn.LSTM(
                input_size=num_channels*4,
                hidden_size=128,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
              
        self.fc1 = nn.Linear(168, 32)
        self.fc1_5 = nn.Linear(148, 32)
        
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.stem_v = Stem(out_dim=640) ## 特征提取器
        self.stem_t2 = Stem(out_dim=640)
        self.hyper2d_tv = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_tv = FFN(640,640*4)
        self.hyper2d_t2 = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_t2 = FFN(640,640*4)
        ## 换一种方式的超图卷积 注意 这种超图卷积的输入是没有batch维度，所以每一个样本 单独初始化一个超图
        # self.layers2= HGNNConv(hid_channels=16, num_classes, is_last=True)
        ## 肿瘤内部空间融合
        self.tumor_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.tumor_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_tumor_fusion_t2_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 模态间融合
        self.tumor_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        self.liver_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        
        self.tumor_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.tumor_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_tumor_fusion_t2_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 模态间融合
        self.tumor_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)
        self.liver_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)

        self.tumor_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.tumor_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_tumor_fusion_t2_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 模态间融合
        self.tumor_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)
        self.liver_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)


        self.tumor_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.tumor_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_tumor_fusion_t2_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 模态间融合
        self.tumor_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        self.liver_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        


    def forward(self,X_image_mask):
        # torch.autograd.set_detect_anomaly(True)
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_t2_liver,X_t2_tumor,liver_t2_mask,tumor_t2_mask,X_tv_liver,X_tv_tumor,liver_tv_mask,tumor_tv_mask=X_image_mask
        b,f,c,h,w = X_tv_tumor.shape
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image= X_V_mask.to(torch.float32),X_V_image.to(torch.float32), X_T2_mask.to(torch.float32),X_T2_image.to(torch.float32)
        X_tumor_v_1 = []
        # X_liver_t2_1 = []
        X_tumor_t2_1 = []
        tumor_tv_mask = tumor_tv_mask.squeeze()
        tumor_t2_mask = tumor_t2_mask.squeeze()
        liver_t2_mask =liver_t2_mask.squeeze()
        liver_tv_mask =liver_tv_mask.squeeze()
        ## 为了gradcam 更好，先将b f合并
        '''
        for i in range(f):
            x_t2_liver_2d = X_t2_liver[:, i,:]
            x_t2_tumor_2d = X_t2_tumor[:, i,:]
            x_tv_liver_2d = X_tv_liver[:, i,:]
            x_tv_tumor_2d = X_tv_tumor[:, i,:]
            
            x_tumor_v =self.tumor_conv1_v(x_tv_tumor_2d)
            x_tumor_t2 =self.tumor_conv1_t2(x_t2_tumor_2d)
            x_liver_v =self.liver_conv1_v(x_tv_tumor_2d)
            x_liver_t2 =self.liver_conv1_t2(x_t2_tumor_2d)

            X_tumor_v_1.append(x_tumor_v)

            # X_liver_v_1.append(x_liver_v)
            X_tumor_t2_1.append(x_tumor_t2)
            # X_liver_t2_1.append(x_liver_t2)
        
        X_tumor_v_1 = torch.stack(X_tumor_v_1, dim=1) 
        # X_liver_t2_1 = torch.stack(X_liver_t2_1, dim=1)
        X_tumor_t2_1 = torch.stack(X_tumor_t2_1, dim=1) 
        '''
        X_t2_liver =X_t2_liver.view(b*f,c,h,w)
        X_t2_tumor =X_t2_tumor.view(b*f,c,h,w)
        X_tv_tumor =X_tv_tumor.view(b*f,c,h,w)
        X_tv_liver =X_tv_liver.view(b*f,c,h,w)
        X_t2_tumor =self.tumor_conv1_t2(X_t2_tumor)
        X_tv_tumor =self.tumor_conv1_v(X_tv_tumor)
        X_tv_liver =self.liver_conv1_v(X_tv_liver)
        X_t2_liver =self.liver_conv1_t2(X_t2_liver)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor.shape
        X_t2_liver_1 =X_t2_liver.view(b,f,c,h,w)
        X_t2_tumor_1 =X_t2_tumor.view(b,f,c,h,w)
        X_tv_tumor_1 =X_tv_tumor.view(b,f,c,h,w)
        X_tv_liver_1=X_tv_liver.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])
        ## 超图的构建中，怎么去采样是
        # new_X_tumor_v_1 = feature_selection(X_tumor_v_1, k=1000)
        # ## 内部融合
        X_tv_tumor_1 = self.tumor_hgnn_fusion_v_1(X_tv_tumor_1)
        X_t2_tumor_1 = self.tumor_hgnn_fusion_t2_1(X_t2_tumor_1)
        X_tv_liver_1 = self.liver_hgnn_fusion_v_1(X_tv_liver_1)
        X_t2_liver_1 = self.liver_hgnn_fusion_t2_1(X_t2_liver_1)
        ## 肝脏肿瘤融合
        start_time = time.time()
        # 执行代码        
        X_tv_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_v_1(X_tv_tumor_1,X_tv_liver_1)
        X_t2_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_t2_1(X_t2_tumor_1,X_t2_liver_1)
        # end_time = time.time()
        # ## 模态间融合
        # X_tv_liver_1,X_t2_liver_1=self.liver_fusion_v_t2_1(X_tv_liver_1,X_t2_liver_1,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_1,X_t2_tumor_1=self.tumor_fusion_v_t2_1(X_tv_tumor_1,X_t2_tumor_1,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_2 =X_t2_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_1.view(b*f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_1.view(b*f,c,h,w)
        X_tv_liver_2 =X_tv_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =self.tumor_conv2_t2(X_t2_tumor_2)
        X_tv_tumor_2 =self.tumor_conv2_v(X_tv_tumor_2)
        X_tv_liver_2 =self.liver_conv2_v(X_tv_liver_2)
        X_t2_liver_2 =self.liver_conv2_t2(X_t2_liver_2)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_2.shape
        X_t2_liver_2 =X_t2_liver_2.view(b,f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_2.view(b,f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_2.view(b,f,c,h,w)
        X_tv_liver_2 =X_tv_liver_2.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_2 = self.tumor_hgnn_fusion_v_2(X_tv_tumor_2)
        X_t2_tumor_2 = self.tumor_hgnn_fusion_t2_2(X_t2_tumor_2)
        X_tv_liver_2 = self.liver_hgnn_fusion_v_2(X_tv_liver_2)
        X_t2_liver_2 = self.liver_hgnn_fusion_t2_2(X_t2_liver_2)
        ## 肝脏肿瘤融合
        X_tv_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_v_2(X_tv_tumor_2,X_tv_liver_2)
        X_t2_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_t2_2(X_t2_tumor_2,X_t2_liver_2)
        ## 模态间融合
        # X_tv_liver_2,X_t2_liver_2=self.liver_fusion_v_t2_2(X_tv_liver_2,X_t2_liver_2,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_2,X_t2_tumor_2=self.tumor_fusion_v_t2_2(X_tv_tumor_2,X_t2_tumor_2,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_3 =X_t2_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_2.view(b*f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_2.view(b*f,c,h,w)
        X_tv_liver_3 =X_tv_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =self.tumor_conv3_t2(X_t2_tumor_3)
        X_tv_tumor_3 =self.tumor_conv3_v(X_tv_tumor_3)
        X_tv_liver_3 =self.liver_conv3_v(X_tv_liver_3)
        X_t2_liver_3 =self.liver_conv3_t2(X_t2_liver_3)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_3.shape
        X_t2_liver_3 =X_t2_liver_3.view(b,f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_3.view(b,f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_3.view(b,f,c,h,w)
        X_tv_liver_3 =X_tv_liver_3.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_3 = self.tumor_hgnn_fusion_v_3(X_tv_tumor_3)
        X_t2_tumor_3 = self.tumor_hgnn_fusion_t2_3(X_t2_tumor_3)
        X_tv_liver_3 = self.liver_hgnn_fusion_v_3(X_tv_liver_3)
        X_t2_liver_3 = self.liver_hgnn_fusion_t2_3(X_t2_liver_3)
        ## 肝脏肿瘤融合
        X_tv_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_v_3(X_tv_tumor_3,X_tv_liver_3)
        X_t2_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_t2_3(X_t2_tumor_3,X_t2_liver_3)
        ## 模态间融合
        # X_tv_liver_3,X_t2_liver_3=self.liver_fusion_v_t2_3(X_tv_liver_3,X_t2_liver_3,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_3,X_t2_tumor_3=self.tumor_fusion_v_t2_3(X_tv_tumor_3,X_t2_tumor_3,tumor_tv_mask,tumor_t2_mask)
       
        
        X_t2_liver_4 =X_t2_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_3.view(b*f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_3.view(b*f,c,h,w)
        X_tv_liver_4 =X_tv_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =self.tumor_conv4_t2(X_t2_tumor_4)
        X_tv_tumor_4 =self.tumor_conv4_v(X_tv_tumor_4)
        X_tv_liver_4 =self.liver_conv4_v(X_tv_liver_4)
        X_t2_liver_4 =self.liver_conv4_t2(X_t2_liver_4)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_4.shape
        X_t2_liver_4 =X_t2_liver_4.view(b,f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_4.view(b,f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_4.view(b,f,c,h,w)
        X_tv_liver_4 =X_tv_liver_4.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w],mode='nearest')
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w],mode='nearest')
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w],mode='nearest')
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w],mode='nearest')

        X_tv_tumor_4 = self.tumor_hgnn_fusion_v_4(X_tv_tumor_4)
        X_t2_tumor_4 = self.tumor_hgnn_fusion_t2_4(X_t2_tumor_4)
        X_tv_liver_4 = self.liver_hgnn_fusion_v_4(X_tv_liver_4)
        X_t2_liver_4 = self.liver_hgnn_fusion_t2_4(X_t2_liver_4)
        # 肝脏肿瘤融合
        X_tv_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_v_4(X_tv_tumor_4,X_tv_liver_4)
        X_t2_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_t2_4(X_t2_tumor_4,X_t2_liver_4)
        # ## 模态间融合
        # X_tv_liver_4,X_t2_liver_4=self.liver_fusion_v_t2_4(X_tv_liver_4,X_t2_liver_4,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_4,X_t2_tumor_4=self.tumor_fusion_v_t2_4(X_tv_tumor_4,X_t2_tumor_4,tumor_tv_mask,tumor_t2_mask)
       


        tumor_tv_mask_pool = self.pool(tumor_tv_mask).squeeze()
        tumor_t2_mask_pool = self.pool(tumor_t2_mask).squeeze()
        liver_tv_mask_pool = self.pool(liver_tv_mask).squeeze()
        liver_t2_mask_pool = self.pool(liver_t2_mask).squeeze()
        # # X_liver_v_gate = self.pool(X_liver_v_gate).squeeze()
        # # X_liver_t2_gate = self.pool(X_liver_t2_gate).squeeze()

        X_liver_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_tumor_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_liver_t2_pool = self.pool(X_t2_liver_4).squeeze()
        X_tumor_t2_pool = self.pool(X_t2_liver_4).squeeze()
        
        
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        # X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        X_pool = torch.cat([ X_tumor_v_pool,  X_tumor_t2_pool,X_liver_v_pool,X_liver_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool,liver_t2_mask_pool,liver_tv_mask_pool],dim=-1)

        # x = self.fc1(X_pool) 
        if f ==10:
            x = self.fc1(x)
        elif f ==5:
            x = self.fc1_5(x)
        # x = self.fc_mask(x)
        # x = F.relu(x,inplace=True)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        # execution_time = end_time - start_time
        # print(f"代码执行时间为：{execution_time}秒")  
        return x

class CNN3d_t2_tv_hgnn_0414_baseline_neibu_multi_top(nn.Module):
    """
    从0414 构建多阶段融合
    """
    def __init__(self,num_classes=1,points=1000, num_channels =128):
        super(CNN3d_t2_tv_hgnn_0414_baseline_neibu_multi_top, self).__init__()
        ## one stage     
        self.liver_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_v = Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv1_t2 =  Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')

        self.tumor_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_v =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')                                                 
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM  = nn.LSTM(
                input_size=num_channels*4,
                hidden_size=128,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
              
        self.fc1 = nn.Linear(168, 32)
        self.fc1_5 = nn.Linear(148, 32)
        
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.stem_v = Stem(out_dim=640) ## 特征提取器
        self.stem_t2 = Stem(out_dim=640)
        self.hyper2d_tv = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_tv = FFN(640,640*4)
        self.hyper2d_t2 = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_t2 = FFN(640,640*4)
        ## 换一种方式的超图卷积 注意 这种超图卷积的输入是没有batch维度，所以每一个样本 单独初始化一个超图
        # self.layers2= HGNNConv(hid_channels=16, num_classes, is_last=True)
        ## 肿瘤内部空间融合
        self.tumor_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.tumor_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_tumor_fusion_t2_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 模态间融合
        self.tumor_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        self.liver_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        
        self.tumor_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.tumor_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_tumor_fusion_t2_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 模态间融合
        self.tumor_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)
        self.liver_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)

        self.tumor_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.tumor_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_tumor_fusion_t2_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 模态间融合
        self.tumor_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)
        self.liver_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)


        self.tumor_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.tumor_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_tumor_fusion_t2_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 模态间融合
        self.tumor_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        self.liver_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        


    def forward(self,X_image_mask):
        # torch.autograd.set_detect_anomaly(True)
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_t2_liver,X_t2_tumor,liver_t2_mask,tumor_t2_mask,X_tv_liver,X_tv_tumor,liver_tv_mask,tumor_tv_mask=X_image_mask
        b,f,c,h,w = X_tv_tumor.shape
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image= X_V_mask.to(torch.float32),X_V_image.to(torch.float32), X_T2_mask.to(torch.float32),X_T2_image.to(torch.float32)
        X_tumor_v_1 = []
        # X_liver_t2_1 = []
        X_tumor_t2_1 = []
        tumor_tv_mask = tumor_tv_mask.squeeze()
        tumor_t2_mask = tumor_t2_mask.squeeze()
        liver_t2_mask =liver_t2_mask.squeeze()
        liver_tv_mask =liver_tv_mask.squeeze()
        ## 为了gradcam 更好，先将b f合并
        '''
        for i in range(f):
            x_t2_liver_2d = X_t2_liver[:, i,:]
            x_t2_tumor_2d = X_t2_tumor[:, i,:]
            x_tv_liver_2d = X_tv_liver[:, i,:]
            x_tv_tumor_2d = X_tv_tumor[:, i,:]
            
            x_tumor_v =self.tumor_conv1_v(x_tv_tumor_2d)
            x_tumor_t2 =self.tumor_conv1_t2(x_t2_tumor_2d)
            x_liver_v =self.liver_conv1_v(x_tv_tumor_2d)
            x_liver_t2 =self.liver_conv1_t2(x_t2_tumor_2d)

            X_tumor_v_1.append(x_tumor_v)

            # X_liver_v_1.append(x_liver_v)
            X_tumor_t2_1.append(x_tumor_t2)
            # X_liver_t2_1.append(x_liver_t2)
        
        X_tumor_v_1 = torch.stack(X_tumor_v_1, dim=1) 
        # X_liver_t2_1 = torch.stack(X_liver_t2_1, dim=1)
        X_tumor_t2_1 = torch.stack(X_tumor_t2_1, dim=1) 
        '''
        X_t2_liver =X_t2_liver.view(b*f,c,h,w)
        X_t2_tumor =X_t2_tumor.view(b*f,c,h,w)
        X_tv_tumor =X_tv_tumor.view(b*f,c,h,w)
        X_tv_liver =X_tv_liver.view(b*f,c,h,w)
        X_t2_tumor =self.tumor_conv1_t2(X_t2_tumor)
        X_tv_tumor =self.tumor_conv1_v(X_tv_tumor)
        X_tv_liver =self.liver_conv1_v(X_tv_liver)
        X_t2_liver =self.liver_conv1_t2(X_t2_liver)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor.shape
        X_t2_liver_1 =X_t2_liver.view(b,f,c,h,w)
        X_t2_tumor_1 =X_t2_tumor.view(b,f,c,h,w)
        X_tv_tumor_1 =X_tv_tumor.view(b,f,c,h,w)
        X_tv_liver_1=X_tv_liver.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])
        ## 超图的构建中，怎么去采样是
        # new_X_tumor_v_1 = feature_selection(X_tumor_v_1, k=1000)
        # ## 内部融合
        X_tv_tumor_1 = self.tumor_hgnn_fusion_v_1(X_tv_tumor_1)
        X_t2_tumor_1 = self.tumor_hgnn_fusion_t2_1(X_t2_tumor_1)
        X_tv_liver_1 = self.liver_hgnn_fusion_v_1(X_tv_liver_1)
        X_t2_liver_1 = self.liver_hgnn_fusion_t2_1(X_t2_liver_1)
        ## 肝脏肿瘤融合
        start_time = time.time()
        # 执行代码        
        # X_tv_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_v_1(X_tv_tumor_1,X_tv_liver_1)
        # X_t2_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_t2_1(X_t2_tumor_1,X_t2_liver_1)
        # end_time = time.time()
        # ## 模态间融合
        X_tv_liver_1,X_t2_liver_1=self.liver_fusion_v_t2_1(X_tv_liver_1,X_t2_liver_1,liver_tv_mask,liver_t2_mask)
        X_tv_tumor_1,X_t2_tumor_1=self.tumor_fusion_v_t2_1(X_tv_tumor_1,X_t2_tumor_1,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_2 =X_t2_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_1.view(b*f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_1.view(b*f,c,h,w)
        X_tv_liver_2 =X_tv_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =self.tumor_conv2_t2(X_t2_tumor_2)
        X_tv_tumor_2 =self.tumor_conv2_v(X_tv_tumor_2)
        X_tv_liver_2 =self.liver_conv2_v(X_tv_liver_2)
        X_t2_liver_2 =self.liver_conv2_t2(X_t2_liver_2)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_2.shape
        X_t2_liver_2 =X_t2_liver_2.view(b,f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_2.view(b,f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_2.view(b,f,c,h,w)
        X_tv_liver_2 =X_tv_liver_2.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_2 = self.tumor_hgnn_fusion_v_2(X_tv_tumor_2)
        X_t2_tumor_2 = self.tumor_hgnn_fusion_t2_2(X_t2_tumor_2)
        X_tv_liver_2 = self.liver_hgnn_fusion_v_2(X_tv_liver_2)
        X_t2_liver_2 = self.liver_hgnn_fusion_t2_2(X_t2_liver_2)
        ## 肝脏肿瘤融合
        # X_tv_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_v_2(X_tv_tumor_2,X_tv_liver_2)
        # X_t2_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_t2_2(X_t2_tumor_2,X_t2_liver_2)
        ## 模态间融合
        X_tv_liver_2,X_t2_liver_2=self.liver_fusion_v_t2_2(X_tv_liver_2,X_t2_liver_2,liver_tv_mask,liver_t2_mask)
        X_tv_tumor_2,X_t2_tumor_2=self.tumor_fusion_v_t2_2(X_tv_tumor_2,X_t2_tumor_2,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_3 =X_t2_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_2.view(b*f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_2.view(b*f,c,h,w)
        X_tv_liver_3 =X_tv_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =self.tumor_conv3_t2(X_t2_tumor_3)
        X_tv_tumor_3 =self.tumor_conv3_v(X_tv_tumor_3)
        X_tv_liver_3 =self.liver_conv3_v(X_tv_liver_3)
        X_t2_liver_3 =self.liver_conv3_t2(X_t2_liver_3)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_3.shape
        X_t2_liver_3 =X_t2_liver_3.view(b,f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_3.view(b,f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_3.view(b,f,c,h,w)
        X_tv_liver_3 =X_tv_liver_3.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_3 = self.tumor_hgnn_fusion_v_3(X_tv_tumor_3)
        X_t2_tumor_3 = self.tumor_hgnn_fusion_t2_3(X_t2_tumor_3)
        X_tv_liver_3 = self.liver_hgnn_fusion_v_3(X_tv_liver_3)
        X_t2_liver_3 = self.liver_hgnn_fusion_t2_3(X_t2_liver_3)
        ## 肝脏肿瘤融合
        # X_tv_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_v_3(X_tv_tumor_3,X_tv_liver_3)
        # X_t2_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_t2_3(X_t2_tumor_3,X_t2_liver_3)
        ## 模态间融合
        # X_tv_liver_3,X_t2_liver_3=self.liver_fusion_v_t2_3(X_tv_liver_3,X_t2_liver_3,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_3,X_t2_tumor_3=self.tumor_fusion_v_t2_3(X_tv_tumor_3,X_t2_tumor_3,tumor_tv_mask,tumor_t2_mask)
       
        
        X_t2_liver_4 =X_t2_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_3.view(b*f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_3.view(b*f,c,h,w)
        X_tv_liver_4 =X_tv_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =self.tumor_conv4_t2(X_t2_tumor_4)
        X_tv_tumor_4 =self.tumor_conv4_v(X_tv_tumor_4)
        X_tv_liver_4 =self.liver_conv4_v(X_tv_liver_4)
        X_t2_liver_4 =self.liver_conv4_t2(X_t2_liver_4)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_4.shape
        X_t2_liver_4 =X_t2_liver_4.view(b,f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_4.view(b,f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_4.view(b,f,c,h,w)
        X_tv_liver_4 =X_tv_liver_4.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w],mode='nearest')
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w],mode='nearest')
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w],mode='nearest')
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w],mode='nearest')

        X_tv_tumor_4 = self.tumor_hgnn_fusion_v_4(X_tv_tumor_4)
        X_t2_tumor_4 = self.tumor_hgnn_fusion_t2_4(X_t2_tumor_4)
        X_tv_liver_4 = self.liver_hgnn_fusion_v_4(X_tv_liver_4)
        X_t2_liver_4 = self.liver_hgnn_fusion_t2_4(X_t2_liver_4)
        # 肝脏肿瘤融合
        # X_tv_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_v_4(X_tv_tumor_4,X_tv_liver_4)
        # X_t2_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_t2_4(X_t2_tumor_4,X_t2_liver_4)
        # ## 模态间融合
        # X_tv_liver_4,X_t2_liver_4=self.liver_fusion_v_t2_4(X_tv_liver_4,X_t2_liver_4,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_4,X_t2_tumor_4=self.tumor_fusion_v_t2_4(X_tv_tumor_4,X_t2_tumor_4,tumor_tv_mask,tumor_t2_mask)
       


        tumor_tv_mask_pool = self.pool(tumor_tv_mask).squeeze()
        tumor_t2_mask_pool = self.pool(tumor_t2_mask).squeeze()
        liver_tv_mask_pool = self.pool(liver_tv_mask).squeeze()
        liver_t2_mask_pool = self.pool(liver_t2_mask).squeeze()
        # # X_liver_v_gate = self.pool(X_liver_v_gate).squeeze()
        # # X_liver_t2_gate = self.pool(X_liver_t2_gate).squeeze()

        X_liver_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_tumor_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_liver_t2_pool = self.pool(X_t2_liver_4).squeeze()
        X_tumor_t2_pool = self.pool(X_t2_liver_4).squeeze()
        
        
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        # X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        X_pool = torch.cat([ X_tumor_v_pool,  X_tumor_t2_pool,X_liver_v_pool,X_liver_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool,liver_t2_mask_pool,liver_tv_mask_pool],dim=-1)

        # x = self.fc1(X_pool) 
        if f ==10:
            x = self.fc1(x)
        elif f ==5:
            x = self.fc1_5(x)
        # x = self.fc_mask(x)
        # x = F.relu(x,inplace=True)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        # execution_time = end_time - start_time
        # print(f"代码执行时间为：{execution_time}秒")  
        return x

class CNN3d_t2_tv_hgnn_0414_baseline_liver_tumor_multi_top(nn.Module):
    """
    从0414 构建多阶段融合
    """
    def __init__(self,num_classes=1,points=1000, num_channels =128):
        super(CNN3d_t2_tv_hgnn_0414_baseline_liver_tumor_multi_top, self).__init__()
        ## one stage     
        self.liver_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_v = Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv1_t2 =  Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.liver_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')

        self.tumor_conv1_v = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_v = Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_v =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_v =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv1_t2 = Conv2dBlock(1, num_channels//8, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv2_t2 =  Conv2dBlock(num_channels//8, num_channels//4, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv3_t2 =  Conv2dBlock(num_channels//4, num_channels//2, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.tumor_conv4_t2 =  Conv2dBlock(num_channels//2, num_channels, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')                                                 
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.LSTM  = nn.LSTM(
                input_size=num_channels*4,
                hidden_size=128,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
              
        self.fc1 = nn.Linear(168, 32)
        self.fc1_5 = nn.Linear(148, 32)
        
        self.fc_mask = nn.Linear(36, 16)
        
        self.fc2 = nn.Linear(32, num_classes)
        self.drop_out = nn.Dropout(0.5)
        self.stem_v = Stem(out_dim=640) ## 特征提取器
        self.stem_t2 = Stem(out_dim=640)
        self.hyper2d_tv = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_tv = FFN(640,640*4)
        self.hyper2d_t2 = HypergraphConv2d(in_channels=640, out_channels=640)
        self.ffn_t2 = FFN(640,640*4)
        ## 换一种方式的超图卷积 注意 这种超图卷积的输入是没有batch维度，所以每一个样本 单独初始化一个超图
        # self.layers2= HGNNConv(hid_channels=16, num_classes, is_last=True)
        ## 肿瘤内部空间融合
        self.tumor_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.tumor_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_v_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_hgnn_fusion_t2_1 = hgnn_fusion1(points=points,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        self.liver_tumor_fusion_t2_1 = hgnn_fusion_liver_tumor(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=points)
        ## 模态间融合
        self.tumor_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        self.liver_fusion_v_t2_1 = hgnn_fusion_top_space_liangjie(points=points*2,in_channels=num_channels//8,out_channels=num_channels//8,top_k=1)
        
        self.tumor_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.tumor_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_v_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_hgnn_fusion_t2_2 = hgnn_fusion1(points=points//2,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        self.liver_tumor_fusion_t2_2 = hgnn_fusion_liver_tumor(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=points//2)
        ## 模态间融合
        self.tumor_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)
        self.liver_fusion_v_t2_2 = hgnn_fusion_top_space_liangjie(points=points,in_channels=num_channels//4,out_channels=num_channels//4,top_k=1)

        self.tumor_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.tumor_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_v_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_hgnn_fusion_t2_3 = hgnn_fusion1(points=points//4,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        self.liver_tumor_fusion_t2_3 = hgnn_fusion_liver_tumor(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=points//4)
        ## 模态间融合
        self.tumor_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)
        self.liver_fusion_v_t2_3 = hgnn_fusion_top_space_liangjie(points=points//2,in_channels=num_channels//2,out_channels=num_channels//2,top_k=1)


        self.tumor_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.tumor_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_v_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_hgnn_fusion_t2_4 = hgnn_fusion1(points=points//8,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 肿瘤肝脏融合
        self.liver_tumor_fusion_v_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        self.liver_tumor_fusion_t2_4 = hgnn_fusion_liver_tumor(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=points//8)
        ## 模态间融合
        self.tumor_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        self.liver_fusion_v_t2_4 = hgnn_fusion_top_space_liangjie(points=points//4,in_channels=num_channels,out_channels=num_channels,top_k=1)
        


    def forward(self,X_image_mask):
        # torch.autograd.set_detect_anomaly(True)
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image = X_image_mask
        X_t2_liver,X_t2_tumor,liver_t2_mask,tumor_t2_mask,X_tv_liver,X_tv_tumor,liver_tv_mask,tumor_tv_mask=X_image_mask
        b,f,c,h,w = X_tv_tumor.shape
        # X_V_mask,X_V_image, X_T2_mask,X_T2_image= X_V_mask.to(torch.float32),X_V_image.to(torch.float32), X_T2_mask.to(torch.float32),X_T2_image.to(torch.float32)
        X_tumor_v_1 = []
        # X_liver_t2_1 = []
        X_tumor_t2_1 = []
        tumor_tv_mask = tumor_tv_mask.squeeze()
        tumor_t2_mask = tumor_t2_mask.squeeze()
        liver_t2_mask =liver_t2_mask.squeeze()
        liver_tv_mask =liver_tv_mask.squeeze()
        ## 为了gradcam 更好，先将b f合并
        '''
        for i in range(f):
            x_t2_liver_2d = X_t2_liver[:, i,:]
            x_t2_tumor_2d = X_t2_tumor[:, i,:]
            x_tv_liver_2d = X_tv_liver[:, i,:]
            x_tv_tumor_2d = X_tv_tumor[:, i,:]
            
            x_tumor_v =self.tumor_conv1_v(x_tv_tumor_2d)
            x_tumor_t2 =self.tumor_conv1_t2(x_t2_tumor_2d)
            x_liver_v =self.liver_conv1_v(x_tv_tumor_2d)
            x_liver_t2 =self.liver_conv1_t2(x_t2_tumor_2d)

            X_tumor_v_1.append(x_tumor_v)

            # X_liver_v_1.append(x_liver_v)
            X_tumor_t2_1.append(x_tumor_t2)
            # X_liver_t2_1.append(x_liver_t2)
        
        X_tumor_v_1 = torch.stack(X_tumor_v_1, dim=1) 
        # X_liver_t2_1 = torch.stack(X_liver_t2_1, dim=1)
        X_tumor_t2_1 = torch.stack(X_tumor_t2_1, dim=1) 
        '''
        X_t2_liver =X_t2_liver.view(b*f,c,h,w)
        X_t2_tumor =X_t2_tumor.view(b*f,c,h,w)
        X_tv_tumor =X_tv_tumor.view(b*f,c,h,w)
        X_tv_liver =X_tv_liver.view(b*f,c,h,w)
        X_t2_tumor =self.tumor_conv1_t2(X_t2_tumor)
        X_tv_tumor =self.tumor_conv1_v(X_tv_tumor)
        X_tv_liver =self.liver_conv1_v(X_tv_liver)
        X_t2_liver =self.liver_conv1_t2(X_t2_liver)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor.shape
        X_t2_liver_1 =X_t2_liver.view(b,f,c,h,w)
        X_t2_tumor_1 =X_t2_tumor.view(b,f,c,h,w)
        X_tv_tumor_1 =X_tv_tumor.view(b,f,c,h,w)
        X_tv_liver_1=X_tv_liver.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])
        ## 超图的构建中，怎么去采样是
        # new_X_tumor_v_1 = feature_selection(X_tumor_v_1, k=1000)
        # ## 内部融合
        X_tv_tumor_1 = self.tumor_hgnn_fusion_v_1(X_tv_tumor_1)
        X_t2_tumor_1 = self.tumor_hgnn_fusion_t2_1(X_t2_tumor_1)
        X_tv_liver_1 = self.liver_hgnn_fusion_v_1(X_tv_liver_1)
        X_t2_liver_1 = self.liver_hgnn_fusion_t2_1(X_t2_liver_1)
        ## 肝脏肿瘤融合
        start_time = time.time()
        # 执行代码        
        # X_tv_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_v_1(X_tv_tumor_1,X_tv_liver_1)
        # X_t2_tumor_1,X_t2_liver_1 =self.liver_tumor_fusion_t2_1(X_t2_tumor_1,X_t2_liver_1)
        # end_time = time.time()
        # ## 模态间融合
        X_tv_liver_1,X_t2_liver_1=self.liver_fusion_v_t2_1(X_tv_liver_1,X_t2_liver_1,liver_tv_mask,liver_t2_mask)
        X_tv_tumor_1,X_t2_tumor_1=self.tumor_fusion_v_t2_1(X_tv_tumor_1,X_t2_tumor_1,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_2 =X_t2_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_1.view(b*f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_1.view(b*f,c,h,w)
        X_tv_liver_2 =X_tv_liver_1.view(b*f,c,h,w)
        X_t2_tumor_2 =self.tumor_conv2_t2(X_t2_tumor_2)
        X_tv_tumor_2 =self.tumor_conv2_v(X_tv_tumor_2)
        X_tv_liver_2 =self.liver_conv2_v(X_tv_liver_2)
        X_t2_liver_2 =self.liver_conv2_t2(X_t2_liver_2)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_2.shape
        X_t2_liver_2 =X_t2_liver_2.view(b,f,c,h,w)
        X_t2_tumor_2 =X_t2_tumor_2.view(b,f,c,h,w)
        X_tv_tumor_2 =X_tv_tumor_2.view(b,f,c,h,w)
        X_tv_liver_2 =X_tv_liver_2.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_2 = self.tumor_hgnn_fusion_v_2(X_tv_tumor_2)
        X_t2_tumor_2 = self.tumor_hgnn_fusion_t2_2(X_t2_tumor_2)
        X_tv_liver_2 = self.liver_hgnn_fusion_v_2(X_tv_liver_2)
        X_t2_liver_2 = self.liver_hgnn_fusion_t2_2(X_t2_liver_2)
        ## 肝脏肿瘤融合
        # X_tv_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_v_2(X_tv_tumor_2,X_tv_liver_2)
        # X_t2_tumor_2,X_t2_liver_2 =self.liver_tumor_fusion_t2_2(X_t2_tumor_2,X_t2_liver_2)
        ## 模态间融合
        X_tv_liver_2,X_t2_liver_2=self.liver_fusion_v_t2_2(X_tv_liver_2,X_t2_liver_2,liver_tv_mask,liver_t2_mask)
        X_tv_tumor_2,X_t2_tumor_2=self.tumor_fusion_v_t2_2(X_tv_tumor_2,X_t2_tumor_2,tumor_tv_mask,tumor_t2_mask)
        
        X_t2_liver_3 =X_t2_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_2.view(b*f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_2.view(b*f,c,h,w)
        X_tv_liver_3 =X_tv_liver_2.view(b*f,c,h,w)
        X_t2_tumor_3 =self.tumor_conv3_t2(X_t2_tumor_3)
        X_tv_tumor_3 =self.tumor_conv3_v(X_tv_tumor_3)
        X_tv_liver_3 =self.liver_conv3_v(X_tv_liver_3)
        X_t2_liver_3 =self.liver_conv3_t2(X_t2_liver_3)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_3.shape
        X_t2_liver_3 =X_t2_liver_3.view(b,f,c,h,w)
        X_t2_tumor_3 =X_t2_tumor_3.view(b,f,c,h,w)
        X_tv_tumor_3 =X_tv_tumor_3.view(b,f,c,h,w)
        X_tv_liver_3 =X_tv_liver_3.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w])
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w])
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w])

        X_tv_tumor_3 = self.tumor_hgnn_fusion_v_3(X_tv_tumor_3)
        X_t2_tumor_3 = self.tumor_hgnn_fusion_t2_3(X_t2_tumor_3)
        X_tv_liver_3 = self.liver_hgnn_fusion_v_3(X_tv_liver_3)
        X_t2_liver_3 = self.liver_hgnn_fusion_t2_3(X_t2_liver_3)
        ## 肝脏肿瘤融合
        # X_tv_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_v_3(X_tv_tumor_3,X_tv_liver_3)
        # X_t2_tumor_3,X_t2_liver_3 =self.liver_tumor_fusion_t2_3(X_t2_tumor_3,X_t2_liver_3)
        ## 模态间融合
        # X_tv_liver_3,X_t2_liver_3=self.liver_fusion_v_t2_3(X_tv_liver_3,X_t2_liver_3,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_3,X_t2_tumor_3=self.tumor_fusion_v_t2_3(X_tv_tumor_3,X_t2_tumor_3,tumor_tv_mask,tumor_t2_mask)
       
        
        X_t2_liver_4 =X_t2_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_3.view(b*f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_3.view(b*f,c,h,w)
        X_tv_liver_4 =X_tv_liver_3.view(b*f,c,h,w)
        X_t2_tumor_4 =self.tumor_conv4_t2(X_t2_tumor_4)
        X_tv_tumor_4 =self.tumor_conv4_v(X_tv_tumor_4)
        X_tv_liver_4 =self.liver_conv4_v(X_tv_liver_4)
        X_t2_liver_4 =self.liver_conv4_t2(X_t2_liver_4)
        ## 第二阶段更新b,f,c,h,w
        b_f,c,h,w = X_tv_tumor_4.shape
        X_t2_liver_4 =X_t2_liver_4.view(b,f,c,h,w)
        X_t2_tumor_4 =X_t2_tumor_4.view(b,f,c,h,w)
        X_tv_tumor_4 =X_tv_tumor_4.view(b,f,c,h,w)
        X_tv_liver_4 =X_tv_liver_4.view(b,f,c,h,w)
        tumor_t2_mask = interpolate(tumor_t2_mask, size=[h, w],mode='nearest')
        liver_t2_mask = interpolate(liver_t2_mask, size=[h, w],mode='nearest')
        tumor_tv_mask = interpolate(tumor_tv_mask, size=[h, w],mode='nearest')
        liver_tv_mask = interpolate(liver_tv_mask, size=[h, w],mode='nearest')

        X_tv_tumor_4 = self.tumor_hgnn_fusion_v_4(X_tv_tumor_4)
        X_t2_tumor_4 = self.tumor_hgnn_fusion_t2_4(X_t2_tumor_4)
        X_tv_liver_4 = self.liver_hgnn_fusion_v_4(X_tv_liver_4)
        X_t2_liver_4 = self.liver_hgnn_fusion_t2_4(X_t2_liver_4)
        # 肝脏肿瘤融合
        # X_tv_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_v_4(X_tv_tumor_4,X_tv_liver_4)
        # X_t2_tumor_4,X_t2_liver_4 =self.liver_tumor_fusion_t2_4(X_t2_tumor_4,X_t2_liver_4)
        # ## 模态间融合
        # X_tv_liver_4,X_t2_liver_4=self.liver_fusion_v_t2_4(X_tv_liver_4,X_t2_liver_4,liver_tv_mask,liver_t2_mask)
        # X_tv_tumor_4,X_t2_tumor_4=self.tumor_fusion_v_t2_4(X_tv_tumor_4,X_t2_tumor_4,tumor_tv_mask,tumor_t2_mask)
       


        tumor_tv_mask_pool = self.pool(tumor_tv_mask).squeeze()
        tumor_t2_mask_pool = self.pool(tumor_t2_mask).squeeze()
        liver_tv_mask_pool = self.pool(liver_tv_mask).squeeze()
        liver_t2_mask_pool = self.pool(liver_t2_mask).squeeze()
        # # X_liver_v_gate = self.pool(X_liver_v_gate).squeeze()
        # # X_liver_t2_gate = self.pool(X_liver_t2_gate).squeeze()

        X_liver_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_tumor_v_pool = self.pool(X_tv_tumor_4).squeeze()
        X_liver_t2_pool = self.pool(X_t2_liver_4).squeeze()
        X_tumor_t2_pool = self.pool(X_t2_liver_4).squeeze()
        
        
        # X_pool = torch.cat([X_liver_pool,X_tumor_pool],dim=-1)
        # RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        # x = RNN_out[:, -1,]
        # x = torch.cat([x,liver_mask_pool,tumor_mask_pool],dim=-1)

        # liver_mask_pool = self.pool(liver_mask).squeeze()
        # tumor_mask_pool = self.pool(tumor_mask).squeeze()

        # X_pool = torch.cat([X_liver_v_pool, X_tumor_v_pool, X_liver_t2_pool, X_tumor_t2_pool], dim=-1)
        X_pool = torch.cat([ X_tumor_v_pool,  X_tumor_t2_pool,X_liver_v_pool,X_liver_t2_pool], dim=-1)
        
        # X_pool = torch.cat([X_tumor_v_gate, X_tumor_t2_gate,tumor_t2_mask_pool,tumor_tv_mask_pool], dim=-1)
        RNN_out, (h_n, h_c) = self.LSTM(X_pool, None)
        x = RNN_out[:, -1,]
        x = torch.cat([x,tumor_tv_mask_pool,tumor_t2_mask_pool,liver_t2_mask_pool,liver_tv_mask_pool],dim=-1)

        # x = self.fc1(X_pool) 
        if f ==10:
            x = self.fc1(x)
        elif f ==5:
            x = self.fc1_5(x)
        # x = self.fc_mask(x)
        # x = F.relu(x,inplace=True)
        x=self.drop_out(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        # execution_time = end_time - start_time
        # print(f"代码执行时间为：{execution_time}秒")  
        return x


def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    # Replace -1s in idx with the index pointing to the dummy node/hyperedge
    idx[idx == -1] = x.size(2) - 1  # Make sure this points to the last node/hyperedge which is the dummy one

    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature

class HypergraphConv2d(nn.Module):
    """
    Hypergraph Convolution based on the GIN mechanism
    """
    def __init__(self, in_channels, out_channels, act='gelu', norm=None, bias=True):
        super(HypergraphConv2d, self).__init__()
        # Node to hyperedge transformation
        self.nn_node_to_hyperedge = BasicConv_2([in_channels, in_channels], act, norm, bias) # in_channels = 128, out_channels = 256
        # Hyperedge to node transformation
        self.nn_hyperedge_to_node = BasicConv_2([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, hyperedge_matrix, point_hyperedge_index, centers):
        with torch.no_grad():
            # Check and append dummy node to x if not present
            if not torch.equal(x[:, :, -1, :], torch.zeros((x.size(0), x.size(1), 1, x.size(3)), device=x.device)): # 假设 n_dims = 128, n_points = 3136, hyperedge_num = 50
                dummy_node = torch.zeros((x.size(0), x.size(1), 1, x.size(3)), device=x.device)
                x = torch.cat([x, dummy_node], dim=2) # (1, 128, 3137, 1)
            
            # Check and append dummy hyperedge to centers if not present
            if not torch.equal(centers[:, :, -1], torch.zeros((centers.size(0), centers.size(1), 1), device=centers.device)):
                dummy_hyperedge = torch.zeros((centers.size(0), centers.size(1), 1), device=centers.device)
                centers = torch.cat([centers, dummy_hyperedge], dim=2) # centers: (1, 128, 51)

        # Step 1: Aggregate node features to get hyperedge features
        node_features_for_hyperedges = batched_index_select(x, hyperedge_matrix)
        aggregated_hyperedge_features = node_features_for_hyperedges.sum(dim=-1, keepdim=True)
        aggregated_hyperedge_features = self.nn_node_to_hyperedge(aggregated_hyperedge_features).squeeze(-1) # (1, 128, 50)
        # Adding the hyperedge center features to the aggregated hyperedge features
        aggregated_hyperedge_features += (1 + self.eps) * centers[:, :, :-1]
        
        # Step 2: Aggregate hyperedge features to update node features
        hyperedge_features_for_nodes = batched_index_select(aggregated_hyperedge_features.unsqueeze(-1), point_hyperedge_index)
        aggregated_node_features_from_hyperedges = self.nn_hyperedge_to_node(hyperedge_features_for_nodes.sum(dim=-1, keepdim=True)).squeeze(-1)

        # Update original node features
        out = aggregated_node_features_from_hyperedges

        return out

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='gelu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Dropout(drop_path)
    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class HGNN(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,   
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 =  Conv2dBlock(1, 128, 6, 1, 0,
                             norm='sn',
                             activation='lrelu',
                             pad_type='zero')
        self.layers = nn.ModuleList()
        self.in_channels = in_channels
        # self.layers1= HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        # self.layers1 = HypergraphConv2d(in_channels=in_channels, out_channels=hid_channels)
        self.layers1 = EdgeConv2d(in_channels=in_channels, out_channels=hid_channels)
        
        # self.layers2= HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        # self.layers2 = HypergraphConv2d(in_channels=hid_channels, out_channels=hid_channels)
        self.layers2 = EdgeConv2d(in_channels=hid_channels, out_channels=hid_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.linear = nn.Linear(96,num_classes)
        self.LSTM  = nn.LSTM(
                input_size=128,
                hidden_size=96,        
                num_layers=1,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        self.graph_constructor = DenseDilatedKnnGraph()
    def forward(self, X):
        """The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            
        """
        b,n,v,h,w=X[1].shape
        X = [x_image.view(b*n*v,1,h,w).to(torch.float32) for x_image in  X]
        ## 设置一个卷积特征提取器
        X = [self.pool(self.conv1(x_image)).squeeze().view(b,-1,self.in_channels) for x_image in  X] # 8x100x50
        X = [F.relu(x_image) for x_image in X]
        ## 第一种方案 将所有的节点叠加

        ## gcn 聚合 ## batch_size, num_dims, num_points, 1
        X = torch.cat(X, dim=1)
        b,num_points,num_dims=X.shape
        selected_features = X.unsqueeze(-1).view(b, num_dims, num_points, 1)
        # hyperedge_matrix, point_hyperedge_index, centers = construct_hyperedges(selected_features,num_clusters=28)
        edge_index = self.graph_constructor(selected_features)
        selected_features_new = self.layers1(selected_features,edge_index)

        edge_index = self.graph_constructor(selected_features_new)
        selected_features_new = self.layers1(selected_features_new,edge_index)
        
        # hyperedge_matrix, point_hyperedge_index, centers = construct_hyperedges(selected_features_new.unsqueeze(-1),num_clusters=28)
        # selected_features_new = self.layers2(selected_features,hyperedge_matrix, point_hyperedge_index, centers)
        selected_features_new = selected_features_new.squeeze().permute(0,2,1)
       
        # A = Hypergraph(1000)
        # X = torch.cat(X, dim=1)
        # X_tumor_1 = []
        # for i in range(b):
        #     x_2d = X[i, :,:]
        #     A = A.from_feature_kNN(x_2d.detach(),k=10)
        #     x_2d = self.layers1(x_2d,A)
        #     X_tumor_1.append(x_2d)
        # X_tumor_1 = torch.stack(X_tumor_1,dim=0)

        # X_tumor_2 = []
        # for i in range(b):
        #     x_2d_2 = X[i, :,:]
        #     A_2 = A.from_feature_kNN(x_2d_2.detach(),k=10)
        #     x_2d_2 = self.layers2(x_2d_2,A_2)
        #     X_tumor_2.append(x_2d_2)
        # X_tumor_2 = torch.stack(X_tumor_2,dim=0)
        RNN_out, (h_n, h_c) = self.LSTM(selected_features_new, None)
        x = RNN_out[:, -1,]

        X = self.linear(x)
        X = self.relu1(X)
        X = F.sigmoid(X) 
        return X


class GIN_hyper(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate=0.1,num_patch=300,num_hyper=100,num_pt=8):
        super(GIN_hyper, self).__init__()
        self.num_hyper = num_hyper
        self.H = torch.nn.Parameter(torch.ones((num_patch, self.num_hyper), requires_grad=True),
                                    requires_grad=True)        
        self.T = torch.nn.Parameter((torch.ones((num_pt, num_patch), requires_grad=True)),
                                    requires_grad=True)
        self.W = torch.nn.Parameter(
            (torch.ones((self.num_hyper), requires_grad=True)),
            requires_grad=True)
        
        # self.Wdiag = torch.diag(self.W)
        self.linear1 = nn.Linear(in_ch, in_ch)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(in_ch, out_ch)
        self.drop_rate= drop_rate
        # 后期看要不要初始化
        # self.H = nn.init.normal_(self.H)    
        # self.T = nn.init.normal_(self.T)
        # self.W = nn.init.normal_(self.W)

    def forward(self, node_feat):
        nd = node_feat
        # message = torch.matmul(F.softmax(self.adj, 0), nd) + nd  # softmax(A)
        # Dv = torch.diag(torch.pow((self.H * self.W).sum(-1)+1e-10, -0.5))
        # De = torch.diag(torch.pow(self.H.sum(0)+1e-10, -1))
        # print('DVDE:', De, Dv)
        # M1 = self.T @ Dv @ self.H @ torch.diag(self.W[0]) @ De @ torch.t(self.H)  @ Dv
        #  M1 = self.T @ Dv @ self.H @ torch.diag(self.W[0]) @ De @ torch.t(self.H) # @ Dv
        D_v = self.H @ torch.diag(self.W) @ self.H.t()
        D_e = torch.diag(torch.sum(self.H, dim=0))
        D_sqrt_inv = torch.diag(torch.pow(D_v.diagonal(), -0.5))
        M1 = D_sqrt_inv @ self.H @ torch.diag(self.W) @ D_e @ self.H.t() @ D_sqrt_inv 
        message = torch.matmul(M1, nd)
        #冯哥初始
      #  M1 = self.T @ self.H @ torch.diag(self.W[0]) @ torch.t(self.H)

        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(M1.cpu().numpy())
        # axs[1].imshow(self.H.cpu().numpy())

        # plt.imshow(torch.diag(self.W[0]).cpu().numpy(), cmap='RdYlBu')
        # plt.colorbar(shrink=0.45)
        # plt.show()
        # plt.imshow(torch.diag(self.W[0]).cpu().numpy(), cmap='RdYlBu')
        # plt.colorbar(shrink=0.45)
        # plt.show()

        # M1 = torch.matmul(torch.matmul(torch.matmul(self.T, self.H), self.W), torch.t(self.H))  # @
        # message = torch.matmul(M1, nd)
        x = F.dropout(message, p=self.drop_rate, training=True)
        x = self.linear1(x)
        x = self.relu1(x)
        x = F.dropout(x, p=self.drop_rate, training=True)
        x = self.linear2(x)
        # x = F.sigmoid(x) 
        return x
    


class hgnn_cnn_lstm_two_branch(nn.Module):
    def __init__(
        self,
        in_channels: int,   
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super(hgnn_cnn_lstm_two_branch, self).__init__()
        self.hgnn = HGNN(in_channels=in_channels,hid_channels=hid_channels,num_classes=1,use_bn=True)
        self.cnn_lstm = CNN3d_t2_tv_hgnn(num_classes=1)
        self.G = Hypergraph(1000) ## 初始化超图
    def forward(self, X_image_mask,X_cor_feature):
        x_0 = self.cnn_lstm(X_image_mask)
        # x_1 = self.hgnn(X_cor_feature)
        
        # x = x_0
        return x_0


class hgnn_cnn_lstm_0313(nn.Module):
    def __init__(
        self,
        in_channels: int,   
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super(hgnn_cnn_lstm_0313, self).__init__()
        self.hgnn = HGNN(in_channels=in_channels,hid_channels=hid_channels,num_classes=1,use_bn=True)
        self.cnn_lstm = CNN3d_t2_tv_hgnn(num_classes=1)
        self.G = Hypergraph(1000) ## 初始化超图
    def forward(self, X_image_mask,X_cor_feature):
        x_0 = self.cnn_lstm(X_image_mask)
        return x_0

class hgnn_cnn_lstm_0314(nn.Module):
    def __init__(
        self,
        in_channels: int,   
        hid_channels: int,
        num_classes: int,
        points: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super(hgnn_cnn_lstm_0314, self).__init__()
        self.hgnn = HGNN(in_channels=128,hid_channels=128,num_classes=1,use_bn=True)
        self.cnn_lstm = CNN3d_t2_tv_hgnn_0414(num_classes=num_classes,points=points,num_channels=in_channels)
        self.G = Hypergraph(1000) ## 初始化超图
    def forward(self, X_image_mask,X_cor_feature):
        # x_1 = self.hgnn(X_cor_feature)
        x_0 = self.cnn_lstm(X_image_mask)
        return x_0


def initialize_memberships(batch_size, n_points, n_clusters, device):
    """
    Initialize the membership matrix for Fuzzy C-Means clustering.

    Args:
        batch_size: int
        n_points: int
        n_clusters: int
        device: torch.device

    Returns:
        memberships: tensor (batch_size, n_points, n_clusters)
    """
    # Randomly initialize the membership matrix ensuring that the sum over clusters for each point is 1
    memberships = torch.rand(batch_size, n_points, n_clusters, device=device)
    memberships = memberships / memberships.sum(dim=2, keepdim=True)
    return memberships

def fuzzy_c_means(x, n_clusters, m=2, epsilon=1e-6, max_iter=1000):
    """
    Fuzzy C-Means clustering

    Args:
        x: tensor (batch_size, num_dims, num_points, 1)
        n_clusters: int, the number of clusters
        m: float, fuzziness parameter
        epsilon: float, threshold for stopping criterion
        max_iter: int, maximum number of iterations

    Returns:
        membership: tensor (batch_size, num_points, n_clusters)
        centers: tensor (batch_size, num_dims, n_clusters)
    """
    batch_size, num_dims, num_points, _ = x.size()
    x = x.squeeze(-1).transpose(1, 2)  # Shape: (batch_size, num_points, num_dims)

    # Initialize the membership matrix
    memberships = initialize_memberships(batch_size, num_points, n_clusters, x.device)

    # Initialize cluster centers
    centers = torch.zeros(batch_size, num_dims, n_clusters, device=x.device)
    prev_memberships = torch.zeros_like(memberships)

    for iteration in range(max_iter):
        # Update cluster centers
        for cluster in range(n_clusters):
            # Calculate the denominator
            weights = memberships[:, :, cluster] ** m
            denominator = weights.sum(dim=1, keepdim=True)
            # Update centers
            numerator = (weights.unsqueeze(2) * x).sum(dim=1)
            centers[:, :, cluster] = numerator / denominator

        # Update memberships
        for cluster in range(n_clusters):
            diff = x - centers[:, :, cluster].unsqueeze(1)
            dist = torch.norm(diff, p=2, dim=2)  # Euclidean distance
            memberships[:, :, cluster] = 1.0 / (dist ** (2 / (m - 1)))

        # Normalize the memberships such that each point's memberships across clusters sum to 1
        memberships_sum = memberships.sum(dim=2, keepdim=True)
        memberships = memberships / memberships_sum

        # Check convergence: stop if memberships do not change significantly
        if iteration > 0 and torch.norm(prev_memberships - memberships) < epsilon:
            break
        prev_memberships = memberships.clone()

    return memberships, centers


def construct_hyperedges(x, num_clusters, threshold=0.005, m=2):
    """
    Constructs hyperedges based on fuzzy c-means clustering.

    Args:
        x (torch.Tensor): Input point cloud data with shape (batch_size, num_dims, num_points, 1).
        num_clusters (int): Number of clusters (hyperedges).
        threshold (float): Threshold value for memberships to consider a point belonging to a cluster.
        m (float): Fuzzifier for fuzzy c-means clustering.

    Returns:
        hyperedge_matrix (torch.Tensor): Tensor of shape (batch_size, n_clusters, num_points_index).
            Represents each cluster's points. Padded with -1 for unequal cluster sizes.
        point_hyperedge_index (torch.Tensor): Tensor of shape (batch_size, num_points, cluster_index).
            Indicates the clusters each point belongs to. Padded with -1 for points belonging to different numbers of clusters.
        hyperedge_features (torch.Tensor): Tensor of shape (batch_size, num_dims, n_clusters).
            The center of each cluster, serving as the feature for each hyperedge.
    """
    
    with torch.no_grad():
        x = x.detach()  # Detach x from the computation graph
        
        batch_size, num_dims, num_points, _ = x.shape
        
        # Get memberships and centers using the fuzzy c-means clustering
        memberships, centers = fuzzy_c_means(x, num_clusters, m)
        
        # Create hyperedge matrix to represent each hyperedge's points
        # Initialized with -1s for padding
        hyperedge_matrix = -torch.ones(batch_size, num_clusters, num_points, dtype=torch.long)
        for b in range(batch_size):
            for c in range(num_clusters):
                idxs = torch.where(memberships[b, :, c] > threshold)[0]
                hyperedge_matrix[b, c, :len(idxs)] = idxs
        
        # Create point to hyperedge index to indicate which hyperedges each point belongs to
        # Initialized with -1s for padding
        max_edges_per_point = (memberships > threshold).sum(dim=-1).max().item()
        point_hyperedge_index = -torch.ones(batch_size, num_points, max_edges_per_point, dtype=torch.long)
        for b in range(batch_size):
            for p in range(num_points):
                idxs = torch.where(memberships[b, p, :] > threshold)[0]
                point_hyperedge_index[b, p, :len(idxs)] = idxs
    
    # Return the three constructed tensors
    return hyperedge_matrix, point_hyperedge_index, centers

def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)

def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k) # b, n, k
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)

class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index

class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=4, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            ####
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            ####
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)
        return self._dilated(edge_index)

class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv_2([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


if __name__=="__main__":    
    X =  nn.init.uniform_(torch.randn(8,50*4,100,1))   ## batch_size, num_dims, num_points, 1
    # hyper2d = HypergraphConv2d(in_channels=50*4, out_channels=50*4)
    # hyperedge_matrix, point_hyperedge_index, centers = construct_hyperedges(X,num_clusters=28)
    # X = hyper2d(X,hyperedge_matrix, point_hyperedge_index, centers)
    graph_constructor = DenseDilatedKnnGraph()
    edge_index = graph_constructor(X)
    gconv = EdgeConv2d(in_channels=50*4, out_channels=50*4)
    x = gconv.forward(X, edge_index)
    print(X.shape)
