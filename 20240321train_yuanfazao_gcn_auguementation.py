from torchvision import datasets, transforms
import torchtuples as tt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import argparse
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from functionsbranch import Dataset_t2_tv_tumor_liver_recurrence_aug
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from our_model import CNN3d_t2_tv_hgnn_0414_three_model
import pandas as pd



def R_set(x):
    """Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	"""
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return indicator_matrix

def negative_log_likelihood_loss(risk, E):
    """
    Return the negative average log-likelihood of the prediction
    of this model under a given target distribution.
    :parameter risk: output of the NN for inputs
    :parameter E: binary tensor providing censor information
    :returns: partial cox negative log likelihood
    """
    risk = risk.squeeze()
    hazard_ratio = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(torch.squeeze(hazard_ratio), dim=0))
    uncensored_likelihood = risk - log_risk
    censored_likelihood = uncensored_likelihood * E
    num_observed_events = torch.sum(E.data)
    neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events
    return neg_likelihood

def get_args():
    parser = argparse.ArgumentParser(description='Train the Net via shell', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fc1', type=int, default=512)
    parser.add_argument('--fc2', type=int, default=256)
    parser.add_argument('--emb', type=int, default=256)
    parser.add_argument('--rnn_layer', type=int, default=1)
    parser.add_argument('--rnn_emb', type=int, default=512)
    parser.add_argument('--rnn_fc', type=int, default=512)
    parser.add_argument('--area', type=int, default=64)
    parser.add_argument('--random', type=int, default=42)
    parser.add_argument('--test', type=float, default=0.3)
    return parser.parse_args()
args = get_args()
t2_liver_data_path = '/data/liver_resection_T2/0112_crop_liver_images_masks_jpg'
t2_tumor_data_path = '/data/liver_resection_T2/0112_crop_tumor_images_masks_jpg'
tv_liver_data_path = '/data/liver_resection_V/0112_crop_liver_images_masks_jpg'
tv_tumor_data_path = '/data/liver_resection_V/0112_crop_tumor_images_masks_jpg'
t2_liver_mask_path = '/data/liver_resection_T2/0112_crop_liver_masks_jpg'
t2_tumor_mask_path = '/data/liver_resection_T2/0112_crop_tumor_masks_jpg'
tv_liver_mask_path = '/data/liver_resection_V/0112_crop_liver_masks_jpg'
tv_tumor_mask_path = '/data/liver_resection_V/0112_crop_tumor_masks_jpg'
CNN_fc_hidden1, CNN_fc_hidden2 = (args.fc1, args.fc2)
res_size = 112
k = 2
epochs = 100
batch_size = 16
learning_rate = 0.005
seed = 3
import random
import numpy as np
random.seed(seed)
np.random.seed(seed)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
setup_seed(seed)

def sort_batch_tumor_liver(e, t, X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask):
    """
    Sorts the batch for loss computation
    """
    t, indices = torch.sort(t.squeeze(), descending=True)
    e = e[indices]
    e = e.squeeze(dim=1)
    X_liver = X_liver[indices]
    X_liver = X_liver.squeeze(dim=1)
    X_tumor = X_tumor[indices]
    X_tumor = X_tumor.squeeze(dim=1)
    X_liver_mask = X_liver_mask[indices]
    X_liver_mask = X_liver_mask.squeeze(dim=1)
    X_tumor_mask = X_tumor_mask[indices]
    X_tumor_mask = X_tumor_mask.squeeze(dim=1)
    X_tv_liver = X_tv_liver[indices]
    X_tv_liver = X_tv_liver.squeeze(dim=1)
    X_tv_tumor = X_tv_tumor[indices]
    X_tv_tumor = X_tv_tumor.squeeze(dim=1)
    X_tv_liver_mask = X_tv_liver_mask[indices]
    X_tv_liver_mask = X_tv_liver_mask.squeeze(dim=1)
    X_tv_tumor_mask = X_tv_tumor_mask[indices]
    X_tv_tumor_mask = X_tv_tumor_mask.squeeze(dim=1)
    return (e, t, X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask)

def train_recurrence(model, device, train_loader, optimizer, epoch):
    model.train()
    training_loss = []
    events_all = []
    times_all = []
    predict = []
    N_count = 0
    for batch_idx, (X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask, events, durations) in enumerate(train_loader):
        events, durations, X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask = sort_batch_tumor_liver(events, durations, X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask)
        X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask, events, durations = (X_liver.to(device), X_tumor.to(device), X_liver_mask.to(device), X_tumor_mask.to(device), X_tv_liver.to(device), X_tv_tumor.to(device), X_tv_liver_mask.to(device), X_tv_tumor_mask.to(device), events.to(device).view(-1), durations.to(device).view(-1))
        output = model((X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask))
        predict.append(output)
        events_all.append(events)
        times_all.append(durations)
        loss = negative_log_likelihood_loss(output, events)
        optimizer.zero_grad()
        training_loss.append(loss.item())
        N_count += X_liver.size(0)
        batch_c_index = round(1 - concordance_index(np.array(durations.detach().cpu()), np.array(output.detach().cpu()), np.array(events.detach().cpu())), 4)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Cindex: {:.2f}%'.format(epoch + 1, N_count, len(train_loader.dataset), 100.0 * (batch_idx + 1) / len(train_loader), loss.item(), batch_c_index))
    events_all = torch.concat(events_all).squeeze()
    times_all = torch.concat(times_all).squeeze()
    predict = torch.concat(predict).squeeze()
    train_loss = np.average(training_loss)
    c_index = round(1 - concordance_index(np.array(times_all.detach().cpu()), np.array(predict.detach().cpu()), np.array(events_all.detach().cpu())), 4)
    median_risk = torch.median(predict)
    high_low_risk_query = torch.where(predict > median_risk, torch.tensor(1).to(device), torch.tensor(0).to(device))
    data_cph = pd.DataFrame({'risk': np.array(high_low_risk_query.detach().cpu()), 'time': np.array(times_all.detach().cpu()), 'status': np.array(events_all.detach().cpu())})
    cph = CoxPHFitter()
    haztio_ratio = 0
    try:
        cph.fit(data_cph, duration_col='time', event_col='status')
        haztio_ratio = cph.hazard_ratios_['risk']
    except Exception as e:
        print(f'Error: {e}')
    print('\nTrain set ({:d} samples): Average loss: {:.4f}, C_index: {:.2f}% ,Hazard_ratio :{:.4f}%\n'.format(len(predict), train_loss, c_index, haztio_ratio))
    return (train_loss, c_index, haztio_ratio, median_risk)

def recurrence_validation(model, median_risk, device, optimizer, test_loader, fold=0, result_f=None, cindex_best=0, cindex_best_epoch=0, hr_best=0, hr_best_epoch=0):
    global checkpoint_k_path
    global km_k_path
    events_all = []
    times_all = []
    predict = []
    validation_loss = []
    model.eval()
    with torch.no_grad():
        for X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask, events, durations in test_loader:
            events, durations, X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask = sort_batch_tumor_liver(events, durations, X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask)
            X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask, events, durations = (X_liver.to(device), X_tumor.to(device), X_liver_mask.to(device), X_tumor_mask.to(device), X_tv_liver.to(device), X_tv_tumor.to(device), X_tv_liver_mask.to(device), X_tv_tumor_mask.to(device), events.to(device).view(-1), durations.to(device).view(-1))
            output = model((X_liver, X_tumor, X_liver_mask, X_tumor_mask, X_tv_liver, X_tv_tumor, X_tv_liver_mask, X_tv_tumor_mask))
            predict.append(output)
            events_all.append(events)
            times_all.append(durations)
            loss = negative_log_likelihood_loss(output, events)
            validation_loss.append(loss.item())
    valid_loss = np.average(validation_loss)
    events_all = torch.concat(events_all).squeeze()
    times_all = torch.concat(times_all).squeeze()
    predict = torch.concat(predict).squeeze()
    c_index = round(1 - concordance_index(np.array(times_all.detach().cpu()), np.array(predict.detach().cpu()), np.array(events_all.detach().cpu())), 4)
    median_risk = median_risk
    high_low_risk_query = torch.where(predict > median_risk, torch.tensor(1).to(device), torch.tensor(0).to(device))
    data_cph = pd.DataFrame({'risk': np.array(high_low_risk_query.detach().cpu()), 'time': np.array(times_all.detach().cpu()), 'status': np.array(events_all.detach().cpu())})
    haztio_ratio = 0
    cph = CoxPHFitter()
    try:
        cph.fit(data_cph, duration_col='time', event_col='status')
        haztio_ratio = cph.hazard_ratios_['risk']
    except Exception as e:
        print(f'Error: {e}')
    data_km = pd.DataFrame({'risk': np.array(predict.detach().cpu()), 'time': np.array(times_all.detach().cpu()), 'status': np.array(events_all.detach().cpu())})
    data_km.to_csv(os.path.join(km_k_path, str(epoch) + 'risk_time_status.csv'))
    median_risk = data_km['risk'].median()
    high_risk_group = data_km[data_km['risk'] > median_risk]
    low_risk_group = data_km[data_km['risk'] <= median_risk]
    kmf_high_risk = KaplanMeierFitter()
    kmf_high_risk.fit(high_risk_group['time'], event_observed=high_risk_group['status'])
    kmf_low_risk = KaplanMeierFitter()
    kmf_low_risk.fit(low_risk_group['time'], event_observed=low_risk_group['status'])
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Cindex: {:.2f}% ,Hazard_ratio :{:.4f}%\n'.format(len(predict), valid_loss, c_index, haztio_ratio))
    if cindex_best < c_index:
        cindex_best = c_index
        cindex_best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_k_path, '{}_cnn_encoder_epoch{}.pth'.format(fold, epoch + 1)))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_k_path, '{}_optimizer_epoch{}.pth'.format(fold, epoch + 1)))
    if hr_best < haztio_ratio:
        hr_best = haztio_ratio
        hr_best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_k_path, '{}_cnn_encoder_epoch{}.pth'.format(fold, epoch + 1)))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_k_path, '{}_optimizer_epoch{}.pth'.format(fold, epoch + 1)))
    return (valid_loss, c_index, haztio_ratio, cindex_best, cindex_best_epoch, hr_best, hr_best_epoch)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)
train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': True, 'drop_last': True} if use_cuda else {}
test_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 0, 'pin_memory': True, 'drop_last': False} if use_cuda else {}
train_transform = transforms.Compose([transforms.Resize([res_size, res_size]), transforms.ToTensor()])
transform = transforms.Compose([transforms.Resize([res_size, res_size]), transforms.ToTensor()])
aug_train = A.Compose([A.OneOf([A.RandomSizedCrop(min_max_height=(80, 101), height=res_size, width=res_size, p=0.5), A.PadIfNeeded(min_height=res_size, min_width=res_size, p=0.5)], p=1), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), ToTensorV2()])
aug_test = A.Compose([A.Resize(res_size, res_size), ToTensorV2()])
root_path = 'result'
if not os.path.exists(root_path):
    os.makedirs(root_path)
cross_validation_roc = 'cross-validation-roc'
cross_validation_checkpoint = 'cross-validation-checkpoint'
result_path = os.path.join(root_path, 'cross-validation-result')
roc_path = os.path.join(root_path, cross_validation_roc)
checkpoint_path = os.path.join(root_path, cross_validation_checkpoint)
if not os.path.exists(roc_path):
    os.makedirs(roc_path)
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
if not os.path.exists(result_path):
    os.mkdir(result_path)
txt_path = 'cross-validation-txt/4-fold-event-time'
qianzhui = '0315_three_model_aug_yuanfazao'
kfolds = 4
for fold in range(kfolds):
    if fold == 2:
        if fold == 0:
            pretain_model_path = 'pretain_weight'
            pretain_model_path = os.path.join(pretain_model_path, '0_cnn_encoder_epoch94.pth')
        elif fold == 1:
            pretain_model_path = 'pretain_weight'
            pretain_model_path = os.path.join(pretain_model_path, '0_cnn_encoder_epoch94.pth')
        elif fold == 2:
            pretain_model_path = 'pretain_weight'
            pretain_model_path = os.path.join(pretain_model_path, '0_cnn_encoder_epoch94.pth')
        elif fold == 3:
            pretain_model_path = 'pretain_weight'
            pretain_model_path = os.path.join(pretain_model_path, '0_cnn_encoder_epoch49.pth')
        elif fold == 4:
            pretain_model_path = 'pretain_weight'
            pretain_model_path = os.path.join(pretain_model_path, '0_cnn_encoder_epoch49.pth')
        cindex_best = 0
        cindex_best_epoch = 0
        hr_best = 0
        hr_best_epoch = 0
        selected_frames = 10
        result_f = open(os.path.join(result_path, qianzhui + pretain_model_path.split('/')[-1].split('.')[0][-2] + '2_cindex_hr' + str(selected_frames) + '_' + str(learning_rate) + '_' + 'result' + '_' + str(fold) + '.txt'), 'a+')
        print(f'第{fold}折交叉验证')
        result_f.write(f'第{fold}折交叉验证')
        km_k_path = os.path.join(roc_path, qianzhui + pretain_model_path.split('/')[-1].split('.')[0][-2] + '2_cindex_hr' + str(selected_frames) + '_' + str(learning_rate) + '_' + 'km' + '_' + str(fold) + '_' + 'folds')
        if not os.path.exists(km_k_path):
            os.mkdir(km_k_path)
        checkpoint_k_path = os.path.join(checkpoint_path, qianzhui + pretain_model_path.split('/')[-1].split('.')[0][-2] + '2_cindex_hr' + str(selected_frames) + '_' + str(learning_rate) + '_' + 'roc' + '_' + 'checkpoint' + '_' + str(fold) + '_' + 'folds')
        if not os.path.exists(checkpoint_k_path):
            os.mkdir(checkpoint_k_path)
        train_data_path = os.path.join(txt_path, 'recurrence_event_time_train_split' + str(fold) + '.txt')
        test_data_path = os.path.join(txt_path, 'recurrence_event_time_val_split' + str(fold) + '.txt')
        train_liver_pathes = []
        train_tumor_pathes = []
        train_liver_mask_pathes = []
        train_tumor_mask_pathes = []
        train_liver_TV_pathes = []
        train_tumor_TV_pathes = []
        train_liver_TV_mask_pathes = []
        train_tumor_TV_mask_pathes = []
        train_labeles = []
        train_events = []
        train_times = []
        test_liver_pathes = []
        test_tumor_pathes = []
        test_liver_mask_pathes = []
        test_tumor_mask_pathes = []
        test_liver_TV_pathes = []
        test_tumor_TV_pathes = []
        test_liver_TV_mask_pathes = []
        test_tumor_TV_mask_pathes = []
        test_labeles = []
        test_events = []
        test_times = []
        with open(train_data_path) as read:
            train_img_list = [line.strip() for line in read.readlines()]
        assert len(train_img_list) > 0, "in '{}' file does not find any information.".format(train_img_list)
        for patient_time_event in train_img_list:
            patient = patient_time_event.split(' ')[0]
            t2_patient = patient
            tv_patient = patient
            path_liver = os.path.join(t2_liver_data_path, t2_patient)
            path_tumor = os.path.join(t2_tumor_data_path, t2_patient)
            path_liver_mask = os.path.join(t2_liver_mask_path, t2_patient)
            path_tumor_mask = os.path.join(t2_tumor_mask_path, t2_patient)
            path_liver_tv = os.path.join(tv_liver_data_path, tv_patient)
            path_tumor_tv = os.path.join(tv_tumor_data_path, tv_patient)
            path_liver_tv_mask = os.path.join(tv_liver_mask_path, tv_patient)
            path_tumor_tv_mask = os.path.join(tv_tumor_mask_path, tv_patient)
            label = patient[0]
            event = patient_time_event.split(' ')[-1]
            time = patient_time_event.split(' ')[-2]
            train_liver_pathes.append(path_liver)
            train_tumor_pathes.append(path_tumor)
            train_liver_mask_pathes.append(path_liver_mask)
            train_tumor_mask_pathes.append(path_tumor_mask)
            train_liver_TV_pathes.append(path_liver_tv)
            train_tumor_TV_pathes.append(path_tumor_tv)
            train_liver_TV_mask_pathes.append(path_liver_tv_mask)
            train_tumor_TV_mask_pathes.append(path_tumor_tv_mask)
            train_labeles.append(label)
            train_events.append(event)
            train_times.append(time)
        with open(test_data_path) as read:
            test_img_list = [line.strip() for line in read.readlines()]
        assert len(test_img_list) > 0, "in '{}' file does not find any information.".format(train_img_list)
        for patient_time_event in test_img_list:
            patient = patient_time_event.split(' ')[0]
            t2_patient = patient
            tv_patient = patient
            path_liver = os.path.join(t2_liver_data_path, t2_patient)
            path_tumor = os.path.join(t2_tumor_data_path, t2_patient)
            path_liver_mask = os.path.join(t2_liver_mask_path, t2_patient)
            path_tumor_mask = os.path.join(t2_tumor_mask_path, t2_patient)
            path_liver_tv = os.path.join(tv_liver_data_path, tv_patient)
            path_tumor_tv = os.path.join(tv_tumor_data_path, tv_patient)
            path_liver_tv_mask = os.path.join(tv_liver_mask_path, tv_patient)
            path_tumor_tv_mask = os.path.join(tv_tumor_mask_path, tv_patient)
            label = patient[0]
            event = patient_time_event.split(' ')[-1]
            time = patient_time_event.split(' ')[-2]
            test_liver_pathes.append(path_liver)
            test_tumor_pathes.append(path_tumor)
            test_liver_mask_pathes.append(path_liver_mask)
            test_tumor_mask_pathes.append(path_tumor_mask)
            test_liver_TV_pathes.append(path_liver_tv)
            test_tumor_TV_pathes.append(path_tumor_tv)
            test_liver_TV_mask_pathes.append(path_liver_tv_mask)
            test_tumor_TV_mask_pathes.append(path_tumor_tv_mask)
            test_labeles.append(label)
            test_events.append(event)
            test_times.append(time)
        train_set, valid_set = (Dataset_t2_tv_tumor_liver_recurrence_aug(train_liver_pathes, train_tumor_pathes, train_liver_mask_pathes, train_tumor_mask_pathes, train_liver_TV_pathes, train_tumor_TV_pathes, train_liver_TV_mask_pathes, train_tumor_TV_mask_pathes, train_labeles, train_events, train_times, selected_frames, transform=(train_transform, aug_train)), Dataset_t2_tv_tumor_liver_recurrence_aug(test_liver_pathes, train_tumor_pathes, test_liver_mask_pathes, test_tumor_mask_pathes, test_liver_TV_pathes, train_tumor_TV_pathes, test_liver_TV_mask_pathes, test_tumor_TV_mask_pathes, test_labeles, test_events, test_times, selected_frames, transform=(transform, aug_test)))
        train_loader = data.DataLoader(train_set, **train_params)
        valid_loader = data.DataLoader(valid_set, **test_params)
        model = CNN3d_t2_tv_hgnn_0414_three_model(num_classes=1, points=128, num_channels=32).to(device)
        print('Using', torch.cuda.device_count(), 'GPU!')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretain_model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        parameters = model.parameters()
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)
        for epoch in tqdm(range(epochs)):
            train_losses, train_cindex, train_haztio, median_risk = train_recurrence(model, device, train_loader, optimizer, epoch)
            test_loss, c_index, haztio_ratio, cindex_best, cindex_best_epoch, hr_best, hr_best_epoch = recurrence_validation(model, median_risk, device, optimizer, test_loader=valid_loader, fold=0, result_f=result_f, cindex_best=cindex_best, cindex_best_epoch=cindex_best_epoch, hr_best=hr_best, hr_best_epoch=hr_best_epoch)
            result_f.write(f'\nkfold:{k} test:,cindex_best:{cindex_best},cindex_best_epoch:{cindex_best_epoch},hr_best:{hr_best},hr_best_epoch:{hr_best_epoch},media_risk={median_risk}')
            result_f.write(f'\nkfold:{k} epoch:{epoch} test:,c_index:{c_index},haztio_ratio:{haztio_ratio}')
            result_f.write(f'\nkfold:{k} train:,cindex:{train_cindex},hr:{train_haztio}')
            scheduler.step(train_losses)
        fold += 1