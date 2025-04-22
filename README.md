# Multi-Modal Hypergraph-Based Liver Tumor Recurrence Prediction  
基于多模态超图神经网络的肝癌复发风险预测模型

---

## 主要文件、目录说明

- `20240321train_yuanfazao_gcn_auguementation.py`：模型训练主入口，包含数据加载、模型初始化、训练流程控制及指标计算等完整逻辑。
- `our_model.py`：模型架构定义，包括：
  - 多模态卷积结构（T2、TV图像）；
  - LSTM 时间建模；
  - 超图神经网络（HGNN）融合模块。
- `functions.py`：训练用的图像预处理与数据加载逻辑，定义了基础的 `Dataset` 类，主要用于单模态/三维图像输入。
- `functionsbranch.py`：论文实验中使用的多模态图像增强版本的数据加载模块（支持掩膜、STN等）。
- `requirements.txt`：项目依赖包及其版本列表，可用于环境搭建。
- `ckpts/`（建议自建）：用于保存训练中生成的模型参数。
- `result/`（建议自建）：可用于保存预测结果、log文件或模型评估图。
- `dataset/`（建议自建）：原始图像与掩膜组织结构，支持按患者目录管理，路径需在主脚本中配置。

---

## 模型说明

本项目的核心模型为 `CNN3d_t2_tv_hgnn_0414_three_model`，结合了以下模块：

- **卷积特征提取**：分支式结构分别提取 T2、TV 模态下的肿瘤区域特征；
- **超图神经网络融合**：通过构建空间超图，实现同一时序帧内邻域区域特征的非局部聚合；
- **时序建模**：使用 LSTM 融合时间维度变化，最终输出复发风险预测值。

---

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行训练主脚本：
   ```bash
   python 20240321train_yuanfazao_gcn_auguementation.py
   ```

3. 修改训练参数：
   可通过 `argparse` 设置模型结构参数、embedding维度等。
