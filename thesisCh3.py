#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三章：增强型聚合优化 - 改进版
添加了3个对比方法：FedAvg、FedProx、SCAFFOLD
支持Linux服务器多GPU环境
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, OrderedDict
import copy
import random
from typing import Dict, List, Tuple, Optional
import warnings
import os
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
import time
import torch.optim as optim
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json

warnings.filterwarnings('ignore')


# ================== 配置类 ==================
class Config:
    def __init__(self):
        # 硬件配置检测
        if torch.cuda.is_available():
            print(f"CUDA可用，版本: {torch.version.cuda}")
            print(f"检测到 {torch.cuda.device_count()} 个GPU")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            self.DEVICE = torch.device('cuda:0')
            self.NUM_GPUS = torch.cuda.device_count()
            self.DEVICES = [torch.device(f'cuda:{i}') for i in range(self.NUM_GPUS)]
            self.NUM_WORKERS = 4
            torch.cuda.empty_cache()
        else:
            self.DEVICE = torch.device('cpu')
            self.NUM_GPUS = 0
            self.DEVICES = [self.DEVICE]
            self.NUM_WORKERS = 0

        # 数据集配置
        self.DATASET = 'CIFAR10'
        self.NUM_CLASSES = 10
        self.NUM_CLIENTS = 20
        self.CLIENTS_PER_ROUND = 5
        self.NUM_ROUNDS = 50
        self.LOCAL_EPOCHS = 5
        self.LOCAL_BATCH_SIZE = 32
        self.LOCAL_LR = 0.01
        self.NON_IID_ALPHA = 0.5
        self.LABEL_RATIO = 0.6

        # FedProx参数
        self.FEDPROX_MU = 0.01

        # SCAFFOLD参数
        self.SCAFFOLD_LR = 0.01

        # 激励机制配置
        self.TOTAL_BUDGET = 1000
        self.BUDGET_PER_ROUND = self.TOTAL_BUDGET / self.NUM_ROUNDS

        # 并行训练配置
        self.PARALLEL_CLIENTS = min(5, self.NUM_GPUS) if self.NUM_GPUS > 0 else 1

        # 混合精度训练
        self.USE_AMP = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

        # 路径
        self.DATA_PATH = ''
        self.CHECKPOINT_PATH = './checkpoints'
        self.RESULT_PATH = './results'
        os.makedirs(self.DATA_PATH, exist_ok=True)
        os.makedirs(self.CHECKPOINT_PATH, exist_ok=True)
        os.makedirs(self.RESULT_PATH, exist_ok=True)


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================== 模型定义 ==================
class CNNModel(nn.Module):
    """用于CIFAR-10的CNN模型"""

    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def extract_features(self, x):
        """提取特征用于掩码学习"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return x


class FMNISTModel(nn.Module):
    """用于Fashion-MNIST的模型"""

    def __init__(self, num_classes=10):
        super(FMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def extract_features(self, x):
        """提取特征"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return x


# ================== 数据管理 ==================
class NonIIDDataManager:
    """Non-IID数据管理器"""

    def __init__(self, dataset_name='CIFAR10', num_clients=20, alpha=0.5, label_ratio=0.6):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.alpha = alpha
        self.label_ratio = label_ratio

        # 加载数据集
        self.train_dataset, self.test_dataset = self._load_dataset()

        # 创建Non-IID分割
        self.client_indices = self._create_non_iid_split()

        # 创建客户端数据集
        self.client_datasets = self._create_client_datasets()

    def _load_dataset(self):
        """加载数据集"""
        if self.dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
        else:  # Fashion-MNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_dataset = torchvision.datasets.FashionMNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.FashionMNIST(
                root='./data', train=False, download=True, transform=transform
            )

        return train_dataset, test_dataset

    def _create_non_iid_split(self):
        """创建Non-IID数据分割"""
        targets = np.array(self.train_dataset.targets)
        num_classes = len(np.unique(targets))

        # Dirichlet分布创建Non-IID
        client_indices = [[] for _ in range(self.num_clients)]
        idx_by_class = [np.where(targets == i)[0] for i in range(num_classes)]

        for c in range(num_classes):
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            proportions = (proportions * len(idx_by_class[c])).astype(int)
            proportions[-1] = len(idx_by_class[c]) - proportions[:-1].sum()

            idx_shuffle = np.random.permutation(idx_by_class[c])
            start = 0
            for i, prop in enumerate(proportions):
                if prop > 0:
                    client_indices[i].extend(idx_shuffle[start:start + prop])
                    start += prop

        return client_indices

    def _create_client_datasets(self):
        """创建客户端数据集"""
        client_datasets = []
        for client_id, indices in enumerate(self.client_indices):
            dataset = ClientDataset(
                self.train_dataset,
                indices,
                self.label_ratio,
                client_id
            )
            client_datasets.append(dataset)
        return client_datasets

    def get_client_dataloader(self, client_id, batch_size=32):
        """获取客户端数据加载器"""
        return DataLoader(
            self.client_datasets[client_id],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

    def get_test_dataloader(self, batch_size=64):
        """获取测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    def get_client_data_distribution(self):
        """获取客户端数据分布信息"""
        distributions = []

        for client_id, dataset in enumerate(self.client_datasets):
            class_counts = defaultdict(int)
            for _, label, _ in dataset:
                if label >= 0:
                    class_counts[label.item() if torch.is_tensor(label) else label] += 1

            total = sum(class_counts.values())
            distribution = np.zeros(10)
            for class_id, count in class_counts.items():
                distribution[class_id] = count / total if total > 0 else 0

            distributions.append({
                'client_id': client_id,
                'total_samples': len(dataset),
                'labeled_samples': dataset.num_labeled,
                'class_distribution': distribution,
                'entropy': -np.sum(distribution * np.log(distribution + 1e-10)) if distribution.sum() > 0 else 0
            })

        return distributions


class ClientDataset(Dataset):
    """客户端数据集"""

    def __init__(self, base_dataset, indices, label_ratio, client_id):
        self.base_dataset = base_dataset
        self.indices = indices
        self.client_id = client_id

        num_labeled = int(len(indices) * label_ratio)
        shuffled_indices = np.random.permutation(len(indices))

        self.labeled_mask = np.zeros(len(indices), dtype=bool)
        self.labeled_mask[shuffled_indices[:num_labeled]] = True

        self.num_labeled = num_labeled
        self.num_unlabeled = len(indices) - num_labeled

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data, label = self.base_dataset[real_idx]
        is_labeled = self.labeled_mask[idx]

        if not is_labeled:
            label = -1

        return data, label, is_labeled


# ================== 掩码学习模块 ==================
class MaskLearningModule:
    """增强型掩码学习模块"""

    def __init__(self, num_classes=10, feature_dim=256):
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # 注意力机制参数
        self.attention_weights = torch.randn(feature_dim, feature_dim) * 0.01

        # 课程学习参数
        self.curriculum_threshold = 0.5
        self.curriculum_warmup = 20

        # 对比学习温度
        self.temperature = 0.5

        # 困难样本挖掘
        self.hard_negative_ratio = 0.3

    def compute_enhanced_loss(self, outputs, labels, is_labeled, features, epoch):
        """计算增强损失"""
        device = outputs.device
        total_loss = torch.tensor(0.0).to(device)

        # 有标签数据的监督损失
        if is_labeled.sum() > 0:
            labeled_outputs = outputs[is_labeled]
            labeled_targets = labels[is_labeled]

            valid_mask = labeled_targets >= 0
            if valid_mask.sum() > 0:
                valid_outputs = labeled_outputs[valid_mask]
                valid_targets = labeled_targets[valid_mask]
                supervised_loss = F.cross_entropy(valid_outputs, valid_targets)
                total_loss = total_loss + supervised_loss

        # 无标签数据的增强损失
        if (~is_labeled).sum() > 0:
            unlabeled_outputs = outputs[~is_labeled]

            # 1. 伪标签损失
            with torch.no_grad():
                pseudo_probs = F.softmax(unlabeled_outputs, dim=1)
                max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)

                # 动态阈值
                threshold = 0.7 + 0.2 * min(epoch / 50, 1.0)
                confident_mask = max_probs > threshold

            if confident_mask.sum() > 0:
                confident_outputs = unlabeled_outputs[confident_mask]
                confident_labels = pseudo_labels[confident_mask]
                pseudo_loss = F.cross_entropy(confident_outputs, confident_labels)
                total_loss = total_loss + 0.5 * pseudo_loss

            # 2. 一致性正则化
            augmented_outputs = outputs[~is_labeled]
            consistency_loss = F.mse_loss(
                F.softmax(unlabeled_outputs, dim=1),
                F.softmax(augmented_outputs.detach(), dim=1)
            )
            total_loss = total_loss + 0.1 * consistency_loss

            # 3. 熵最小化
            entropy = -(pseudo_probs * torch.log(pseudo_probs + 1e-8)).sum(dim=1).mean()
            total_loss = total_loss + 0.1 * entropy

        # 4. 对比学习损失（如果有特征）
        if features is not None and is_labeled.sum() > 1:
            # 只传递有标签数据的特征和标签
            labeled_features = features[is_labeled]
            labeled_labels = labels[is_labeled]
            # 创建全为True的掩码，因为我们已经筛选了有标签数据
            labeled_mask = torch.ones(len(labeled_features), dtype=torch.bool, device=features.device)

            contrast_loss = self.compute_contrastive_loss(
                labeled_features, labeled_labels, labeled_mask
            )
            total_loss = total_loss + 0.2 * contrast_loss

        return total_loss

    def compute_contrastive_loss(self, features, labels, is_labeled):
        """计算对比学习损失"""
        if len(features) < 2:
            return torch.tensor(0.0).to(features.device)

        # 归一化特征
        features = F.normalize(features, p=2, dim=1)

        # 计算相似度矩阵
        similarity = torch.mm(features, features.t()) / self.temperature

        # 创建标签掩码
        labels_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        labels_mask = labels_mask & is_labeled.unsqueeze(1) & is_labeled.unsqueeze(0)
        labels_mask.fill_diagonal_(False)

        if not labels_mask.any():
            return torch.tensor(0.0).to(features.device)

        # 计算对比损失
        exp_sim = torch.exp(similarity)
        positive_sim = (exp_sim * labels_mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1) - exp_sim.diag()

        # 修复：找到有正样本对的索引
        valid_indices = positive_sim > 0

        if valid_indices.sum() > 0:
            # 使用相同的索引来获取positive_sim和all_sim
            valid_positive_sim = positive_sim[valid_indices]
            valid_all_sim = all_sim[valid_indices]
            loss = -torch.log(valid_positive_sim / (valid_all_sim + 1e-8)).mean()
        else:
            loss = torch.tensor(0.0).to(features.device)

        return loss


# ================== 增强聚合模块 ==================
class EnhancedAggregation:
    """增强型质量感知聚合"""

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.anomaly_detector = LocalOutlierFactor(contamination=0.1)
        self.mask_learning = MaskLearningModule(num_classes, feature_dim=256)

        # 元学习参数
        self.meta_weights = {
            'W_q': torch.randn(4, 256) * 0.01,
            'W_g': torch.randn(256, 256) * 0.01,
            'v': torch.randn(256) * 0.01,
            'b': torch.zeros(256)
        }

    def compute_quality_scores(self, models, data_info, global_model):
        """计算四维质量得分"""
        scores = []

        for i, model in enumerate(models):
            # 数据质量
            data_quality = self._compute_data_quality(data_info[i])

            # 模型质量
            model_quality = self._compute_model_quality(model, global_model)

            # 稳定性
            stability = self._compute_stability(model, global_model)

            # 多样性
            diversity = self._compute_diversity(model, models)

            # 综合得分
            score = 0.3 * data_quality + 0.3 * model_quality + 0.2 * stability + 0.2 * diversity
            scores.append(score)

        return np.array(scores)

    def _compute_data_quality(self, info):
        """计算数据质量"""
        # 基于类分布的熵
        entropy = info.get('entropy', 0)

        # 标签数据比例
        label_ratio = info['labeled_samples'] / max(info['total_samples'], 1)

        # 样本数量因子
        sample_factor = min(info['total_samples'] / 500, 1.0)

        return 0.4 * (1 - entropy / 3.32) + 0.3 * label_ratio + 0.3 * sample_factor

    def _compute_model_quality(self, model, global_model):
        """计算模型质量"""
        device = next(global_model.parameters()).device

        # 确保模型在同一设备上
        if next(model.parameters()).device != device:
            model = model.to(device)

        # 计算参数差异
        diff_norm = 0
        param_count = 0

        for (name1, param1), (name2, param2) in zip(model.named_parameters(), global_model.named_parameters()):
            diff = param1.data - param2.data
            diff_norm += torch.norm(diff).item()
            param_count += 1

        # 归一化
        if param_count > 0:
            diff_norm = diff_norm / param_count

        # 转换为质量分数
        quality = 1 / (1 + diff_norm)

        return quality

    def _compute_stability(self, model, global_model):
        """计算稳定性"""
        device = next(global_model.parameters()).device

        if next(model.parameters()).device != device:
            model = model.to(device)

        # 计算梯度方差
        variance = 0
        param_count = 0

        for param1, param2 in zip(model.parameters(), global_model.parameters()):
            diff = param1.data - param2.data
            variance += torch.var(diff).item()
            param_count += 1

        if param_count > 0:
            variance = variance / param_count

        # 稳定性得分
        stability = 1 / (1 + variance)

        return stability

    def _compute_diversity(self, model, all_models):
        """计算多样性"""
        if len(all_models) <= 1:
            return 1.0

        device = next(model.parameters()).device

        # 计算与其他模型的平均差异
        total_diff = 0
        count = 0

        for other_model in all_models:
            if other_model is not model:
                if next(other_model.parameters()).device != device:
                    other_model = other_model.to(device)

                diff = 0
                param_count = 0
                for p1, p2 in zip(model.parameters(), other_model.parameters()):
                    diff += torch.norm(p1.data - p2.data).item()
                    param_count += 1

                if param_count > 0:
                    total_diff += diff / param_count
                    count += 1

        if count > 0:
            diversity = total_diff / count
            # 归一化到[0,1]
            diversity = min(diversity / 10, 1.0)
        else:
            diversity = 0.5

        return diversity

    def detect_anomalies(self, models, threshold=0.3):
        """检测异常模型"""
        if len(models) < 3:
            return np.ones(len(models), dtype=bool)

        # 提取模型特征
        features = []
        for model in models:
            feature_vector = []
            for param in model.parameters():
                feature_vector.append(torch.norm(param.data).item())
                feature_vector.append(torch.mean(param.data).item())
                feature_vector.append(torch.std(param.data).item())
            features.append(feature_vector[:100])

        features = np.array(features)

        # 使用LOF检测异常
        try:
            predictions = self.anomaly_detector.fit_predict(features)
            return predictions == 1
        except:
            return np.ones(len(models), dtype=bool)

    def aggregate_with_quality(self, models, quality_scores, weights=None, round_idx=0):
        """质量感知聚合"""
        if len(models) == 0:
            return None

        # 检测并过滤异常模型
        normal_mask = self.detect_anomalies(models)

        valid_indices = np.where(normal_mask)[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(len(models))

        valid_models = [models[i] for i in valid_indices]
        valid_scores = quality_scores[valid_indices]

        # 激励驱动的权重融合
        if weights is not None:
            incentive_weights = weights[valid_indices]
            gamma = 0.6
            aggregation_weights = gamma * valid_scores + (1 - gamma) * incentive_weights
        else:
            aggregation_weights = valid_scores

        # 公平性约束
        aggregation_weights = self._apply_fairness_constraint(aggregation_weights)

        # 归一化
        if aggregation_weights.sum() > 0:
            aggregation_weights = aggregation_weights / aggregation_weights.sum()
        else:
            aggregation_weights = np.ones(len(valid_models)) / len(valid_models)

        # 加权聚合
        aggregated_state = OrderedDict()

        for key in valid_models[0].state_dict().keys():
            params = [model.state_dict()[key] for model in valid_models]

            # 检查参数是否为浮点类型
            if params[0].is_floating_point():
                if round_idx % 10 == 0 and round_idx > 0:
                    # 定期使用Trimmed Mean增强鲁棒性
                    aggregated_state[key] = self._trimmed_mean_aggregation(params, trim_ratio=0.1)
                else:
                    # 正常加权聚合
                    aggregated_state[key] = sum([
                        aggregation_weights[i] * params[i]
                        for i in range(len(valid_models))
                    ])
            else:
                # 对于非浮点参数，直接复制第一个模型的值
                aggregated_state[key] = params[0].clone()

        return aggregated_state

    def _apply_fairness_constraint(self, weights):
        """应用公平性约束"""
        if len(weights) == 0:
            return weights

        # 最大权重不超过平均值的3倍
        max_weight = 3.0 / len(weights)
        weights = np.minimum(weights, max_weight)

        return weights

    def _trimmed_mean_aggregation(self, params, trim_ratio=0.1):
        """裁剪均值聚合"""
        stacked = torch.stack(params)
        trim_num = int(len(params) * trim_ratio)

        if trim_num > 0 and len(params) > 2:
            if len(stacked.shape) > 1:
                sorted_params, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_params[trim_num:-trim_num]
            else:
                sorted_params = torch.sort(stacked)[0]
                trimmed = sorted_params[trim_num:-trim_num]

            return trimmed.mean(dim=0)
        else:
            return stacked.mean(dim=0)


# ================== 基线方法：FedAvg ==================
class FedAvgAggregator:
    """FedAvg聚合器"""

    def aggregate(self, models, data_info=None):
        """简单平均聚合"""
        if len(models) == 0:
            return None

        aggregated_state = OrderedDict()

        for key in models[0].state_dict().keys():
            params = [model.state_dict()[key] for model in models]

            if params[0].is_floating_point():
                # 浮点参数：取平均
                aggregated_state[key] = torch.stack(params).mean(0)
            else:
                # 非浮点参数：复制第一个
                aggregated_state[key] = params[0].clone()

        return aggregated_state


# ================== 基线方法：FedProx ==================
class FedProxTrainer:
    """FedProx训练器"""

    def __init__(self, mu=0.01):
        self.mu = mu

    def train_client(self, client_id, local_model, global_model, dataloader, config, device):
        """FedProx客户端训练"""
        local_model = local_model.to(device)
        global_model = global_model.to(device)
        local_model.train()

        optimizer = optim.SGD(local_model.parameters(), lr=config.LOCAL_LR, momentum=0.9)

        for epoch in range(config.LOCAL_EPOCHS):
            for batch_data, batch_labels, is_labeled in dataloader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                is_labeled = is_labeled.to(device)

                optimizer.zero_grad()

                outputs = local_model(batch_data)

                # 标准损失
                loss = 0
                if is_labeled.sum() > 0:
                    labeled_outputs = outputs[is_labeled]
                    labeled_targets = batch_labels[is_labeled]

                    valid_mask = labeled_targets >= 0
                    if valid_mask.sum() > 0:
                        valid_outputs = labeled_outputs[valid_mask]
                        valid_targets = labeled_targets[valid_mask]
                        loss = F.cross_entropy(valid_outputs, valid_targets)

                # FedProx近端项
                proximal_term = 0
                for w, w_global in zip(local_model.parameters(), global_model.parameters()):
                    proximal_term += (self.mu / 2) * torch.norm(w - w_global) ** 2

                total_loss = loss + proximal_term

                if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                    total_loss.backward()
                    optimizer.step()

        return local_model.cpu()


# ================== 基线方法：SCAFFOLD ==================
class SCAFFOLDTrainer:
    """SCAFFOLD训练器"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.control_variates = {}  # 存储控制变量
        self.server_control = None

    def initialize_controls(self, num_clients, model_template):
        """初始化控制变量"""
        # 初始化客户端控制变量
        for i in range(num_clients):
            self.control_variates[i] = {
                key: torch.zeros_like(param)
                for key, param in model_template.state_dict().items()
                if param.is_floating_point()
            }

        # 初始化服务器控制变量
        self.server_control = {
            key: torch.zeros_like(param)
            for key, param in model_template.state_dict().items()
            if param.is_floating_point()
        }

    def train_client(self, client_id, local_model, global_model, dataloader, config, device):
        """SCAFFOLD客户端训练"""
        local_model = local_model.to(device)
        global_model = global_model.to(device)
        local_model.train()

        # 保存初始模型
        initial_model = copy.deepcopy(local_model)

        optimizer = optim.SGD(local_model.parameters(), lr=self.lr)

        for epoch in range(config.LOCAL_EPOCHS):
            for batch_data, batch_labels, is_labeled in dataloader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                is_labeled = is_labeled.to(device)

                optimizer.zero_grad()

                outputs = local_model(batch_data)

                # 计算损失
                loss = 0
                if is_labeled.sum() > 0:
                    labeled_outputs = outputs[is_labeled]
                    labeled_targets = batch_labels[is_labeled]

                    valid_mask = labeled_targets >= 0
                    if valid_mask.sum() > 0:
                        valid_outputs = labeled_outputs[valid_mask]
                        valid_targets = labeled_targets[valid_mask]
                        loss = F.cross_entropy(valid_outputs, valid_targets)

                if isinstance(loss, torch.Tensor) and loss.requires_grad:
                    loss.backward()

                    # 应用SCAFFOLD修正
                    with torch.no_grad():
                        for (name, param), (_, global_param) in zip(
                                local_model.named_parameters(),
                                global_model.named_parameters()
                        ):
                            if param.grad is not None and name in self.control_variates[client_id]:
                                # 应用控制变量修正
                                param.grad.data -= self.control_variates[client_id][name].to(device)
                                param.grad.data += self.server_control[name].to(device)

                    optimizer.step()

        # 更新控制变量
        self._update_control_variates(client_id, local_model, initial_model, config)

        return local_model.cpu()

    def _update_control_variates(self, client_id, local_model, initial_model, config):
        """更新控制变量"""
        with torch.no_grad():
            for (name, param), (_, init_param) in zip(
                    local_model.named_parameters(),
                    initial_model.named_parameters()
            ):
                if name in self.control_variates[client_id]:
                    # 计算新的控制变量
                    delta = (init_param - param) / (config.LOCAL_EPOCHS * self.lr)
                    self.control_variates[client_id][name] = delta.cpu()

    def update_server_control(self, selected_clients):
        """更新服务器控制变量"""
        if len(selected_clients) == 0:
            return

        with torch.no_grad():
            for key in self.server_control.keys():
                # 平均选中客户端的控制变量
                avg_control = torch.zeros_like(self.server_control[key])
                for client_id in selected_clients:
                    if client_id in self.control_variates:
                        avg_control += self.control_variates[client_id][key]

                self.server_control[key] = avg_control / len(selected_clients)


# ================== 并行训练器 ==================
class ParallelLocalTrainer:
    """并行本地训练器"""

    def __init__(self, data_manager, config):
        self.data_manager = data_manager
        self.config = config
        self.devices = config.DEVICES

    def train_clients_parallel(self, selected_clients, global_model, method='enhanced',
                               fedprox_trainer=None, scaffold_trainer=None):
        """并行训练多个客户端"""
        if len(self.devices) == 0:
            self.devices = [torch.device('cpu')]

        # 将模型状态字典移到CPU
        global_model_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

        # 准备参数
        args_list = []
        for i, client_id in enumerate(selected_clients):
            device_id = i % len(self.devices)
            args_list.append((
                client_id,
                global_model_state,
                device_id,
                method,
                fedprox_trainer,
                scaffold_trainer
            ))

        # 使用线程池并行训练
        with ThreadPoolExecutor(max_workers=self.config.PARALLEL_CLIENTS) as executor:
            trained_models = list(executor.map(self.train_single_client, args_list))

        return trained_models

    def train_single_client(self, args):
        """单个客户端训练"""
        client_id, global_model_state, device_id, method, fedprox_trainer, scaffold_trainer = args

        # 选择设备
        if len(self.devices) > 1:
            device = self.devices[device_id]
        else:
            device = self.devices[0]

        # 创建本地模型
        if self.config.DATASET == 'CIFAR10':
            local_model = CNNModel(num_classes=self.config.NUM_CLASSES)
            global_model = CNNModel(num_classes=self.config.NUM_CLASSES)
        else:
            local_model = FMNISTModel(num_classes=self.config.NUM_CLASSES)
            global_model = FMNISTModel(num_classes=self.config.NUM_CLASSES)

        local_model.load_state_dict(global_model_state)
        global_model.load_state_dict(global_model_state)

        # 获取数据加载器
        dataloader = self.data_manager.get_client_dataloader(
            client_id, self.config.LOCAL_BATCH_SIZE
        )

        # 根据方法选择训练方式
        if method == 'fedprox' and fedprox_trainer is not None:
            # FedProx训练
            local_model = fedprox_trainer.train_client(
                client_id, local_model, global_model, dataloader, self.config, device
            )
        elif method == 'scaffold' and scaffold_trainer is not None:
            # SCAFFOLD训练
            local_model = scaffold_trainer.train_client(
                client_id, local_model, global_model, dataloader, self.config, device
            )
        else:
            # 增强型训练（默认）
            local_model = self._enhanced_train(
                local_model, dataloader, device
            )

        return local_model

    def _enhanced_train(self, local_model, dataloader, device):
        """增强型训练"""
        local_model = local_model.to(device)
        local_model.train()

        optimizer = optim.SGD(local_model.parameters(), lr=self.config.LOCAL_LR, momentum=0.9)

        # 掩码学习模块
        mask_module = MaskLearningModule(num_classes=self.config.NUM_CLASSES)

        for epoch in range(self.config.LOCAL_EPOCHS):
            for batch_data, batch_labels, is_labeled in dataloader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                is_labeled = is_labeled.to(device)

                optimizer.zero_grad()

                outputs = local_model(batch_data)

                # 提取特征
                if hasattr(local_model, 'extract_features'):
                    features = local_model.extract_features(batch_data)
                else:
                    features = outputs.detach()

                # 计算增强损失
                loss = mask_module.compute_enhanced_loss(
                    outputs, batch_labels, is_labeled, features, epoch
                )

                if isinstance(loss, torch.Tensor) and loss.requires_grad:
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)

                    optimizer.step()

        return local_model.cpu()


# ================== 实验运行函数 ==================
def run_chapter3_experiment():
    """运行第三章实验 - 对比4种聚合方法"""
    print("=" * 60)
    print("第三章：增强型聚合优化对比实验")
    print("对比方法: Enhanced (Ours), FedAvg, FedProx, SCAFFOLD")
    print("=" * 60)

    # 初始化配置
    config = Config()

    # 初始化数据管理器
    data_manager = NonIIDDataManager(
        dataset_name=config.DATASET,
        num_clients=config.NUM_CLIENTS,
        alpha=config.NON_IID_ALPHA,
        label_ratio=config.LABEL_RATIO
    )

    # 初始化并行训练器
    parallel_trainer = ParallelLocalTrainer(data_manager, config)

    # 初始化模型类
    if config.DATASET == 'CIFAR10':
        model_class = CNNModel
    else:
        model_class = FMNISTModel

    # 初始化聚合器和训练器
    aggregators = {
        'Enhanced (Ours)': EnhancedAggregation(num_classes=config.NUM_CLASSES),
        'FedAvg': FedAvgAggregator(),
        'FedProx': None,  # FedProx使用特殊训练器
        'SCAFFOLD': None  # SCAFFOLD使用特殊训练器
    }

    # 初始化特殊训练器
    fedprox_trainer = FedProxTrainer(mu=config.FEDPROX_MU)
    scaffold_trainer = SCAFFOLDTrainer(lr=config.SCAFFOLD_LR)

    # 初始化SCAFFOLD控制变量
    template_model = model_class(num_classes=config.NUM_CLASSES)
    scaffold_trainer.initialize_controls(config.NUM_CLIENTS, template_model)

    # 结果存储
    results = {
        method: {
            'accuracy': [],
            'loss': [],
            'convergence_time': [],
            'robustness': []
        }
        for method in aggregators.keys()
    }

    # 初始化全局模型
    global_models = {
        method: model_class(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        for method in aggregators.keys()
    }

    # 训练循环
    for round_idx in range(config.NUM_ROUNDS):
        print(f"\nRound {round_idx + 1}/{config.NUM_ROUNDS}")

        # 随机选择客户端
        selected_clients = np.random.choice(
            config.NUM_CLIENTS,
            min(config.CLIENTS_PER_ROUND, config.NUM_CLIENTS),
            replace=False
        )

        # 获取数据分布信息
        data_distributions = data_manager.get_client_data_distribution()

        # 模拟拜占庭攻击（10%概率）
        is_byzantine_round = np.random.random() < 0.1

        for method_name in aggregators.keys():
            print(f"\nTraining with {method_name}...")
            start_time = time.time()

            # 选择训练方法
            if method_name == 'FedProx':
                train_method = 'fedprox'
            elif method_name == 'SCAFFOLD':
                train_method = 'scaffold'
            else:
                train_method = 'enhanced' if method_name == 'Enhanced (Ours)' else 'fedavg'

            # 并行训练
            local_models = parallel_trainer.train_clients_parallel(
                selected_clients,
                global_models[method_name],
                method=train_method,
                fedprox_trainer=fedprox_trainer if method_name == 'FedProx' else None,
                scaffold_trainer=scaffold_trainer if method_name == 'SCAFFOLD' else None
            )

            # 模拟拜占庭攻击
            if is_byzantine_round and round_idx > 20:
                # 随机选择一个模型进行攻击
                attack_idx = np.random.randint(0, len(local_models))
                for param in local_models[attack_idx].parameters():
                    param.data += torch.randn_like(param.data) * 10
                print(f"  Byzantine attack on client {selected_clients[attack_idx]}")

            # 聚合模型
            if method_name == 'Enhanced (Ours)':
                # 计算质量分数
                data_info = [data_distributions[i] for i in selected_clients]
                quality_scores = aggregators[method_name].compute_quality_scores(
                    local_models, data_info, global_models[method_name]
                )

                # 质量感知聚合
                aggregated_state = aggregators[method_name].aggregate_with_quality(
                    local_models, quality_scores, round_idx=round_idx
                )
            elif method_name == 'FedAvg':
                # FedAvg聚合
                aggregated_state = aggregators[method_name].aggregate(local_models)
            elif method_name == 'SCAFFOLD':
                # SCAFFOLD聚合
                aggregated_state = aggregators['FedAvg'].aggregate(local_models)
                # 更新服务器控制变量
                scaffold_trainer.update_server_control(selected_clients)
            else:  # FedProx
                # FedProx使用标准聚合
                aggregated_state = aggregators['FedAvg'].aggregate(local_models)

            if aggregated_state is not None:
                global_models[method_name].load_state_dict(aggregated_state)

            # 评估性能
            global_models[method_name].eval()
            test_loader = data_manager.get_test_dataloader()

            correct = 0
            total = 0
            total_loss = 0

            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(config.DEVICE), labels.to(config.DEVICE)
                    outputs = global_models[method_name](data)

                    loss = F.cross_entropy(outputs, labels)
                    total_loss += loss.item() * data.size(0)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            avg_loss = total_loss / total
            elapsed_time = time.time() - start_time

            # 记录结果
            results[method_name]['accuracy'].append(accuracy)
            results[method_name]['loss'].append(avg_loss)
            results[method_name]['convergence_time'].append(elapsed_time)
            results[method_name]['robustness'].append(0.5 if is_byzantine_round else 1.0)

            print(f"  {method_name}: Acc={accuracy:.4f}, Loss={avg_loss:.4f}, Time={elapsed_time:.2f}s")

    return results


# ================== 可视化函数 ==================
def plot_chapter3_results(results, save_path='chapter3_results.png'):
    """绘制第三章结果对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 设置颜色
    colors = {
        'Enhanced (Ours)': 'red',
        'FedAvg': 'blue',
        'FedProx': 'green',
        'SCAFFOLD': 'orange'
    }

    # 图1：准确率
    ax = axes[0, 0]
    for method, data in results.items():
        if len(data['accuracy']) > 0:
            smoothed = pd.Series(data['accuracy']).rolling(window=5, min_periods=1).mean()
            ax.plot(smoothed, label=method, linewidth=2, color=colors[method])
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图2：损失
    ax = axes[0, 1]
    for method, data in results.items():
        if len(data['loss']) > 0:
            smoothed = pd.Series(data['loss']).rolling(window=5, min_periods=1).mean()
            ax.plot(smoothed, label=method, linewidth=2, color=colors[method])
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图3：收敛速度
    ax = axes[0, 2]
    for method, data in results.items():
        if len(data['convergence_time']) > 0:
            ax.plot(data['convergence_time'], label=method, linewidth=2, color=colors[method])
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Time (s)')
    ax.set_title('Convergence Time per Round')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图4：鲁棒性（拜占庭攻击）
    ax = axes[1, 0]
    for method, data in results.items():
        if len(data['robustness']) > 0 and len(data['accuracy']) > 0:
            attack_rounds = [i for i, r in enumerate(data['robustness']) if r < 1.0]
            normal_rounds = [i for i, r in enumerate(data['robustness']) if r >= 1.0]

            if len(attack_rounds) > 0:
                ax.scatter(attack_rounds, [data['accuracy'][i] for i in attack_rounds],
                           color=colors[method], s=100, marker='x', label=f'{method} (attack)', alpha=0.7)

            ax.plot(data['accuracy'], linewidth=1, color=colors[method], alpha=0.5, label=method)

    ax.set_xlabel('Rounds')
    ax.set_ylabel('Accuracy')
    ax.set_title('Robustness Against Byzantine Attacks')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # 图5：最终性能对比
    ax = axes[1, 1]
    final_accuracies = {}
    for method, data in results.items():
        if len(data['accuracy']) >= 10:
            final_accuracies[method] = np.mean(data['accuracy'][-10:])

    if len(final_accuracies) > 0:
        bars = ax.bar(final_accuracies.keys(), final_accuracies.values(),
                      color=[colors[m] for m in final_accuracies.keys()])
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Final Performance Comparison')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom')

    # 图6：效率对比
    ax = axes[1, 2]
    avg_times = {}
    for method, data in results.items():
        if len(data['convergence_time']) > 0:
            avg_times[method] = np.mean(data['convergence_time'])

    if len(avg_times) > 0:
        bars = ax.bar(avg_times.keys(), avg_times.values(),
                      color=[colors[m] for m in avg_times.keys()])
        ax.set_ylabel('Average Time (s)')
        ax.set_title('Training Efficiency')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    print("\n" + "=" * 60)
    print("实验结果统计")
    print("=" * 60)
    for method in results.keys():
        if len(results[method]['accuracy']) > 0:
            final_acc = np.mean(results[method]['accuracy'][-10:]) if len(
                results[method]['accuracy']) >= 10 else np.mean(results[method]['accuracy'])
            avg_loss = np.mean(results[method]['loss'])
            avg_time = np.mean(results[method]['convergence_time'])

            # 计算鲁棒性指标
            attack_rounds = [i for i, r in enumerate(results[method]['robustness']) if r < 1.0]
            if len(attack_rounds) > 0:
                robustness_score = np.mean([results[method]['accuracy'][i] for i in attack_rounds]) / final_acc
            else:
                robustness_score = 1.0

            print(f"{method}:")
            print(f"  最终准确率: {final_acc:.4f}")
            print(f"  平均损失: {avg_loss:.4f}")
            print(f"  平均训练时间: {avg_time:.2f}s")
            print(f"  鲁棒性得分: {robustness_score:.4f}")


def save_results(results, filename='chapter3_results.json'):
    """保存实验结果"""
    config = Config()

    # 转换为可序列化格式
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, list):
                    serializable_results[key][k] = v
                elif isinstance(v, np.ndarray):
                    serializable_results[key][k] = v.tolist()
                else:
                    serializable_results[key][k] = v
        else:
            serializable_results[key] = value

    filepath = os.path.join(config.RESULT_PATH, filename)
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to {filepath}")


# ================== 主函数 ==================
def main():
    """主函数 - GPU优化版"""
    print("=" * 80)
    print("联邦学习增强型聚合优化实验 - 第三章改进版")
    print("=" * 80)

    # 初始化配置
    config = Config()
    print(f"使用设备: {config.DEVICES}")
    print(f"并行客户端数: {config.PARALLEL_CLIENTS}")
    print(f"混合精度训练: {config.USE_AMP}")

    # 设置随机种子
    set_seed(42)

    # GPU测试
    if torch.cuda.is_available():
        print("\n=== GPU计算能力测试 ===")
        device = config.DEVICES[0]

        # 测试矩阵运算
        print("测试GPU矩阵运算...")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        start_time = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        compute_time = time.time() - start_time
        print(f"  矩阵乘法时间: {compute_time:.4f}秒")

        # 清理测试张量
        del a, b, c
        torch.cuda.empty_cache()
        print("GPU测试完成！")

    # 运行实验
    try:
        print("\n开始运行第三章实验...")
        print("包含4个对比方法: Enhanced (Ours), FedAvg, FedProx, SCAFFOLD")

        # 运行实验
        results = run_chapter3_experiment()

        # 保存结果
        save_results(results, 'chapter3_results.json')

        # 绘制结果图表
        plot_chapter3_results(results, 'chapter3_results.png')

        print("\n" + "=" * 80)
        print("第三章实验完成！")
        print(f"结果已保存到: {config.RESULT_PATH}")
        print("=" * 80)

    except Exception as e:
        print(f"实验出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU缓存已清理")


if __name__ == "__main__":
    main()