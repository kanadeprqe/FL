#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二章：分层激励机制 - 改进版
添加了3个对比方法：原始激励机制、IncEFL、Stackelberg博弈
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
from sklearn.cluster import SpectralClustering
from scipy.stats import wasserstein_distance
import time
import torch.optim as optim
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import cvxpy as cp  # 用于IncEFL的优化求解

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
        self.DATA_PATH = ''
        self.INCEFL_MAX_LATENCY = 700.0
        self.INCEFL_SATISFACTION_DELTA = 800.0
        self.INCEFL_REWARD_UNIT_COST_PHI = 0.01

        # IncEFL Worker模型和通信参数
        self.INCEFL_CPU_CYCLES_PER_SAMPLE = 5.0
        self.INCEFL_COMM_TIME = 10.0
        # 激励机制配置
        self.TOTAL_BUDGET = 1000
        self.BUDGET_PER_ROUND = self.TOTAL_BUDGET / self.NUM_ROUNDS

        # 并行训练配置
        self.PARALLEL_CLIENTS = min(5, self.NUM_GPUS) if self.NUM_GPUS > 0 else 1

        # 混合精度训练
        self.USE_AMP = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

        # IncEFL特定参数
        self.WORKER_THETA_DIST = {'mean': 0.5, 'std': 0.15}
        self.CPU_FREQ_RANGE = [1.0, 3.0]  # GHz
        self.COMMUNICATION_DELAY = 0.01  # 秒

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
        """提取特征用于相似度计算"""
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


# ================== 数据管理 ==================
class NonIIDDataManager:
    """Non-IID数据管理器 - 处理半监督场景"""

    def __init__(self, dataset_name='CIFAR10', num_clients=20, alpha=0.5, label_ratio=0.6):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.alpha = alpha
        self.label_ratio = label_ratio

        # 加载数据集
        self.train_dataset, self.test_dataset = self._load_dataset()

        # 创建Non-IID分割
        self.client_indices = self._create_non_iid_split()

        # 创建半监督标签
        self.client_labels_mask = self._create_semi_supervised_labels()

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

    def _create_semi_supervised_labels(self):
        """创建半监督标签掩码"""
        labels_mask = {}
        for client_id, indices in enumerate(self.client_indices):
            num_samples = len(indices)
            num_labeled = int(num_samples * self.label_ratio)

            mask = np.zeros(num_samples, dtype=bool)
            labeled_idx = np.random.choice(num_samples, num_labeled, replace=False)
            mask[labeled_idx] = True

            labels_mask[client_id] = mask

        return labels_mask

    def get_client_dataloader(self, client_id, batch_size=32):
        """获取客户端数据加载器"""
        indices = self.client_indices[client_id]
        mask = self.client_labels_mask[client_id]

        dataset = ClientDataset(self.train_dataset, indices, mask)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def get_test_dataloader(self, batch_size=64):
        """获取测试数据加载器"""
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    def get_client_data_distribution(self):
        """获取客户端数据分布信息"""
        distributions = {}
        for client_id, indices in enumerate(self.client_indices):
            targets = [self.train_dataset.targets[i] for i in indices]
            class_counts = np.bincount(targets, minlength=10)

            distributions[client_id] = {
                'total_samples': len(indices),
                'labeled_samples': self.client_labels_mask[client_id].sum(),
                'class_distribution': class_counts,
                'entropy': -np.sum((class_counts / len(indices) + 1e-10) *
                                   np.log(class_counts / len(indices) + 1e-10))
            }

        return distributions


class ClientDataset(Dataset):
    """客户端数据集 - 支持半监督学习"""

    def __init__(self, dataset, indices, labels_mask):
        self.dataset = dataset
        self.indices = indices
        self.labels_mask = labels_mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data, label = self.dataset[self.indices[idx]]
        is_labeled = self.labels_mask[idx]

        if not is_labeled:
            label = -1  # 无标签数据用-1表示

        return data, label, is_labeled


# ================== 贡献度评估器 ==================
class ContributionEvaluator:
    """贡献度评估器 - 完整实现四维贡献度模型"""

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.history_window = 10
        self.contribution_history = defaultdict(lambda: deque(maxlen=self.history_window))

    def evaluate_comprehensive_contribution(self, client_id, local_model, global_model,
                                            data_info, compatibility_matrix, selected_clients):
        """计算四维贡献度"""

        # 1. 数据质量贡献 Q_i^data
        data_quality = self._compute_data_quality_contribution(client_id, data_info)

        # 2. 模型改进贡献 Q_i^model
        model_improvement = self._compute_model_improvement_contribution(
            client_id, local_model, global_model
        )

        # 3. 参与稳定性贡献 Q_i^stable
        stability = self._compute_stability_contribution(client_id)

        # 4. 协作贡献 Q_i^coop
        cooperation = self._compute_cooperation_contribution(
            client_id, compatibility_matrix, selected_clients
        )

        # 使用熵权法计算权重
        contributions = np.array([data_quality, model_improvement, stability, cooperation])
        weights = self._compute_entropy_weights(contributions)

        # 综合贡献度
        total_contribution = np.dot(weights, contributions)

        # 更新历史
        self.contribution_history[client_id].append(total_contribution)

        return {
            'total': total_contribution,
            'data_quality': data_quality,
            'model_improvement': model_improvement,
            'stability': stability,
            'cooperation': cooperation,
            'weights': weights
        }

    def _compute_data_quality_contribution(self, client_id, data_info):
        """计算数据质量贡献 - 基于Wasserstein距离"""
        # 获取客户端数据分布
        client_dist = data_info[client_id]['class_distribution']

        # 计算全局分布（所有客户端的平均）
        all_dists = [info['class_distribution'] for info in data_info.values()]
        global_dist = np.mean(all_dists, axis=0)

        # 计算Wasserstein距离
        w_distance = wasserstein_distance(client_dist, global_dist)

        # 数据量因子
        n_samples = data_info[client_id]['total_samples']
        sample_factor = np.sqrt(n_samples / 1000)  # 归一化

        # 噪声水平（这里用标签比例代替）
        noise_level = 1 - data_info[client_id]['labeled_samples'] / n_samples

        # 综合数据质量贡献
        data_quality = (1 / (1 + w_distance)) * sample_factor * (1 - noise_level * 0.5)

        return data_quality

    def _compute_model_improvement_contribution(self, client_id, local_model, global_model):
        """计算模型改进贡献"""
        # 确保模型在同一设备上进行比较
        device = next(global_model.parameters()).device

        # 如果local_model不在相同设备，将其移动到相同设备
        if next(local_model.parameters()).device != device:
            local_model = local_model.to(device)

        # 计算梯度
        local_params = list(local_model.parameters())
        global_params = list(global_model.parameters())

        # 梯度范数
        grad_norm = 0
        # 梯度余弦相似度
        dot_product = 0
        local_norm = 0
        global_norm = 0

        for l_param, g_param in zip(local_params, global_params):
            # 确保参数在同一设备上
            l_data = l_param.data.to(device)
            g_data = g_param.data.to(device)

            diff = l_data - g_data
            grad_norm += torch.norm(diff).item()

            # 计算余弦相似度
            dot_product += torch.sum(l_data * g_data).item()
            local_norm += torch.norm(l_data).item() ** 2
            global_norm += torch.norm(g_data).item() ** 2

        # 归一化梯度范数
        grad_norm = grad_norm / len(local_params)

        # 余弦相似度
        cos_similarity = dot_product / (np.sqrt(local_norm) * np.sqrt(global_norm) + 1e-8)

        # 模型改进贡献（结合梯度信息）
        model_improvement = 0.5 * (1 / (1 + np.exp(-grad_norm))) + 0.5 * cos_similarity

        return model_improvement

    def _compute_stability_contribution(self, client_id):
        """计算参与稳定性贡献"""
        if len(self.contribution_history[client_id]) == 0:
            return 0.5  # 默认值

        # 历史参与频率
        history = list(self.contribution_history[client_id])
        participation_rate = len(history) / self.history_window

        # 贡献方差（稳定性）
        if len(history) > 1:
            variance = np.var(history)
            stability_factor = 1 / (1 + variance)
        else:
            stability_factor = 0.5

        # 时间衰减因子
        time_weights = np.exp(-0.1 * np.arange(len(history))[::-1])
        if len(history) > 0:
            weighted_avg = np.average(history, weights=time_weights[:len(history)])
        else:
            weighted_avg = 0

        # 综合稳定性贡献
        stability = participation_rate * stability_factor * (0.5 + 0.5 * weighted_avg)

        return stability

    def _compute_cooperation_contribution(self, client_id, compatibility_matrix, selected_clients):
        """计算协作贡献"""
        if len(selected_clients) <= 1:
            return 0.5

        # 计算与其他选中设备的兼容性
        cooperation_score = 0
        count = 0

        for other_id in selected_clients:
            if other_id != client_id:
                # 兼容性得分
                compat_score = compatibility_matrix[client_id][other_id]
                cooperation_score += compat_score
                count += 1

        if count > 0:
            cooperation_score /= count
        else:
            cooperation_score = 0.5

        return cooperation_score

    def _compute_entropy_weights(self, contributions):
        """使用熵权法计算权重"""
        # 归一化
        normalized = contributions / (np.sum(contributions) + 1e-8)

        # 计算熵
        entropy = -normalized * np.log(normalized + 1e-8)

        # 计算权重
        weights = (1 - entropy) / (len(contributions) - np.sum(entropy))

        return weights / np.sum(weights)


# ================== 激励机制实现 ==================

class LayeredIncentiveMechanism:
    """我们提出的分层激励机制"""

    def __init__(self, num_clients, budget_per_round):
        self.num_clients = num_clients
        self.budget_per_round = budget_per_round

        # 预算分配
        self.base_budget_ratio = 0.4
        self.perf_budget_ratio = 0.4
        self.coop_budget_ratio = 0.2

        # 贡献度评估器
        self.contribution_evaluator = ContributionEvaluator()

        # 客户端类型（成本参数）
        self.client_types = np.random.uniform(0.1, 1.0, num_clients)

    def calculate_layered_incentives(self, contributions, compatibility_matrix, selected_clients):
        """计算三层激励"""

        # 第一层：基础激励（合约理论）
        base_payments = self._calculate_contract_incentives(contributions)

        # 第二层：性能激励（Stackelberg博弈）
        perf_payments = self._calculate_stackelberg_incentives(contributions)

        # 第三层：协作激励
        coop_payments = self._calculate_cooperation_incentives(
            contributions, compatibility_matrix, selected_clients
        )

        # 知识共享激励
        knowledge_payments = self._calculate_knowledge_sharing_incentives(
            contributions, compatibility_matrix
        )

        # 综合支付
        total_payments = (base_payments + perf_payments +
                          coop_payments + knowledge_payments)

        # 确保不超预算
        if total_payments.sum() > self.budget_per_round:
            total_payments = total_payments * self.budget_per_round / total_payments.sum()

        return {
            'total_payments': total_payments,
            'base_payments': base_payments,
            'performance_payments': perf_payments,
            'cooperation_payments': coop_payments,
            'knowledge_payments': knowledge_payments
        }

    def _calculate_contract_incentives(self, contributions):
        """计算合约理论激励"""
        budget = self.budget_per_round * self.base_budget_ratio
        payments = np.zeros(self.num_clients)

        # 设计合约菜单
        contracts = self._design_optimal_contracts(contributions)

        for i in range(self.num_clients):
            if contributions[i] > 0:
                # 选择最优合约
                best_contract = self._select_optimal_contract(
                    i, contracts, contributions[i]
                )
                if best_contract:
                    payments[i] = best_contract['payment']

        return payments

    def _design_optimal_contracts(self, contributions):
        """设计最优合约菜单 - 满足IC和IR约束"""
        contracts = []

        # 根据贡献度分布设计5个合约等级
        contribution_levels = np.percentile(contributions[contributions > 0],
                                            [20, 40, 60, 80, 100])

        for level in contribution_levels:
            # 要求的贡献度
            q = level

            # 支付（满足激励相容）
            # r = q^2 / θ（简化的最优合约）
            r = (q ** 2) * self.budget_per_round * self.base_budget_ratio / (5 * np.mean(self.client_types))

            contracts.append({'contribution': q, 'payment': r})

        return contracts

    def _select_optimal_contract(self, client_id, contracts, actual_contribution):
        """选择最优合约"""
        client_type = self.client_types[client_id]
        best_utility = -float('inf')
        best_contract = None

        for contract in contracts:
            # 成本函数 c(q) = θ * q^2 / 2
            cost = client_type * (contract['contribution'] ** 2) / 2

            # 效用函数
            utility = contract['payment'] - cost

            # 个体理性约束
            if utility > 0 and utility > best_utility:
                if actual_contribution >= contract['contribution'] * 0.8:  # 允许20%偏差
                    best_utility = utility
                    best_contract = contract

        return best_contract

    def _calculate_stackelberg_incentives(self, contributions):
        """计算Stackelberg博弈激励"""
        budget = self.budget_per_round * self.perf_budget_ratio
        payments = np.zeros(self.num_clients)

        # 领导者设定激励率
        total_type = np.sum(1 / (self.client_types + 1e-8))
        alpha = budget / (0.5 * total_type)

        for i in range(self.num_clients):
            if contributions[i] > 0:
                # 跟随者的最优努力
                optimal_effort = min(alpha / (2 * self.client_types[i]), 1.0)

                # 基于实际贡献调整支付
                actual_effort = contributions[i]
                payments[i] = alpha * actual_effort * optimal_effort

        # 归一化到预算内
        if payments.sum() > budget:
            payments = payments * budget / payments.sum()

        return payments

    def _calculate_cooperation_incentives(self, contributions, compatibility_matrix, selected_clients):
        """计算协作激励"""
        budget = self.budget_per_round * self.coop_budget_ratio * 0.7
        payments = np.zeros(self.num_clients)

        if len(selected_clients) <= 1:
            return payments

        # 使用谱聚类识别协作组
        groups = self._identify_cooperation_groups(compatibility_matrix, selected_clients)

        for group in groups:
            if len(group) > 1:
                # 计算组内平均贡献
                group_avg_contribution = np.mean([contributions[i] for i in group])

                # 计算组内兼容性
                group_compatibility = np.mean([
                    compatibility_matrix[i, j]
                    for i in group for j in group if i != j
                ])

                # 分配协作奖励
                group_bonus = budget * group_compatibility * group_avg_contribution / len(groups)

                for i in group:
                    payments[i] = group_bonus / len(group)

        return payments

    def _calculate_knowledge_sharing_incentives(self, contributions, compatibility_matrix):
        """计算知识共享激励 - 基于互信息"""
        budget = self.budget_per_round * self.coop_budget_ratio * 0.3
        payments = np.zeros(self.num_clients)

        # 简化：使用兼容性矩阵近似知识转移
        knowledge_transfer = np.zeros(self.num_clients)

        for i in range(self.num_clients):
            if contributions[i] > 0:
                # 计算知识转移量（与其他设备的信息交换）
                transfer_score = np.sum(compatibility_matrix[i] * contributions) / self.num_clients
                knowledge_transfer[i] = transfer_score

        # 分配知识共享奖励
        if knowledge_transfer.sum() > 0:
            payments = budget * knowledge_transfer / knowledge_transfer.sum()

        return payments

    def _identify_cooperation_groups(self, compatibility_matrix, selected_clients):
        """识别协作组"""
        if len(selected_clients) < 2:
            return [selected_clients]

        # 提取子矩阵
        sub_matrix = compatibility_matrix[np.ix_(selected_clients, selected_clients)]

        # 谱聚类
        n_clusters = min(3, len(selected_clients) // 2)
        if n_clusters < 2:
            return [selected_clients]

        try:
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            labels = clustering.fit_predict(sub_matrix)

            groups = []
            for label in range(n_clusters):
                group_indices = np.where(labels == label)[0]
                group = [selected_clients[i] for i in group_indices]
                if len(group) > 0:
                    groups.append(group)

            return groups
        except:
            return [selected_clients]


class BaselineIncentive:
    """基线激励机制 - 简单的贡献度比例分配"""

    def __init__(self, num_clients, budget_per_round):
        self.num_clients = num_clients
        self.budget_per_round = budget_per_round

    def calculate_incentives(self, contributions, selected_clients):
        """基于贡献度比例分配预算"""
        payments = np.zeros(self.num_clients)

        selected_contributions = contributions[selected_clients]
        if selected_contributions.sum() > 0:
            for client_id in selected_clients:
                payments[client_id] = (contributions[client_id] / selected_contributions.sum()) * self.budget_per_round

        return payments


class StackelbergGameIncentive:
    """纯Stackelberg博弈激励机制"""

    def __init__(self, num_clients, budget_per_round):
        self.num_clients = num_clients
        self.budget_per_round = budget_per_round
        self.client_types = np.random.uniform(0.1, 1.0, num_clients)

    def calculate_incentives(self, contributions, selected_clients):
        """Stackelberg博弈激励计算"""
        payments = np.zeros(self.num_clients)

        # 领导者设定激励率
        total_type = np.sum(1 / (self.client_types[selected_clients] + 1e-8))
        alpha = self.budget_per_round / (0.5 * total_type)

        for client_id in selected_clients:
            if contributions[client_id] > 0:
                # 跟随者的最优努力
                optimal_effort = min(alpha / (2 * self.client_types[client_id]), 1.0)

                # 基于实际贡献调整支付
                actual_effort = contributions[client_id]
                payments[client_id] = alpha * actual_effort * optimal_effort

        # 归一化到预算内
        if payments.sum() > self.budget_per_round:
            payments = payments * self.budget_per_round / payments.sum()

        return payments


class IncEFLIncentive:
    """
    IncEFL激励机制实现 - 基于原始论文的凸优化模型
    这个版本整合了复现代码的正确逻辑
    """

    def __init__(self, num_clients, budget_per_round, config):
        self.num_clients = num_clients
        self.budget_per_round = budget_per_round
        self.config = config

        # 初始化所有worker的参数，并按类型theta排序
        # 这是IncEFL论文中的一个关键预处理步骤
        all_workers = []
        for i in range(num_clients):
            all_workers.append({
                'id': i,
                'theta': np.clip(np.random.normal(config.WORKER_THETA_DIST['mean'],
                                                  config.WORKER_THETA_DIST['std']), 0.1, 1.0),
                'f_n': np.random.uniform(config.CPU_FREQ_RANGE[0], config.CPU_FREQ_RANGE[1]),
                's_n': 0.0  # 初始数据大小为0
            })
        # 按 theta 升序排序
        self.all_workers = sorted(all_workers, key=lambda w: w['theta'])
        # 创建一个从 worker id 到其详细信息的映射，方便查找
        self.worker_map = {w['id']: w for w in self.all_workers}

    def _solve_contract_optimization(self, workers_to_optimize):
        """解DCP合规的优化问题 (来自复现代码)"""
        K = len(workers_to_optimize)
        if K == 0:
            return np.zeros(0)

        s = cp.Variable(K, name='data_size', nonneg=True)
        objective_terms = []

        for i in range(K):
            worker = workers_to_optimize[i]
            time_delay = self.config.INCEFL_COMM_TIME + (self.config.INCEFL_CPU_CYCLES_PER_SAMPLE * s[i]) / worker[
                'f_n']
            satisfaction = self.config.INCEFL_SATISFACTION_DELTA * cp.log(
                self.config.INCEFL_MAX_LATENCY - time_delay + 1e-6)
            reward_cost = self.config.INCEFL_REWARD_UNIT_COST_PHI * worker['theta'] * s[i]
            objective_terms.append(satisfaction - reward_cost)

        total_objective = cp.Maximize(cp.sum(objective_terms))

        # 约束条件
        constraints = [s >= 1]  # 最小数据量为1，避免log(negative)
        for i in range(K):
            # 延迟约束
            constraints.append(
                (self.config.INCEFL_CPU_CYCLES_PER_SAMPLE * s[i]) / workers_to_optimize[i]['f_n'] <=
                self.config.INCEFL_MAX_LATENCY - self.config.INCEFL_COMM_TIME - 1
            )

        # 预算约束
        total_reward = cp.sum(
            [self.config.INCEFL_REWARD_UNIT_COST_PHI * workers_to_optimize[i]['theta'] * s[i] for i in range(K)])
        constraints.append(total_reward <= self.budget_per_round)

        problem = cp.Problem(total_objective, constraints)
        try:
            problem.solve(solver=cp.SCS)  # 使用SCS或ECOS求解器
            if s.value is None or problem.status not in ['optimal', 'optimal_inaccurate']:
                # print("IncEFL 优化失败或无解.")
                return np.zeros(K)
            return np.maximum(s.value, 0)
        except Exception as e:
            # print(f"IncEFL 优化求解器异常: {e}")
            return np.zeros(K)

    def _progressive_adjustment_algorithm(self, s_star):
        """确保单调性 (来自复现代码)"""
        if len(s_star) <= 1:
            return s_star

        for _ in range(len(s_star)):  # 迭代确保收敛
            for i in range(len(s_star) - 1):
                if s_star[i] > s_star[i + 1]:
                    # 找到违反单调性的整个区间
                    j = i + 1
                    while j < len(s_star) and s_star[i] > s_star[j]:
                        j += 1
                    avg_value = np.mean(s_star[i:j])
                    s_star[i:j] = avg_value
                    break  # 从头开始重新检查
        return s_star

    def calculate_incentives(self, available_client_ids):
        """
        主计算函数。注意：它不再使用 contributions 参数。
        它会根据传入的本轮可用客户端ID，自己决定选择谁并计算支付。
        """
        payments = np.zeros(self.num_clients)

        # 1. 从本轮可用的客户端中，筛选出我们已经排好序的worker对象
        #    并保持这个列表仍然是按 theta 排序的
        available_workers_sorted = [w for w in self.all_workers if w['id'] in available_client_ids]

        if not available_workers_sorted:
            return payments

        # 2. 解最优化问题
        s_initial = self._solve_contract_optimization(available_workers_sorted)

        # 3. 强制单调性
        s_final = self._progressive_adjustment_algorithm(s_initial.copy())

        # 4. 根据最优数据量 s_final 计算支付
        for i, worker in enumerate(available_workers_sorted):
            optimal_s = s_final[i]
            if optimal_s > 0:
                # 支付金额就是论文中的奖励成本
                payment_amount = self.config.INCEFL_REWARD_UNIT_COST_PHI * worker['theta'] * optimal_s
                # 将计算出的支付金额存到对应 client_id 的位置
                payments[worker['id']] = payment_amount

        # 确保总支付不超过预算（由于求解器精度问题，可能微小超出）
        if np.sum(payments) > self.budget_per_round:
            payments = payments * (self.budget_per_round / np.sum(payments))

        return payments

# ================== 并行训练器 ==================
class ParallelLocalTrainer:
    """并行本地训练器"""

    def __init__(self, data_manager, config):
        self.data_manager = data_manager
        self.config = config
        self.devices = config.DEVICES

    def train_clients_parallel(self, selected_clients, global_model):
        """并行训练多个客户端"""
        if len(self.devices) == 0:
            self.devices = [torch.device('cpu')]

        # 将模型状态字典移到CPU
        global_model_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

        # 准备参数
        args_list = [
            (client_id, global_model_state, i % len(self.devices))
            for i, client_id in enumerate(selected_clients)
        ]

        # 使用线程池并行训练
        with ThreadPoolExecutor(max_workers=self.config.PARALLEL_CLIENTS) as executor:
            trained_model_states = list(executor.map(self.train_single_client, args_list))

        # 加载训练好的模型 - 返回CPU上的模型
        trained_models = []
        for state_dict in trained_model_states:
            if self.config.DATASET == 'CIFAR10':
                model = CNNModel(num_classes=self.config.NUM_CLASSES)
            else:
                model = FMNISTModel(num_classes=self.config.NUM_CLASSES)

            model.load_state_dict(state_dict)
            # 确保模型在CPU上，便于后续操作
            model = model.cpu()
            trained_models.append(model)

        return trained_models

    def train_single_client(self, args):
        """单个客户端训练"""
        client_id, global_model_state, device_id = args

        # 选择设备
        if len(self.devices) > 1:
            device = self.devices[device_id]
        else:
            device = self.devices[0]

        # 创建本地模型
        if self.config.DATASET == 'CIFAR10':
            local_model = CNNModel(num_classes=self.config.NUM_CLASSES)
        else:
            local_model = FMNISTModel(num_classes=self.config.NUM_CLASSES)

        local_model.load_state_dict(global_model_state)
        local_model = local_model.to(device)
        local_model.train()

        # 获取数据加载器
        dataloader = self.data_manager.get_client_dataloader(
            client_id, self.config.LOCAL_BATCH_SIZE
        )
        optimizer = optim.SGD(local_model.parameters(), lr=self.config.LOCAL_LR, momentum=0.9)

        # 混合精度训练
        if self.config.USE_AMP:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.config.LOCAL_EPOCHS):
            for batch_data, batch_labels, is_labeled in dataloader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                is_labeled = is_labeled.to(device)

                optimizer.zero_grad()

                if self.config.USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = local_model(batch_data)
                        loss = self._compute_loss(outputs, batch_labels, is_labeled)

                    if loss is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    outputs = local_model(batch_data)
                    loss = self._compute_loss(outputs, batch_labels, is_labeled)

                    if loss is not None:
                        loss.backward()
                        optimizer.step()

        # 返回CPU上的状态字典
        local_model = local_model.cpu()
        return local_model.state_dict()

    def _compute_loss(self, outputs, batch_labels, is_labeled):
        """计算损失函数"""
        if is_labeled.sum() == 0:
            return None

        labeled_outputs = outputs[is_labeled]
        labeled_targets = batch_labels[is_labeled]

        valid_mask = labeled_targets >= 0
        if valid_mask.sum() == 0:
            return None

        valid_outputs = labeled_outputs[valid_mask]
        valid_targets = labeled_targets[valid_mask]

        loss = F.cross_entropy(valid_outputs, valid_targets)

        # 对无标签数据使用熵最小化
        if (~is_labeled).sum() > 0:
            unlabeled_outputs = outputs[~is_labeled]
            probs = F.softmax(unlabeled_outputs, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            loss = loss + 0.1 * entropy

        return loss


# ================== 实验运行函数 ==================
def run_chapter2_experiment():
    """运行第二章实验 - 对比4种激励机制"""
    print("=" * 60)
    print("第二章：分层激励机制对比实验")
    print("对比方法: Layered (Ours), Baseline, Stackelberg, IncEFL")
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

    # 初始化各激励机制
    incentive_mechanisms = {
        'Layered (Ours)': LayeredIncentiveMechanism(config.NUM_CLIENTS, config.BUDGET_PER_ROUND),
        'Baseline': BaselineIncentive(config.NUM_CLIENTS, config.BUDGET_PER_ROUND),
        'Stackelberg': StackelbergGameIncentive(config.NUM_CLIENTS, config.BUDGET_PER_ROUND),
        'IncEFL': IncEFLIncentive(config.NUM_CLIENTS, config.BUDGET_PER_ROUND, config)
    }

    # 初始化贡献度评估器
    contribution_evaluator = ContributionEvaluator()

    # 初始化兼容性矩阵
    compatibility_matrix = np.random.uniform(0.3, 1.0, (config.NUM_CLIENTS, config.NUM_CLIENTS))
    compatibility_matrix = (compatibility_matrix + compatibility_matrix.T) / 2
    np.fill_diagonal(compatibility_matrix, 1.0)

    # 结果存储
    results = {
        method: {
            'accuracy': [],
            'participation': [],
            'total_payment': [],
            'total_contribution': [],
            'fairness': [],
            'time': []
        }
        for method in incentive_mechanisms.keys()
    }

    # 初始化全局模型
    global_models = {
        method: model_class(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        for method in incentive_mechanisms.keys()
    }

    # 训练循环
    for round_idx in range(config.NUM_ROUNDS):
        print(f"\nRound {round_idx + 1}/{config.NUM_ROUNDS}")

        # 随机选择客户端参与
        num_participants = np.random.randint(3, min(config.CLIENTS_PER_ROUND + 2, config.NUM_CLIENTS))
        selected_clients = np.random.choice(config.NUM_CLIENTS, num_participants, replace=False)

        # 获取数据分布信息
        data_distributions = data_manager.get_client_data_distribution()

        for method_name, incentive_mechanism in incentive_mechanisms.items():
            print(f"\nTraining with {method_name}...")
            start_time = time.time()

            # 并行训练选中的客户端
            local_models = parallel_trainer.train_clients_parallel(
                selected_clients, global_models[method_name]
            )

            # 评估贡献度
            contributions = np.zeros(config.NUM_CLIENTS)
            data_info = {}

            for i, client_id in enumerate(selected_clients):
                # 获取数据信息
                dist_info = data_distributions[client_id]
                data_info[client_id] = {
                    'class_distribution': dist_info['class_distribution'],
                    'total_samples': dist_info['total_samples'],
                    'labeled_samples': dist_info['labeled_samples'],
                    'entropy': dist_info['entropy']
                }

                # 评估贡献度
                contribution_result = contribution_evaluator.evaluate_comprehensive_contribution(
                    client_id, local_models[i].cpu(), global_models[method_name].cpu(),
                    data_info, compatibility_matrix, selected_clients
                )
                contributions[client_id] = contribution_result['total']

            # 计算激励
            if method_name == 'Layered (Ours)':
                incentive_result = incentive_mechanism.calculate_layered_incentives(
                    contributions, compatibility_matrix, selected_clients
                )
                payments = incentive_result['total_payments']
            elif method_name == 'Baseline':
                payments = incentive_mechanism.calculate_incentives(contributions, selected_clients)
            elif method_name == 'Stackelberg':
                payments = incentive_mechanism.calculate_incentives(contributions, selected_clients)
            else:  # IncEFL
                payments = incentive_mechanism.calculate_incentives(selected_clients)

            # 聚合模型
            if len(local_models) > 0:
                aggregated_state = OrderedDict()
                # 确保权重是numpy数组
                contributions_np = np.array(contributions)
                selected_contributions = contributions_np[selected_clients]

                # 检查总贡献是否为零，避免除零错误
                if selected_contributions.sum() > 1e-8:
                    weights = selected_contributions / selected_contributions.sum()
                else:
                    # 如果贡献都为0，则使用均等权重
                    weights = np.ones(len(selected_clients)) / len(selected_clients)

                # 以第一个本地模型的状态字典作为参考
                first_model_state = local_models[0].state_dict()

                for key in first_model_state.keys():
                    # 检查当前参数是否为浮点类型
                    if first_model_state[key].is_floating_point():
                        # ✅ 如果是浮点数（如权重、偏置、BN的running_mean/var），进行加权平均

                        # 从所有本地模型中收集这个key对应的参数
                        params = [model.state_dict()[key].cpu() for model in local_models]

                        # 执行加权求和
                        weighted_sum = torch.zeros_like(params[0])
                        for i in range(len(params)):
                            weighted_sum += weights[i] * params[i]
                        aggregated_state[key] = weighted_sum
                    else:
                        # ❌ 如果不是浮点数（如BN的num_batches_tracked），直接复制第一个模型的值
                        # 因为对这个整数计数器求平均没有意义，而且会引发错误
                        aggregated_state[key] = first_model_state[key].cpu().clone()

                # 加载聚合后的状态到全局模型
                global_models[method_name].load_state_dict(aggregated_state)
                global_models[method_name] = global_models[method_name].to(config.DEVICE)

            # 评估性能
            global_models[method_name].eval()
            test_loader = data_manager.get_test_dataloader()
            correct = 0
            total = 0

            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(config.DEVICE), labels.to(config.DEVICE)
                    outputs = global_models[method_name](data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total

            # 计算公平性（Gini系数）
            if payments[payments > 0].sum() > 0:
                sorted_payments = np.sort(payments[payments > 0])
                n = len(sorted_payments)
                cumsum = np.cumsum(sorted_payments)
                gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_payments)) / (n * cumsum[-1]) - (n + 1) / n
            else:
                gini = 0

            # 记录结果
            elapsed_time = time.time() - start_time
            results[method_name]['accuracy'].append(accuracy)
            results[method_name]['participation'].append(len(selected_clients))
            results[method_name]['total_payment'].append(payments.sum())
            results[method_name]['total_contribution'].append(contributions.sum())
            results[method_name]['fairness'].append(gini)
            results[method_name]['time'].append(elapsed_time)

            print(f"  {method_name}: Acc={accuracy:.4f}, Payment={payments.sum():.2f}, Time={elapsed_time:.2f}s")

    return results


# ================== 可视化函数 ==================
def plot_chapter2_results(results, save_path='chapter2_results.png'):
    """绘制第二章结果对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 设置颜色
    colors = {
        'Layered (Ours)': 'red',
        'Baseline': 'blue',
        'Stackelberg': 'green',
        'IncEFL': 'orange'
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

    # 图2：参与率
    ax = axes[0, 1]
    for method, data in results.items():
        if len(data['participation']) > 0:
            ax.plot(data['participation'], label=method, linewidth=2, color=colors[method])
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Number of Participants')
    ax.set_title('Client Participation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图3：总支付
    ax = axes[0, 2]
    for method, data in results.items():
        if len(data['total_payment']) > 0:
            ax.plot(data['total_payment'], label=method, linewidth=2, color=colors[method])
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Total Payment')
    ax.set_title('Payment Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图4：总贡献
    ax = axes[1, 0]
    for method, data in results.items():
        if len(data['total_contribution']) > 0:
            smoothed = pd.Series(data['total_contribution']).rolling(window=5, min_periods=1).mean()
            ax.plot(smoothed, label=method, linewidth=2, color=colors[method])
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Total Contribution')
    ax.set_title('Contribution Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图5：公平性
    ax = axes[1, 1]
    for method, data in results.items():
        if len(data['fairness']) > 0:
            smoothed = pd.Series(data['fairness']).rolling(window=5, min_periods=1).mean()
            ax.plot(smoothed, label=method, linewidth=2, color=colors[method])
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Payment Fairness (Lower is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图6：效率对比
    ax = axes[1, 2]
    efficiency_data = {}
    for method, data in results.items():
        efficiency = []
        for i in range(len(data['total_contribution'])):
            if data['total_payment'][i] > 0:
                eff = data['total_contribution'][i] / data['total_payment'][i]
            else:
                eff = 0
            efficiency.append(eff)

        if len(efficiency) > 0:
            efficiency_data[method] = np.mean(efficiency[-10:])  # 最后10轮的平均效率

    if len(efficiency_data) > 0:
        bars = ax.bar(efficiency_data.keys(), efficiency_data.values(),
                      color=[colors[m] for m in efficiency_data.keys()])
        ax.set_ylabel('Efficiency (Contribution/Payment)')
        ax.set_title('Incentive Efficiency')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom')

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
            avg_payment = np.mean(results[method]['total_payment'])
            avg_fairness = np.mean(results[method]['fairness'])
            avg_time = np.mean(results[method]['time'])
            print(f"{method}:")
            print(f"  最终准确率: {final_acc:.4f}")
            print(f"  平均支付: {avg_payment:.2f}")
            print(f"  平均Gini系数: {avg_fairness:.4f}")
            print(f"  平均训练时间: {avg_time:.2f}s")


def save_results(results, filename='chapter2_results.json'):
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
    print("联邦学习激励机制实验 - 第二章改进版")
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
        print("\n开始运行第二章实验...")
        print("包含4个对比方法: Layered (Ours), Baseline, Stackelberg, IncEFL")

        # 运行实验
        results = run_chapter2_experiment()

        # 保存结果
        save_results(results, 'chapter2_results.json')

        # 绘制结果图表
        plot_chapter2_results(results, 'chapter2_results.png')

        print("\n" + "=" * 80)
        print("第二章实验完成！")
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