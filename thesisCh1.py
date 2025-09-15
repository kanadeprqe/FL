#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一章：兼容性感知的DQN设备选择 - 改进版
添加了3个对比方法：FedAvg, SelfFed, FedProx
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
            self.NUM_WORKERS = 4  # 适度的worker数量
            # GPU内存管理
            torch.cuda.empty_cache()
        else:
            self.DEVICE = torch.device('cpu')
            self.NUM_GPUS = 0
            self.DEVICES = [self.DEVICE]
            self.NUM_WORKERS = 0
        self.DATA_PATH = ''

        # 数据集配置
        self.DATASET = 'CIFAR10'
        self.NUM_CLASSES = 10
        self.NUM_CLIENTS = 20
        self.CLIENTS_PER_ROUND = 5
        self.NUM_ROUNDS = 50
        self.LOCAL_EPOCHS = 5
        self.LOCAL_BATCH_SIZE = 32
        self.LOCAL_LR = 0.01
        self.NON_IID_ALPHA = 0.5  # Dirichlet分布参数
        self.LABEL_RATIO = 0.6  # 每个客户端的标签数据比例
        self.TOTAL_BUDGET = 1000
        self.BUDGET_PER_ROUND = self.TOTAL_BUDGET / self.NUM_ROUNDS

        # 并行训练配置
        self.PARALLEL_CLIENTS = min(5, self.NUM_GPUS) if self.NUM_GPUS > 0 else 1

        # 混合精度训练
        self.USE_AMP = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

        # FedProx参数
        self.FEDPROX_MU = 0.01  # FedProx正则化参数

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


# ================== DQN相关类 ==================
class CompatibilityAwareDQN(nn.Module):
    """兼容性感知的DQN网络"""

    def __init__(self, state_dim, num_clients, hidden_dim=256):
        super().__init__()
        self.num_clients = num_clients

        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 兼容性注意力机制
        self.compatibility_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Q值输出层
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_clients)
        )

    def forward(self, state, compatibility_matrix=None):
        # 提取特征
        features = self.feature_net(state)

        # 兼容性感知处理
        if compatibility_matrix is not None:
            # 使用兼容性注意力
            features_expanded = features.unsqueeze(0)
            attn_output, _ = self.compatibility_attention(
                features_expanded,
                features_expanded,
                features_expanded
            )
            attn_features = attn_output.squeeze(0).squeeze(0)

            # 特征融合
            combined_features = torch.cat([features, attn_features], dim=-1)
            features = self.fusion_layer(combined_features)

        # 计算Q值
        q_values = self.q_network(features)
        return q_values


class DQNAgent:
    """改进的DQN智能体"""

    def __init__(self, state_dim, num_clients, lr=1e-4, gamma=0.99, epsilon=1.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_clients = num_clients
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Q网络和目标网络
        self.q_network = CompatibilityAwareDQN(state_dim, num_clients).to(self.device)
        self.target_network = CompatibilityAwareDQN(state_dim, num_clients).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 经验回放池
        self.memory = deque(maxlen=10000)

    def select_action(self, state, compatibility_matrix, valid_clients, k=5):
        """选择动作 - 考虑兼容性约束"""
        if np.random.random() <= self.epsilon:
            # 探索：随机选择
            selected = np.random.choice(valid_clients, min(k, len(valid_clients)), replace=False)
            return selected.tolist()

        # 利用：根据Q值和兼容性选择
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor, compatibility_matrix).squeeze(0)

        # 计算考虑兼容性的综合得分
        q_values_np = q_values.cpu().numpy()

        # 选择兼容性最好的top-k组合
        selected = self._select_compatible_group(
            q_values_np, compatibility_matrix, valid_clients, k
        )

        return selected

    def _select_compatible_group(self, q_values, compatibility_matrix, valid_clients, k):
        """选择兼容性最好的设备组"""
        # 贪心选择
        selected = []
        remaining = list(valid_clients)

        for _ in range(min(k, len(valid_clients))):
            best_device = None
            best_marginal_score = -float('inf')

            for device in remaining:
                # 计算边际得分（Q值 + 兼容性奖励）
                marginal_score = q_values[device]

                # 加入兼容性得分
                if len(selected) > 0 and compatibility_matrix is not None:
                    compat_score = np.mean([compatibility_matrix[device][s] for s in selected])
                    marginal_score += 0.3 * compat_score

                if marginal_score > best_marginal_score:
                    best_marginal_score = marginal_score
                    best_device = device

            if best_device is not None:
                selected.append(best_device)
                remaining.remove(best_device)

        return selected

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        """训练网络"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = [e[1] for e in batch]
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = 0
        for i in range(batch_size):
            selected_clients = actions[i]
            if len(selected_clients) > 0:
                for client_id in selected_clients:
                    if client_id < self.num_clients:
                        predicted_q = current_q_values[i, client_id]
                        loss += F.mse_loss(predicted_q, target_q_values[i])

        loss = loss / batch_size

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())


# ================== 设备选择环境 ==================
class DeviceSelectionEnvironment:
    """改进的设备选择环境"""

    def __init__(self, data_manager, num_clients=20):
        self.data_manager = data_manager
        self.num_clients = num_clients

        # 初始化设备异构性
        self.device_profiles = self._init_device_profiles()

        # 初始化兼容性矩阵
        self.compatibility_matrix = self._init_compatibility_matrix()

        # 设备疲劳度
        self.fatigue_levels = np.zeros(num_clients)
        self.fatigue_recovery_factor = 0.9

        # 数据分布信息
        self.data_distributions = data_manager.get_client_data_distribution()

        # 历史信息
        self.participation_history = np.zeros((num_clients, 10))
        self.contribution_history = np.zeros((num_clients, 10))

    def _init_device_profiles(self):
        """初始化设备配置"""
        profiles = []
        for i in range(self.num_clients):
            profiles.append({
                'compute_power': np.random.uniform(0.3, 1.0),
                'memory': np.random.uniform(2, 8),
                'energy_budget': np.random.uniform(100, 500),
                'battery': np.random.uniform(0.5, 1.0),
                'bandwidth': np.random.uniform(10, 100),
                'latency': np.random.uniform(10, 100),
                'packet_loss': np.random.uniform(0, 0.1),
                'comm_protocol': np.random.choice(['WiFi', '4G', '5G']),
                'reliability': np.random.uniform(0.7, 1.0)
            })
        return profiles

    def _init_compatibility_matrix(self):
        """初始化兼容性矩阵"""
        matrix = np.zeros((self.num_clients, self.num_clients))

        for i in range(self.num_clients):
            for j in range(self.num_clients):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    # 协议兼容性
                    proto_compat = 1.0 if self.device_profiles[i]['comm_protocol'] == \
                                          self.device_profiles[j]['comm_protocol'] else 0.5

                    # 数据格式兼容性
                    format_compat = np.random.uniform(0.5, 1.0)

                    # 同步能力匹配度
                    sync_compat = 1.0 - abs(self.device_profiles[i]['latency'] -
                                            self.device_profiles[j]['latency']) / 100

                    # 综合兼容性
                    matrix[i, j] = 0.4 * proto_compat + 0.3 * format_compat + 0.3 * sync_compat
                    matrix[j, i] = matrix[i, j]

        return matrix

    def get_state(self):
        """获取当前状态"""
        state = []

        for i in range(self.num_clients):
            profile = self.device_profiles[i]

            # 计算异构性得分
            comp_score = (0.4 * profile['compute_power'] +
                          0.3 * profile['memory'] / 8 +
                          0.3 * profile['energy_budget'] * profile['battery'] / 500)

            # 设备特征向量
            state.extend([
                comp_score,
                profile['compute_power'],
                profile['battery'],
                profile['bandwidth'] / 100,
                profile['latency'] / 100,
                profile['reliability'],
                self.fatigue_levels[i]
            ])

            # 数据异构性特征
            dist_info = self.data_distributions[i]
            state.extend([
                dist_info['total_samples'] / 1000,
                dist_info['labeled_samples'] / (dist_info['total_samples'] + 1e-6),
                dist_info['entropy'] / (np.log(10) + 1e-6)
            ])

            # 历史参与信息
            state.append(np.mean(self.participation_history[i]))
            state.append(np.mean(self.contribution_history[i]))

        return np.array(state, dtype=np.float32)

    def get_valid_clients(self):
        """获取可用客户端"""
        valid = []
        for i in range(self.num_clients):
            effective_power = self.device_profiles[i]['compute_power'] * (1 - self.fatigue_levels[i])

            if (self.device_profiles[i]['battery'] > 0.2 and
                    self.fatigue_levels[i] < 0.8 and
                    effective_power > 0.1):
                valid.append(i)
        return valid

    def step(self, selected_clients, global_model):
        """执行一步环境交互"""
        trained_models = []
        training_times = []
        communication_costs = []

        for client_id in selected_clients:
            start_time = time.time()

            # 本地训练
            local_model = self._local_training(client_id, global_model)
            trained_models.append(local_model)

            training_time = time.time() - start_time
            training_times.append(training_time)

            # 计算通信成本
            comm_cost = self._calculate_communication_cost(client_id)
            communication_costs.append(comm_cost)

        # 聚合模型
        if len(trained_models) > 0:
            aggregated_model = self._fedavg_aggregation(trained_models)
        else:
            aggregated_model = global_model

        # 评估性能
        accuracy, loss = self._evaluate_model(aggregated_model)

        # 计算完整奖励
        reward = self._calculate_comprehensive_reward(
            selected_clients, accuracy, training_times, communication_costs
        )

        # 更新疲劳度和历史信息
        self._update_fatigue(selected_clients)
        self._update_history(selected_clients, accuracy)

        # 获取下一状态
        next_state = self.get_state()

        return aggregated_model, next_state, reward, accuracy, loss

    def _calculate_comprehensive_reward(self, selected_clients, accuracy, training_times, comm_costs):
        """计算完整奖励"""
        if len(selected_clients) == 0:
            return 0

        # 1. 精度奖励
        accuracy_reward = accuracy

        # 2. 效率奖励
        if len(training_times) > 0:
            avg_time = np.mean(training_times)
            target_time = 10
            efficiency_reward = np.exp(-avg_time / target_time)
        else:
            efficiency_reward = 0

        # 3. 兼容性奖励
        compatibility_reward = 0
        if len(selected_clients) > 1:
            for i in range(len(selected_clients)):
                for j in range(i + 1, len(selected_clients)):
                    compatibility_reward += self.compatibility_matrix[
                        selected_clients[i], selected_clients[j]
                    ]
            compatibility_reward /= (len(selected_clients) * (len(selected_clients) - 1) / 2)

        # 4. 成本惩罚
        cost_penalty = 0
        if len(selected_clients) > 0 and len(comm_costs) > 0:
            energy_cost = np.mean([
                self.device_profiles[i]['energy_budget'] / 500
                for i in selected_clients
            ])
            comm_cost = np.mean(comm_costs)
            cost_penalty = 0.5 * energy_cost + 0.5 * comm_cost

        # 综合奖励
        reward = (0.4 * accuracy_reward +
                  0.25 * efficiency_reward +
                  0.25 * compatibility_reward -
                  0.1 * cost_penalty)

        return reward

    def _calculate_communication_cost(self, client_id):
        """计算通信成本"""
        profile = self.device_profiles[client_id]
        cost = (1 / profile['bandwidth']) * (1 + profile['latency'] / 100) * (1 + profile['packet_loss'])
        return cost

    def _local_training(self, client_id, global_model):
        """本地训练"""
        config = Config()
        local_model = copy.deepcopy(global_model)
        local_model.train()

        dataloader = self.data_manager.get_client_dataloader(client_id, config.LOCAL_BATCH_SIZE)
        optimizer = optim.SGD(local_model.parameters(), lr=config.LOCAL_LR, momentum=0.9)

        for epoch in range(config.LOCAL_EPOCHS):
            for batch_data, batch_labels, is_labeled in dataloader:
                batch_data = batch_data.to(config.DEVICE)
                batch_labels = batch_labels.to(config.DEVICE)
                is_labeled = is_labeled.to(config.DEVICE)

                optimizer.zero_grad()
                outputs = local_model(batch_data)

                if is_labeled.sum() > 0:
                    labeled_outputs = outputs[is_labeled]
                    labeled_targets = batch_labels[is_labeled]

                    valid_mask = labeled_targets >= 0
                    if valid_mask.sum() > 0:
                        loss = F.cross_entropy(
                            labeled_outputs[valid_mask],
                            labeled_targets[valid_mask]
                        )

                        # 对无标签数据使用熵最小化
                        if (~is_labeled).sum() > 0:
                            unlabeled_outputs = outputs[~is_labeled]
                            probs = F.softmax(unlabeled_outputs, dim=1)
                            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                            loss = loss + 0.1 * entropy

                        loss.backward()
                        optimizer.step()

        return local_model

    def _fedavg_aggregation(self, models):
        """FedAvg聚合 - 修复版"""
        if not models:
            return None

        aggregated_state = OrderedDict()
        # Use the state_dict from the first model as a reference
        first_model_state = models[0].state_dict()

        for key in first_model_state.keys():
            # Check if the parameter is a floating-point type
            if first_model_state[key].is_floating_point():
                # ✅ If it's a float, average it across all models
                # We need to ensure all tensors are on the CPU for stacking
                params = [model.state_dict()[key].cpu() for model in models]
                aggregated_state[key] = torch.stack(params).mean(dim=0)
            else:
                # ❌ If it's not a float (like num_batches_tracked), just copy the value
                aggregated_state[key] = first_model_state[key].cpu().clone()

        config = Config()
        if config.DATASET == 'CIFAR10':
            aggregated_model = CNNModel(num_classes=config.NUM_CLASSES)

        aggregated_model.load_state_dict(aggregated_state)
        return aggregated_model

    def _evaluate_model(self, model):
        """评估模型性能"""
        config = Config()
        # 确保模型在正确的设备上
        model = model.to(config.DEVICE)
        model.eval()
        test_loader = self.data_manager.get_test_dataloader()

        correct = 0
        total = 0
        total_loss = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(data)
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)

        return accuracy, avg_loss

    def _update_fatigue(self, selected_clients):
        """更新疲劳度"""
        for i in range(self.num_clients):
            if i in selected_clients:
                self.fatigue_levels[i] += 0.1
            else:
                self.fatigue_levels[i] *= self.fatigue_recovery_factor

            self.fatigue_levels[i] = min(1.0, max(0.0, self.fatigue_levels[i]))

    def _update_history(self, selected_clients, accuracy):
        """更新历史信息"""
        self.participation_history = np.roll(self.participation_history, -1, axis=1)
        self.participation_history[:, -1] = 0
        for client_id in selected_clients:
            self.participation_history[client_id, -1] = 1

        self.contribution_history = np.roll(self.contribution_history, -1, axis=1)
        self.contribution_history[:, -1] = 0
        for client_id in selected_clients:
            self.contribution_history[client_id, -1] = accuracy


# ================== 对比方法实现 ==================

class SelfFedAgent:
    """SelfFed方法实现 - 基于论文的DQN设备选择"""

    def __init__(self, num_clients, state_size):
        self.num_clients = num_clients
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 简化的DQN网络（根据SelfFed论文）
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_clients)
        ).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=2000)
        self.epsilon = 0.3

    def select_clients(self, state, num_select=5):
        """根据SelfFed策略选择客户端"""
        if random.random() < self.epsilon:
            return np.random.choice(self.num_clients, num_select, replace=False).tolist()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze()
                top_indices = torch.topk(q_values, num_select).indices
                return top_indices.cpu().numpy().tolist()


class FedProxTrainer:
    """FedProx训练器 - 添加近端项正则化"""

    def __init__(self, mu=0.01):
        self.mu = mu

    def train_client(self, client_id, global_model, data_manager, config):
        """FedProx本地训练"""
        # 关键修正 1: 从传入的模型中获取正确的设备
        device = next(global_model.parameters()).device

        local_model = copy.deepcopy(global_model)
        local_model.train()

        dataloader = data_manager.get_client_dataloader(client_id, config.LOCAL_BATCH_SIZE)
        optimizer = optim.SGD(local_model.parameters(), lr=config.LOCAL_LR, momentum=0.9)

        global_params = {name: param.clone() for name, param in global_model.named_parameters()}

        for epoch in range(config.LOCAL_EPOCHS):
            for batch_data, batch_labels, is_labeled in dataloader:
                # 关键修正 2: 使用正确的设备，而不是全局的 config.DEVICE
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                is_labeled = is_labeled.to(device)

                optimizer.zero_grad()
                outputs = local_model(batch_data)

                loss = 0
                if is_labeled.sum() > 0:
                    labeled_outputs = outputs[is_labeled]
                    labeled_targets = batch_labels[is_labeled]
                    valid_mask = labeled_targets >= 0
                    if valid_mask.sum() > 0:
                        loss = F.cross_entropy(
                            labeled_outputs[valid_mask],
                            labeled_targets[valid_mask]
                        )
                        # 添加FedProx近端项
                        proximal_term = 0
                        for name, param in local_model.named_parameters():
                            proximal_term += ((param - global_params[name]) ** 2).sum()
                        loss += (self.mu / 2) * proximal_term
                        loss.backward()
                        optimizer.step()
        return local_model

# ================== 并行训练器 ==================
class ParallelLocalTrainer:
    """并行本地训练器"""

    def __init__(self, data_manager, config):
        self.data_manager = data_manager
        self.config = config
        self.devices = config.DEVICES

    def train_clients_parallel(self, selected_clients, global_model, method='standard'):
        """并行训练多个客户端"""
        if len(self.devices) == 0:
            self.devices = [torch.device('cpu')]

        # 将模型状态字典移到CPU
        global_model_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

        # 准备参数
        args_list = [
            (client_id, global_model_state, i % len(self.devices), method)
            for i, client_id in enumerate(selected_clients)
        ]

        # 使用线程池并行训练
        with ThreadPoolExecutor(max_workers=self.config.PARALLEL_CLIENTS) as executor:
            trained_model_states = list(executor.map(self.train_single_client, args_list))

        # 加载训练好的模型
        trained_models = []
        for state_dict in trained_model_states:
            if self.config.DATASET == 'CIFAR10':
                model = CNNModel(num_classes=self.config.NUM_CLASSES)

            model.load_state_dict(state_dict)
            trained_models.append(model)

        return trained_models

    def train_single_client(self, args):
        """单个客户端训练"""
        client_id, global_model_state, device_id, method = args

        # 选择设备
        if len(self.devices) > 1:
            device = self.devices[device_id]
        else:
            device = self.devices[0]

        # 创建本地模型
        if self.config.DATASET == 'CIFAR10':
            local_model = CNNModel(num_classes=self.config.NUM_CLASSES)

        local_model.load_state_dict(global_model_state)
        local_model = local_model.to(device)
        local_model.train()

        # 根据方法选择训练方式
        if method == 'fedprox':
            fedprox_trainer = FedProxTrainer(mu=self.config.FEDPROX_MU)
            local_model = fedprox_trainer.train_client(
                client_id, local_model, self.data_manager, self.config
            )
        else:
            # 标准训练
            dataloader = self.data_manager.get_client_dataloader(
                client_id, self.config.LOCAL_BATCH_SIZE
            )
            optimizer = optim.SGD(local_model.parameters(), lr=self.config.LOCAL_LR, momentum=0.9)

            for epoch in range(self.config.LOCAL_EPOCHS):
                for batch_data, batch_labels, is_labeled in dataloader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    is_labeled = is_labeled.to(device)

                    optimizer.zero_grad()
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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import warnings

warnings.filterwarnings('ignore')

# ================== 实验运行函数 ==================
def run_chapter1_experiment():
    """运行第一章实验 - 包含4个对比方法"""
    print("=" * 60)
    print("第一章：兼容性感知的DQN设备选择实验")
    print("对比方法: DQN-Compatibility (Ours), FedAvg, SelfFed, FedProx")
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

    # 初始化环境
    env = DeviceSelectionEnvironment(data_manager, config.NUM_CLIENTS)

    # 初始化并行训练器
    parallel_trainer = ParallelLocalTrainer(data_manager, config)

    # 状态维度
    state_dim = config.NUM_CLIENTS * 14  # 注意：特征维度从12变为14

    # 初始化各方法的智能体
    dqn_agent = DQNAgent(state_dim, config.NUM_CLIENTS)
    selffed_agent = SelfFedAgent(config.NUM_CLIENTS, state_dim // 2)  # SelfFed

    # 初始化结果存储
    results = {
        'DQN-Compatibility (Ours)': {'accuracy': [], 'loss': [], 'time': [], 'selected_clients': []},
        'FedAvg': {'accuracy': [], 'loss': [], 'time': [], 'selected_clients': []},
        'SelfFed': {'accuracy': [], 'loss': [], 'time': [], 'selected_clients': []},
        'FedProx': {'accuracy': [], 'loss': [], 'time': [], 'selected_clients': []}
    }

    # 初始化全局模型
    if config.DATASET == 'CIFAR10':
        model_class = CNNModel


    global_models = {
        method: model_class(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        for method in results.keys()
    }

    # 训练循环
    for round_idx in range(config.NUM_ROUNDS):
        print(f"\nRound {round_idx + 1}/{config.NUM_ROUNDS}")

        # 获取当前状态
        state = env.get_state()
        valid_clients = env.get_valid_clients()

        if len(valid_clients) < config.CLIENTS_PER_ROUND:
            print(f"Warning: Only {len(valid_clients)} valid clients available")
            if len(valid_clients) == 0:
                continue

        # 1. DQN-Compatibility (我们的方法)
        print("Training DQN-Compatibility...")
        start_time = time.time()
        dqn_selected = dqn_agent.select_action(
            state,
            env.compatibility_matrix,
            valid_clients,
            k=config.CLIENTS_PER_ROUND,
            env=env  # 传入环境对象以使用数据质量信息
        )

        # 确保选择的客户端有效
        dqn_selected = [c for c in dqn_selected if c in valid_clients]
        if len(dqn_selected) == 0:
            dqn_selected = valid_clients[:min(config.CLIENTS_PER_ROUND, len(valid_clients))]

        # 并行训练
        trained_models = parallel_trainer.train_clients_parallel(
            dqn_selected, global_models['DQN-Compatibility (Ours)']
        )

        # 聚合和评估
        # +++ 这是新的、正确的代码 +++
        if len(trained_models) > 0:
            aggregated_state = OrderedDict()
            first_model_state = trained_models[0].state_dict()  # 以第一个模型为参考

            for key in first_model_state.keys():
                # 关键修改：检查参数是否为浮点数
                if first_model_state[key].is_floating_point():
                    # 如果是，正常求平均
                    params = [m.state_dict()[key].cpu() for m in trained_models]
                    aggregated_state[key] = torch.stack(params).mean(dim=0)
                else:
                    # 如果不是（例如num_batches_tracked），直接复制第一个的值
                    aggregated_state[key] = first_model_state[key].clone()

            aggregated_model = model_class(num_classes=config.NUM_CLASSES)
            # ... 后续代码不变
            aggregated_model.load_state_dict(aggregated_state)
            aggregated_model = aggregated_model.to(config.DEVICE)
            global_models['DQN-Compatibility (Ours)'] = aggregated_model
            accuracy, loss = env._evaluate_model(aggregated_model)
        else:
            accuracy, loss = 0, float('inf')

        # 计算奖励并更新DQN
        next_state = env.get_state()
        reward = env._calculate_comprehensive_reward(
            dqn_selected, accuracy, [time.time() - start_time], []
        )
        done = round_idx == config.NUM_ROUNDS - 1
        dqn_agent.store_transition(state, dqn_selected, reward, next_state, done)

        if round_idx > 10:
            dqn_agent.train()

        if round_idx % 10 == 0:
            dqn_agent.update_target_network()

        elapsed_time = time.time() - start_time
        results['DQN-Compatibility (Ours)']['accuracy'].append(accuracy)
        results['DQN-Compatibility (Ours)']['loss'].append(loss)
        results['DQN-Compatibility (Ours)']['time'].append(elapsed_time)
        results['DQN-Compatibility (Ours)']['selected_clients'].append(dqn_selected)
        print(f"  DQN-Compatibility: Acc={accuracy:.4f}, Loss={loss:.4f}, Time={elapsed_time:.2f}s")

        # 2. FedAvg (标准联邦平均)
        print("Training FedAvg...")
        start_time = time.time()
        # 随机选择客户端
        num_select = min(config.CLIENTS_PER_ROUND, len(valid_clients))
        fedavg_selected = np.random.choice(valid_clients, num_select, replace=False).tolist()

        # 并行训练
        trained_models = parallel_trainer.train_clients_parallel(
            fedavg_selected, global_models['FedAvg']
        )

        # 聚合和评估
        # 聚合和评估
        if len(trained_models) > 0:
            # --- 这是修正后的聚合逻辑 ---
            aggregated_state = OrderedDict()
            first_model_state = trained_models[0].state_dict()  # 以第一个模型为参考

            for key in first_model_state.keys():
                # 关键修正：检查参数是否为浮点数
                if first_model_state[key].is_floating_point():
                    # 如果是，正常求平均
                    params = [m.state_dict()[key].cpu() for m in trained_models]
                    aggregated_state[key] = torch.stack(params).mean(dim=0)
                else:
                    # 如果不是整数（例如num_batches_tracked），直接复制第一个的值
                    aggregated_state[key] = first_model_state[key].clone()
            # --- 修正逻辑结束 ---

            aggregated_model = model_class(num_classes=config.NUM_CLASSES)
            aggregated_model.load_state_dict(aggregated_state)
            aggregated_model = aggregated_model.to(config.DEVICE)

            # 确保这里的键名与当前方法一致
            global_models['SelfFed'] = aggregated_model

            accuracy, loss = env._evaluate_model(aggregated_model)
        else:
            accuracy, loss = 0, float('inf')

        elapsed_time = time.time() - start_time
        results['FedAvg']['accuracy'].append(accuracy)
        results['FedAvg']['loss'].append(loss)
        results['FedAvg']['time'].append(elapsed_time)
        results['FedAvg']['selected_clients'].append(fedavg_selected)
        print(f"  FedAvg: Acc={accuracy:.4f}, Loss={loss:.4f}, Time={elapsed_time:.2f}s")

        # 3. SelfFed
        print("Training SelfFed...")
        start_time = time.time()
        # 使用SelfFed的DQN选择策略
        selffed_selected = selffed_agent.select_clients(
            state[:len(state) // 2],  # 使用简化的状态
            num_select=min(config.CLIENTS_PER_ROUND, len(valid_clients))
        )

        # 确保选择的客户端有效
        selffed_selected = [c for c in selffed_selected if c in valid_clients]
        if len(selffed_selected) == 0:
            selffed_selected = valid_clients[:min(config.CLIENTS_PER_ROUND, len(valid_clients))]

        # 并行训练
        trained_models = parallel_trainer.train_clients_parallel(
            selffed_selected, global_models['SelfFed']
        )

        # 聚合和评估
        if len(trained_models) > 0:
            aggregated_state = OrderedDict()
            first_model_state = trained_models[0].state_dict()  # 以第一个模型为参考

            for key in first_model_state.keys():
                # 关键修正：检查参数是否为浮点数
                if first_model_state[key].is_floating_point():
                    # 如果是，正常求平均 (确保在CPU上操作以避免跨设备错误)
                    params = [m.state_dict()[key].cpu() for m in trained_models]
                    aggregated_state[key] = torch.stack(params).mean(dim=0)
                else:
                    # 如果不是整数（例如num_batches_tracked），直接复制第一个的值
                    aggregated_state[key] = first_model_state[key].clone()

            aggregated_model = model_class(num_classes=config.NUM_CLASSES)
            aggregated_model.load_state_dict(aggregated_state)
            aggregated_model = aggregated_model.to(config.DEVICE)
            global_models['SelfFed'] = aggregated_model
            accuracy, loss = env._evaluate_model(aggregated_model)
        else:
            accuracy, loss = 0, float('inf')

        elapsed_time = time.time() - start_time
        results['SelfFed']['accuracy'].append(accuracy)
        results['SelfFed']['loss'].append(loss)
        results['SelfFed']['time'].append(elapsed_time)
        results['SelfFed']['selected_clients'].append(selffed_selected)
        print(f"  SelfFed: Acc={accuracy:.4f}, Loss={loss:.4f}, Time={elapsed_time:.2f}s")

        # 4. FedProx
        print("Training FedProx...")
        start_time = time.time()
        # 随机选择客户端（FedProx不改变选择策略）
        fedprox_selected = np.random.choice(valid_clients, num_select, replace=False).tolist()

        # 使用FedProx训练方式
        trained_models = parallel_trainer.train_clients_parallel(
            fedprox_selected, global_models['FedProx'], method='fedprox'
        )

        # 聚合和评估
        if len(trained_models) > 0:
            aggregated_state = OrderedDict()
            first_model_state = trained_models[0].state_dict()  # 以第一个模型为参考

            for key in first_model_state.keys():
                # 关键修正：检查参数是否为浮点数
                if first_model_state[key].is_floating_point():
                    # 如果是，正常求平均 (确保在CPU上操作以避免跨设备错误)
                    params = [m.state_dict()[key].cpu() for m in trained_models]
                    aggregated_state[key] = torch.stack(params).mean(dim=0)
                else:
                    # 如果不是整数（例如num_batches_tracked），直接复制第一个的值
                    aggregated_state[key] = first_model_state[key].clone()

            aggregated_model = model_class(num_classes=config.NUM_CLASSES)
            aggregated_model.load_state_dict(aggregated_state)
            aggregated_model = aggregated_model.to(config.DEVICE)
            global_models['FedProx'] = aggregated_model
            accuracy, loss = env._evaluate_model(aggregated_model)
        else:
            accuracy, loss = 0, float('inf')

        elapsed_time = time.time() - start_time
        results['FedProx']['accuracy'].append(accuracy)
        results['FedProx']['loss'].append(loss)
        results['FedProx']['time'].append(elapsed_time)
        results['FedProx']['selected_clients'].append(fedprox_selected)
        print(f"  FedProx: Acc={accuracy:.4f}, Loss={loss:.4f}, Time={elapsed_time:.2f}s")

        # 更新环境状态
        all_selected = list(set(dqn_selected + fedavg_selected + selffed_selected + fedprox_selected))
        env._update_fatigue(all_selected)
        env._update_history(all_selected, np.mean([
            results['DQN-Compatibility (Ours)']['accuracy'][-1],
            results['FedAvg']['accuracy'][-1],
            results['SelfFed']['accuracy'][-1],
            results['FedProx']['accuracy'][-1]
        ]))

    return results


# ================== 可视化函数 ==================
def plot_chapter1_results(results, save_path='chapter1_results.png'):
    """绘制第一章结果对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 设置颜色
    colors = {
        'DQN-Compatibility (Ours)': 'red',
        'FedAvg': 'blue',
        'SelfFed': 'green',
        'FedProx': 'orange'
    }

    # 图1：准确率曲线
    ax = axes[0, 0]
    for method, data in results.items():
        if len(data['accuracy']) > 0:
            smoothed = pd.Series(data['accuracy']).rolling(window=5, min_periods=1).mean()
            ax.plot(smoothed, label=method, linewidth=2, color=colors[method])
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图2：损失曲线
    ax = axes[0, 1]
    for method, data in results.items():
        if len(data['loss']) > 0:
            smoothed = pd.Series(data['loss']).rolling(window=5, min_periods=1).mean()
            ax.plot(smoothed, label=method, linewidth=2, color=colors[method])
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图3：训练时间
    ax = axes[1, 0]
    avg_times = {}
    for method, data in results.items():
        if len(data['time']) > 0:
            avg_times[method] = np.mean(data['time'])

    if len(avg_times) > 0:
        bars = ax.bar(avg_times.keys(), avg_times.values(),
                      color=[colors[m] for m in avg_times.keys()])
        ax.set_ylabel('Average Time (seconds)')
        ax.set_title('Training Efficiency')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom')

    # 图4：最终性能对比
    ax = axes[1, 1]
    final_accs = {}
    for method, data in results.items():
        if len(data['accuracy']) >= 10:
            final_accs[method] = np.mean(data['accuracy'][-10:])
        elif len(data['accuracy']) > 0:
            final_accs[method] = np.mean(data['accuracy'])

    if len(final_accs) > 0:
        bars = ax.bar(final_accs.keys(), final_accs.values(),
                      color=[colors[m] for m in final_accs.keys()])
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Final Model Performance')
        ax.set_ylim([0, 1])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 添加数值标签
        for bar, (method, acc) in zip(bars, final_accs.items()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{acc:.3f}', ha='center', va='bottom')

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
            avg_time = np.mean(results[method]['time'])
            print(f"{method}:")
            print(f"  最终准确率: {final_acc:.4f}")
            print(f"  平均训练时间: {avg_time:.2f}s")


def save_results(results, filename='chapter1_results.json'):
    """保存实验结果"""
    config = Config()

    # 转换为可序列化格式
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, list):
                    if k != 'selected_clients':  # 不保存选中的客户端列表
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
    print("联邦学习激励机制实验 - 第一章改进版")
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

        # 测试CNN前向传播
        print("测试CNN前向传播...")
        test_model = CNNModel(num_classes=10).to(device)
        test_input = torch.randn(32, 3, 32, 32, device=device)
        start_time = time.time()
        with torch.no_grad():
            test_output = test_model(test_input)
        torch.cuda.synchronize()
        compute_time = time.time() - start_time
        print(f"  前向传播时间: {compute_time:.4f}秒")

        # 清理测试张量
        del a, b, c, test_model, test_input, test_output
        torch.cuda.empty_cache()
        print("GPU测试完成！")

    # 运行实验
    try:
        print("\n开始运行第一章实验...")
        print("包含4个对比方法: DQN-Compatibility (Ours), FedAvg, SelfFed, FedProx")

        # 运行实验
        results = run_chapter1_experiment()

        # 保存结果
        save_results(results, 'chapter1_results.json')

        # 绘制结果图表
        plot_chapter1_results(results, 'chapter1_results.png')

        print("\n" + "=" * 80)
        print("第一章实验完成！")
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