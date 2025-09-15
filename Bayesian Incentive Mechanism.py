import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from copy import deepcopy


# ---------------------------------------------------
# 1. 环境设置与超参数 (根据论文第五节)
# ---------------------------------------------------
class Config:
    NUM_CLIENTS = 100
    NUM_ROUNDS = 40
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    MALICIOUS_FRACTION = 0.4  # 可调整恶意客户端比例 (例如 0.3, 0.4, 0.5)

    # 数据集与分区
    DATASET = 'MNIST'  # 可选 'MNIST' 或 'FashionMNIST'
    DIRICHLET_ALPHA = 0.5

    # 激励机制参数
    REWARD = 10
    COST = 2  # 理论分析中使用
    VERIFICATION_THRESHOLD = 2.5
    VALIDATION_SET_SIZE = 200

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")


# ---------------------------------------------------
# 2. CNN模型定义 (根据论文第五节)
# ---------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------
# 3. 数据加载与Non-IID分区
# ---------------------------------------------------
def get_data(config):
    """加载数据集, 创建服务器验证集, 并为客户端进行Non-IID分区"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])

    if config.DATASET == 'FashionMNIST':
        full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:  # 默认为 MNIST
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 从训练集中划分服务器的私有验证集
    indices = list(range(len(full_train_dataset)))
    np.random.shuffle(indices)

    val_indices = indices[:config.VALIDATION_SET_SIZE]
    train_indices = indices[config.VALIDATION_SET_SIZE:]

    server_validation_set = Subset(full_train_dataset, val_indices)
    client_train_dataset = Subset(full_train_dataset, train_indices)

    # 使用狄利克雷分布为客户端进行Non-IID分区
    num_classes = 10
    labels = np.array([full_train_dataset.targets[i] for i in client_train_dataset.indices])

    client_indices = [[] for _ in range(config.NUM_CLIENTS)]
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(config.DIRICHLET_ALPHA, config.NUM_CLIENTS))

        # 按比例分配样本索引
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_indices = [
            old_indices + new_indices.tolist()
            for old_indices, new_indices in zip(client_indices, np.split(idx_k, proportions))
        ]

    # 创建客户端数据集
    client_datasets = []
    for indices in client_indices:
        # 映射回原始数据集的索引
        original_indices = [client_train_dataset.indices[i] for i in indices]
        client_datasets.append(Subset(full_train_dataset, original_indices))

    return client_datasets, server_validation_set, test_dataset


# ---------------------------------------------------
# 4. 客户端实现
# ---------------------------------------------------
class LabelFlippingDataset(Dataset):
    """一个包装器, 用于实现标签翻转攻击"""

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        flipped_label = (label + 1) % 10  # y' = (y+1) mod 10
        return image, flipped_label


class Client:
    def __init__(self, client_id, dataset, config, is_malicious=False):
        self.client_id = client_id
        self.config = config
        self.is_malicious = is_malicious

        if self.is_malicious:
            self.dataset = LabelFlippingDataset(dataset)
        else:
            self.dataset = dataset

        self.dataloader = DataLoader(self.dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.model = CNN().to(self.config.DEVICE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_model_state):
        """在本地数据上训练模型"""
        self.model.load_state_dict(global_model_state)
        self.model.train()
        for epoch in range(self.config.LOCAL_EPOCHS):
            for data, target in self.dataloader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()


# ---------------------------------------------------
# 5. 服务器实现 (核心: 贝叶斯激励机制)
# ---------------------------------------------------
class Server:
    def __init__(self, clients, validation_set, test_set, config):
        self.clients = clients
        self.validation_loader = DataLoader(validation_set, batch_size=config.BATCH_SIZE)
        self.test_loader = DataLoader(test_set, batch_size=1000)
        self.config = config
        self.global_model = CNN().to(config.DEVICE)
        self.criterion = nn.CrossEntropyLoss()

    def run_simulation(self):
        """执行完整的联邦学习模拟流程"""
        for t in range(self.config.NUM_ROUNDS):
            print(f"\n--- Communication Round {t + 1}/{self.config.NUM_ROUNDS} ---")

            # 1. 广播全局模型并收集客户端更新 (Algorithm 1, Lines 2-7)
            global_model_state = deepcopy(self.global_model.state_dict())
            client_updates = []
            for client in self.clients:
                update = client.train(global_model_state)
                client_updates.append(update)

            # 2. 验证与支付循环 (Algorithm 1, Lines 8-17)
            verified_updates = self.verify_and_filter_updates(client_updates)

            # 3. 选择性聚合 (Algorithm 1, Lines 18-22)
            if len(verified_updates) > 0:
                print(f"Aggregating {len(verified_updates)} verified updates.")
                self.aggregate_updates(verified_updates)
            else:
                print("No updates passed verification. Maintaining current model.")

            # 4. 评估全局模型性能
            self.evaluate_model(t + 1)

    def verify_and_filter_updates(self, client_updates):
        """
        核心激励机制: 使用私有验证集评估每个更新, 过滤掉不合格的更新
        (Algorithm 1, Lines 9-17)
        """
        verified_updates = []
        temp_model = CNN().to(self.config.DEVICE)

        for i, update in enumerate(client_updates):
            temp_model.load_state_dict(update)
            loss = self.evaluate_loss(temp_model)

            client_type = "Malicious" if self.clients[i].is_malicious else "Benevolent"

            if loss < self.config.VERIFICATION_THRESHOLD:
                # 更新通过验证, 支付奖励 R, 将更新加入已验证集合
                print(
                    f"Client {i} ({client_type}): Update VERIFIED (Loss: {loss:.4f} < {self.config.VERIFICATION_THRESHOLD}). Paying reward {self.config.REWARD}.")
                verified_updates.append(update)
            else:
                # 更新未通过, 支付 0, 客户端承担成本 C
                print(
                    f"Client {i} ({client_type}): Update REJECTED (Loss: {loss:.4f} >= {self.config.VERIFICATION_THRESHOLD}). Paying 0.")

        return verified_updates

    def evaluate_loss(self, model):
        """在服务器的私有验证集上计算模型损失 (Algorithm 1, Line 10)"""
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in self.validation_loader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                output = model(data)
                total_loss += self.criterion(output, target).item() * data.size(0)
        return total_loss / len(self.validation_loader.dataset)

    def aggregate_updates(self, verified_updates):
        """对所有已验证的更新进行平均 (Algorithm 1, Line 19)"""
        aggregated_state_dict = deepcopy(verified_updates[0])
        for key in aggregated_state_dict:
            # 将所有已验证更新的参数相加
            for i in range(1, len(verified_updates)):
                aggregated_state_dict[key] += verified_updates[i][key]
            # 取平均
            aggregated_state_dict[key] = torch.div(aggregated_state_dict[key], len(verified_updates))

        self.global_model.load_state_dict(aggregated_state_dict)

    def evaluate_model(self, round_num):
        """在测试集上评估全局模型的准确率"""
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f"Round {round_num} - Global Model Test Accuracy: {accuracy:.2f}%")


# ---------------------------------------------------
# 6. 主执行流程
# ---------------------------------------------------
if __name__ == '__main__':
    config = Config()

    # 1. 加载数据并分区
    print("Loading and partitioning data...")
    client_datasets, server_val_set, test_set = get_data(config)
    print("Data setup complete.")

    # 2. 创建客户端
    num_malicious = int(config.NUM_CLIENTS * config.MALICIOUS_FRACTION)
    clients = []
    for i in range(config.NUM_CLIENTS):
        is_malicious = i < num_malicious
        client_type = "Malicious" if is_malicious else "Benevolent"
        clients.append(Client(i, client_datasets[i], config, is_malicious))
    print(
        f"Created {config.NUM_CLIENTS} clients ({num_malicious} malicious, {config.NUM_CLIENTS - num_malicious} benevolent).")

    # 3. 创建服务器并开始模拟
    server = Server(clients, server_val_set, test_set, config)
    server.run_simulation()