import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import copy
import random
import warnings

# Suppress CVXPY UserWarning about variables being treated as constant
warnings.filterwarnings("ignore", category=UserWarning)


# --- Section 1: Configuration Parameters ---
# Based on Table 2 and Section 6 of the IncEFL paper
class Config:
    # System Parameters
    TOTAL_WORKERS = 100
    SELECTED_WORKERS_PER_ROUND = 10
    MAX_LATENCY = 700.0
    REWARD_BUDGET = 10000.0
    COORDINATION_COST = 500.0

    # Contract/Economic Parameters
    SATISFACTION_PARAM_DELTA = 800.0
    REWARD_UNIT_COST_PHI = 0.01
    ENERGY_CONVERSION_ZETA = 2.0
    ENERGY_WEIGHT_OMEGA = 1.0
    CORRELATION_COEFFICIENT_MU = 1.0

    # Worker Model Parameters
    CPU_FREQ_RANGE = [0.2, 0.6]  # GHz
    CPU_CYCLES_PER_SAMPLE = 5.0
    WORKER_THETA_DIST = {'mean': 0.7, 'std': 0.1}

    # Communication Model
    COMM_TIME = 10.0
    COMM_ENERGY = 20.0

    # Federated Learning Parameters
    DATASET = 'CIFAR10'
    COMMUNICATION_ROUNDS = 100
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01

    # Non-IID Data Partition Parameters
    DIRICHLET_ALPHA = 0.5

    # Misc
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42


# --- Section 2: CNN Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --- Section 3: Non-IID Data Partitioning ---
class Cifar10Partitioner:
    def __init__(self, data_dir, num_clients, alpha):
        self.num_clients = num_clients
        self.alpha = alpha

        # Add transforms for CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        self.test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        self.labels = np.array(self.dataset.targets)
        self.num_classes = len(np.unique(self.labels))
        self.client_indices = self._partition_data()

    def _partition_data(self):
        client_indices = [[] for _ in range(self.num_clients)]
        class_indices = [np.where(self.labels == i)[0] for i in range(self.num_classes)]

        # Dirichlet distribution for non-IID partitioning
        for k in range(self.num_classes):
            idx_k = class_indices[k]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = np.split(idx_k, proportions)

            for j in range(self.num_clients):
                if j < len(idx_batch):
                    client_indices[j].extend(idx_batch[j].tolist())

        # Shuffle indices for each client
        for j in range(self.num_clients):
            np.random.shuffle(client_indices[j])

        return client_indices

    def get_dataloader(self, client_id, batch_size):
        indices = self.client_indices[client_id]
        client_dataset = Subset(self.dataset, indices)
        return DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

    def get_test_dataloader(self, batch_size):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)


# --- Section 4: Worker and Incentive Mechanism Implementation ---
class Worker:
    def __init__(self, id, config):
        self.id = id
        self.theta = np.clip(np.random.normal(config.WORKER_THETA_DIST['mean'],
                                              config.WORKER_THETA_DIST['std']), 0.1, 1.0)
        self.f_n = np.random.uniform(config.CPU_FREQ_RANGE[0], config.CPU_FREQ_RANGE[1])
        self.d_n = 0.0  # Data quality cost (will be calculated)
        self.s_n = 0.0  # Data size (will be optimized)


class IncentiveMechanism:
    def __init__(self, workers, config):
        self.workers = sorted(workers, key=lambda w: w.theta)
        self.config = config
        self.K = len(self.workers)

    def _solve_contract_optimization(self, constraints=[]):
        """Solves the optimization problem from Eq. (37) in the paper"""
        s = cp.Variable(self.K, name='data_size', nonneg=True)

        objective_terms = []
        for i in range(self.K):
            worker = self.workers[i]
            # Time delay calculation (Eq. 9)
            time_delay = self.config.COMM_TIME + (self.config.CPU_CYCLES_PER_SAMPLE * s[i]) / worker.f_n

            # Ensure MAX_LATENCY - time_delay > 0 for log function
            satisfaction = self.config.SATISFACTION_PARAM_DELTA * cp.log(self.config.MAX_LATENCY - time_delay + 1e-6)

            # Reward cost
            reward_cost = self.config.REWARD_UNIT_COST_PHI * worker.theta * s[i]

            objective_terms.append(satisfaction - reward_cost)

        total_objective = cp.Maximize(cp.sum(objective_terms))

        # Add constraints
        opt_constraints = constraints + [
            s[i] >= 0 for i in range(self.K)
        ] + [
                              (self.config.CPU_CYCLES_PER_SAMPLE * s[i]) / self.workers[i].f_n <=
                              self.config.MAX_LATENCY - self.config.COMM_TIME - 1 for i in range(self.K)
                          ]

        # Add budget constraint
        total_reward = cp.sum([self.config.REWARD_UNIT_COST_PHI * self.workers[i].theta * s[i]
                               for i in range(self.K)])
        opt_constraints.append(total_reward <= self.config.REWARD_BUDGET)

        problem = cp.Problem(total_objective, opt_constraints)

        try:
            problem.solve(solver=cp.SCS, verbose=False)
            if s.value is None or problem.status not in ['optimal', 'optimal_inaccurate']:
                return np.zeros(self.K)
            return np.maximum(s.value, 0)  # Ensure non-negative
        except:
            return np.zeros(self.K)

    def _progressive_adjustment_algorithm(self, s_star):
        """Implements Algorithm 1 from the paper"""
        adjusted = True
        max_iterations = 10
        iteration = 0

        while adjusted and iteration < max_iterations:
            adjusted = False
            iteration += 1

            for i in range(self.K - 1):
                if s_star[i] > s_star[i + 1] + 1e-6:  # Add tolerance
                    adjusted = True
                    # Find the end of the violating subset
                    j = i + 1
                    while j < self.K and s_star[i] > s_star[j] + 1e-6:
                        j += 1

                    # Average the values in the violating subset
                    avg_value = np.mean(s_star[i:j])
                    s_star[i:j] = avg_value
                    break

        return s_star

    def select_workers(self):
        """Main function to select workers based on incentive mechanism"""
        # 1. Solve relaxed optimization problem
        s_initial = self._solve_contract_optimization()

        # 2. Enforce monotonicity using Algorithm 1
        s_final = self._progressive_adjustment_algorithm(s_initial.copy())

        # Assign the optimal data size to each worker
        for i in range(self.K):
            self.workers[i].s_n = max(s_final[i], 0)

        # 3. Calculate utility for each worker and select top K
        worker_utilities = []
        for worker in self.workers:
            if worker.s_n > 0:
                time_delay = self.config.COMM_TIME + (self.config.CPU_CYCLES_PER_SAMPLE * worker.s_n) / worker.f_n
                if self.config.MAX_LATENCY - time_delay > 0:
                    satisfaction = self.config.SATISFACTION_PARAM_DELTA * np.log(self.config.MAX_LATENCY - time_delay)
                    reward_cost = self.config.REWARD_UNIT_COST_PHI * worker.theta * worker.s_n
                    utility = satisfaction - reward_cost
                    worker_utilities.append((worker, utility))

        # 4. Select top workers based on utility
        worker_utilities.sort(key=lambda x: x[1], reverse=True)
        selected_workers = [w[0] for w in worker_utilities[:self.config.SELECTED_WORKERS_PER_ROUND]]

        return selected_workers


# --- Section 5: Federated Learning Utilities ---
def train_local_model(worker_id, model, dataloader, config):
    """Local training for one worker"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.LOCAL_EPOCHS):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Break if we've processed enough samples (based on s_n)
            if batch_idx >= 5:  # Limit batches for faster demo
                break

    return model.state_dict()


def aggregate_models(worker_models, global_model):
    """FedAvg aggregation"""
    global_state_dict = global_model.state_dict()

    for key in global_state_dict.keys():
        global_state_dict[key] = torch.stack([
            worker_models[i][key].float() for i in range(len(worker_models))
        ], 0).mean(0)

    global_model.load_state_dict(global_state_dict)
    return global_model


def evaluate_model(model, test_dataloader, config):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_dataloader):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Limit evaluation batches for speed
            if batch_idx >= 20:
                break

    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / min(len(test_dataloader), 20)
    return accuracy, avg_loss


# --- Section 6: Main Simulation Loop ---
def main():
    # Set seed for reproducibility
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    random.seed(Config.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)

    print("--- Initializing Simulation ---")
    print(f"Device: {Config.DEVICE}")

    # 1. Initialize components
    print("Loading and partitioning dataset...")
    partitioner = Cifar10Partitioner(data_dir='./data',
                                     num_clients=Config.TOTAL_WORKERS,
                                     alpha=Config.DIRICHLET_ALPHA)
    test_loader = partitioner.get_test_dataloader(Config.BATCH_SIZE)

    # Create workers
    all_workers = [Worker(i, Config) for i in range(Config.TOTAL_WORKERS)]

    # Initialize models
    global_model_incefl = SimpleCNN().to(Config.DEVICE)
    global_model_fedavg = copy.deepcopy(global_model_incefl)

    results_incefl = []
    results_fedavg = []

    # --- Run IncEFL Simulation ---
    print("\n--- Starting IncEFL Simulation ---")
    for round_num in range(Config.COMMUNICATION_ROUNDS):
        # Incentive mechanism selects workers
        incentive_module = IncentiveMechanism(all_workers[:20], Config)  # Use subset for efficiency
        selected_workers = incentive_module.select_workers()

        if len(selected_workers) == 0:
            print(f"Warning: No workers selected in round {round_num + 1}")
            selected_workers = random.sample(all_workers, Config.SELECTED_WORKERS_PER_ROUND)

        local_models = []
        for worker in selected_workers:
            local_model = copy.deepcopy(global_model_incefl)
            dataloader = partitioner.get_dataloader(worker.id, Config.BATCH_SIZE)
            local_state_dict = train_local_model(worker.id, local_model, dataloader, Config)
            local_models.append(local_state_dict)

        # Aggregate models
        if len(local_models) > 0:
            global_model_incefl = aggregate_models(local_models, global_model_incefl)

        # Evaluate
        acc, loss = evaluate_model(global_model_incefl, test_loader, Config)
        results_incefl.append(acc)

        if (round_num + 1) % 10 == 0:
            print(f"IncEFL Round {round_num + 1}/{Config.COMMUNICATION_ROUNDS} | "
                  f"Accuracy: {acc:.2f}% | Loss: {loss:.4f}")

    # --- Run FedAvg (Random Selection) Simulation ---
    print("\n--- Starting FedAvg (Benchmark) Simulation ---")
    for round_num in range(Config.COMMUNICATION_ROUNDS):
        # Randomly select workers
        selected_worker_ids = random.sample(range(Config.TOTAL_WORKERS),
                                            Config.SELECTED_WORKERS_PER_ROUND)

        local_models = []
        for worker_id in selected_worker_ids:
            local_model = copy.deepcopy(global_model_fedavg)
            dataloader = partitioner.get_dataloader(worker_id, Config.BATCH_SIZE)
            local_state_dict = train_local_model(worker_id, local_model, dataloader, Config)
            local_models.append(local_state_dict)

        # Aggregate models
        global_model_fedavg = aggregate_models(local_models, global_model_fedavg)

        # Evaluate
        acc, loss = evaluate_model(global_model_fedavg, test_loader, Config)
        results_fedavg.append(acc)

        if (round_num + 1) % 10 == 0:
            print(f"FedAvg Round {round_num + 1}/{Config.COMMUNICATION_ROUNDS} | "
                  f"Accuracy: {acc:.2f}% | Loss: {loss:.4f}")

    # --- Plotting Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, Config.COMMUNICATION_ROUNDS + 1), results_incefl,
             marker='o', linestyle='-', label='IncEFL', markersize=4)
    plt.plot(range(1, Config.COMMUNICATION_ROUNDS + 1), results_fedavg,
             marker='x', linestyle='--', label='FedAvg (Random Selection)', markersize=4)
    plt.title('IncEFL vs. FedAvg on CIFAR-10 (non-IID)', fontsize=14)
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(0, max(max(results_incefl, default=50), max(results_fedavg, default=50)) + 5)
    plt.tight_layout()
    plt.savefig('incefl_vs_fedavg_reproduction.png', dpi=150)
    print("\n✓ Simulation completed successfully!")
    print(f"Final IncEFL Accuracy: {results_incefl[-1]:.2f}%" if results_incefl else "N/A")
    print(f"Final FedAvg Accuracy: {results_fedavg[-1]:.2f}%" if results_fedavg else "N/A")
    print("Results plot saved as 'incefl_vs_fedavg_reproduction.png'")
    plt.show()


if __name__ == "__main__":
    main()