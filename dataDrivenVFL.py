import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class ResourceNetwork(tf.keras.Model):
    """资源网络，学习 x(theta)"""

    def __init__(self, f_min, f_max, n_hidden=2, n_neurons=25):
        super(ResourceNetwork, self).__init__()
        self.f_min = f_min
        self.f_max = f_max
        # 修正：正确初始化隐藏层列表
        self.hidden_layers = [layers.Dense(n_neurons, activation='tanh') for _ in range(n_hidden)]
        # 定制激活函数，确保输出在 [f_min, f_max] 范围内
        self.output_layer = layers.Dense(1, activation=lambda x: self.f_min + (self.f_max - self.f_min) * tf.sigmoid(x))

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class RewardNetwork(tf.keras.Model):
    """奖励网络，学习 R_hat(theta)"""

    def __init__(self, n_hidden=2, n_neurons=7):
        super(RewardNetwork, self).__init__()
        # 修正：正确初始化隐藏层列表
        self.hidden_layers = [layers.Dense(n_neurons, activation='tanh') for _ in range(n_hidden)]
        # 定制激活函数，确保输出 R_hat >= 1
        self.output_layer = layers.Dense(1, activation=lambda x: tf.sigmoid(x) + 1.0)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class ContractModel:
    """将资源和奖励网络组合成一个完整的合约模型"""

    def __init__(self, params):
        self.params = params
        self.resource_net = ResourceNetwork(params['f_min'], params['f_max'],
                                            params['n_hidden_res'], params['n_neurons_res'])
        self.reward_net = RewardNetwork(params['n_hidden_rew'], params['n_neurons_rew'])

        # 构建模型以初始化变量
        dummy_input = tf.ones((1, 1))
        self.resource_net(dummy_input)
        self.reward_net(dummy_input)

        # 现在可以获取 trainable_variables
        self.trainable_variables = self.resource_net.trainable_variables + \
                                   self.reward_net.trainable_variables

    def get_contract(self, theta):
        """根据 theta 计算合约对 (x, R)"""
        # 确保输入是 float32
        theta = tf.cast(theta, tf.float32)

        x = self.resource_net(theta)
        r_hat = self.reward_net(theta)

        # 根据论文公式计算成本 E_i^t 和最终奖励 R
        cost = self._calculate_cost(theta, x)

        # 确保 tau 是 float32
        tau = tf.cast(self.params['tau'], tf.float32)
        reward = r_hat * tau * cost

        return x, reward

    def _calculate_cost(self, theta, x):
        """计算车辆的总成本 E_i^t (Eq. 4)"""
        # 确保所有参数都是 float32
        psi = tf.cast(self.params['psi'], tf.float32)
        c = tf.cast(self.params['c'], tf.float32)
        s = tf.cast(self.params['s'], tf.float32)
        E_com = tf.cast(self.params['E_com'], tf.float32)
        T_com = tf.cast(self.params['T_com'], tf.float32)
        zeta = tf.cast(self.params['zeta'], tf.float32)
        P_idle = tf.cast(self.params['P_idle'], tf.float32)
        d_k = tf.cast(self.params['d_k'], tf.float32)

        # 确保输入也是 float32
        theta = tf.cast(theta, tf.float32)
        x = tf.cast(x, tf.float32)

        # 确保 x 不为0以避免除零错误
        x_safe = tf.maximum(x, 1e-9)

        # 计算时间和能量
        T_cmp = (c * s) / x_safe
        E_cmp = zeta * c * s * (x_safe ** 2)

        # 使用论文中的非凹成本模型 (Eq. 4)
        total_cost = (psi / theta) * E_cmp + E_com + (d_k - (psi / theta) * T_cmp - T_com) * P_idle
        return total_cost

    def get_utility(self, true_theta, reported_theta):
        """计算车辆在真实类型为 true_theta，报告类型为 reported_theta 时的效用"""
        # 确保输入是 float32
        true_theta = tf.cast(true_theta, tf.float32)
        reported_theta = tf.cast(reported_theta, tf.float32)

        x_reported, r_reported = self.get_contract(reported_theta)

        # 成本总是基于真实类型 true_theta 计算
        cost_true = self._calculate_cost(true_theta, x_reported)

        # 确保 tau 是 float32
        tau = tf.cast(self.params['tau'], tf.float32)
        utility = r_reported - tau * cost_true
        return utility


def generate_data(num_samples, params):
    """生成车辆类型 theta 的数据"""
    eps = np.random.uniform(params['eps_min'], params['eps_max'], (num_samples, 1))
    theta = params['psi'] / np.log(1.0 / eps)
    return tf.convert_to_tensor(theta, dtype=tf.float32)


def calculate_rsu_satisfaction(theta, x, params):
    """计算 RSU 的满意度 S_i (Eq. 21)"""
    # 确保所有参数都是 float32
    c = tf.cast(params['c'], tf.float32)
    s = tf.cast(params['s'], tf.float32)
    psi = tf.cast(params['psi'], tf.float32)
    T_com = tf.cast(params['T_com'], tf.float32)
    nu = tf.cast(params['nu'], tf.float32)
    T_max = tf.cast(params['T_max'], tf.float32)

    # 确保输入也是 float32
    theta = tf.cast(theta, tf.float32)
    x = tf.cast(x, tf.float32)

    T_cmp = (c * s) / tf.maximum(x, 1e-9)
    T_total = (psi / theta) * T_cmp + T_com
    # 确保 log 内参数为正
    satisfaction = nu * tf.math.log(tf.maximum(T_max - T_total, 1e-9))
    return satisfaction


@tf.function
def calculate_regret(model, true_theta, params):
    """计算事后后悔值 (Eq. 10 & 16)"""
    # 初始化谎报类型为真实类型
    reported_theta = tf.identity(true_theta)

    # 使用梯度上升法寻找最大化效用的谎报类型 (内层循环)
    for _ in tf.range(params['M_regret']):
        with tf.GradientTape() as tape:
            tape.watch(reported_theta)
            utility = model.get_utility(true_theta, reported_theta)
        grad = tape.gradient(utility, reported_theta)
        # 防止梯度为 None
        if grad is None:
            break
        reported_theta = reported_theta + params['alpha_regret'] * grad

    # 计算最大效用与真实效用之差，即后悔值
    max_utility = model.get_utility(true_theta, reported_theta)
    true_utility = model.get_utility(true_theta, true_theta)
    regret = tf.maximum(max_utility - true_utility, 0)
    return tf.reduce_mean(regret)


def train():
    """训练主循环"""
    # --- 参数设置 (基于论文 Table I & II) ---
    PARAMS = {
        # FL 环境参数
        'T_max': 250.0, 'R_max': 150.0, 'T_com': 10.0, 'E_com': 20.0,
        'f_min': 0.5, 'f_max': 2.0, 'c': 5e9, 's': 230.0, 'nu': 800.0,
        'tau': 1.0, 'psi': 1.0, 'eps_min': 0.31, 'eps_max': 0.91,
        'P_idle': 0.15, 'd_k': 270.0, 'zeta': 1e-27, 'N_vehicles': 20,

        # DNN 架构参数
        'n_hidden_res': 2, 'n_neurons_res': 25,
        'n_hidden_rew': 2, 'n_neurons_rew': 7,

        # 训练参数
        'epochs': 500, 'batch_size': 128, 'learning_rate': 2e-3,
        'rho': 5e-4, 'lambda_init': 0.01, 'mu_init': 1e-3, 'K_update': 100,
        'alpha_regret': 0.1, 'M_regret': 20,
    }

    # 初始化模型和优化器
    model = ContractModel(PARAMS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=PARAMS['learning_rate'])

    # 初始化拉格朗日乘子
    lambda_val = tf.Variable(PARAMS['lambda_init'], dtype=tf.float32)
    mu_val = tf.Variable(PARAMS['mu_init'], dtype=tf.float32)

    # 记录训练过程
    history = {'loss': [], 'regret': [], 'rsu_utility': [], 'total_reward': []}

    # 生成训练数据
    train_data = generate_data(PARAMS['epochs'] * PARAMS['batch_size'], PARAMS)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(PARAMS['batch_size'])

    step = 0
    for theta_batch in tqdm(train_dataset, desc="Training Progress"):
        with tf.GradientTape() as tape:
            # 获取合约并计算各项指标
            x_batch, r_batch = model.get_contract(theta_batch)
            s_batch = calculate_rsu_satisfaction(theta_batch, x_batch, PARAMS)

            # 1. RSU 效用 (目标函数)
            rsu_utility_batch = tf.reduce_mean(s_batch - r_batch)

            # 2. 事后后悔值 (IC 约束)
            regret_batch = calculate_regret(model, theta_batch, PARAMS)

            # 3. 总奖励预算 (预算约束)
            total_reward_batch = tf.reduce_mean(r_batch) * PARAMS['N_vehicles']
            reward_constraint_val = total_reward_batch - PARAMS['R_max']

            # 4. 计算增广拉格朗日损失 (Eq. 15)
            loss_objective = -rsu_utility_batch  # 最小化负效用
            loss_regret = lambda_val * regret_batch + (PARAMS['rho'] / 2) * tf.square(regret_batch)

            mu_term = mu_val + PARAMS['rho'] * reward_constraint_val
            loss_reward_budget = (1 / (2 * PARAMS['rho'])) * (tf.square(tf.maximum(0.0, mu_term)) - tf.square(mu_val))

            total_loss = loss_objective + loss_regret + loss_reward_budget

        # 更新网络权重
        grads = tape.gradient(total_loss, model.trainable_variables)

        # 检查梯度是否为None并过滤
        if grads is not None and any(g is not None for g in grads):
            valid_grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            if valid_grads_and_vars:
                optimizer.apply_gradients(valid_grads_and_vars)

        # 定期更新拉格朗日乘子 (Eq. 18 & 19)
        if step % PARAMS['K_update'] == 0:
            lambda_val.assign_add(PARAMS['rho'] * regret_batch)
            mu_val.assign(tf.maximum(0.0, mu_val + PARAMS['rho'] * reward_constraint_val))

        # 记录历史数据
        history['loss'].append(total_loss.numpy())
        history['regret'].append(regret_batch.numpy())
        history['rsu_utility'].append(rsu_utility_batch.numpy() * PARAMS['N_vehicles'])
        history['total_reward'].append(total_reward_batch.numpy())

        step += 1

    print("Training finished.")
    return model, history, PARAMS


def plot_results(history, model, params):
    """结果可视化"""
    # 图 4: 训练过程收敛图
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Process Analysis (Replication of Fig. 4)', fontsize=16)

    axs[0, 0].plot(history['loss'])
    axs[0, 0].set_title('Augmented Lagrangian Loss')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Loss')

    axs[0, 1].plot(history['regret'])
    axs[0, 1].axhline(y=0, color='r', linestyle='--')
    axs[0, 1].set_title('Mean Ex-post Regret')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Regret')
    axs[0, 1].set_ylim(bottom=-0.1)

    axs[1, 0].plot(history['rsu_utility'])
    axs[1, 0].set_title('Utility of RSU')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Utility')

    axs[1, 1].plot(history['total_reward'])
    axs[1, 1].axhline(y=params['R_max'], color='r', linestyle='--', label=f'Budget R_max={params["R_max"]}')
    axs[1, 1].set_title('Total Reward of All Vehicles')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Reward')
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # 图 5: 车辆效用 vs 报告类型
    plt.figure(figsize=(8, 6))
    theta_min = params['psi'] / np.log(1.0 / params['eps_max'])
    theta_max = params['psi'] / np.log(1.0 / params['eps_min'])
    theta_range = tf.linspace(theta_min, theta_max, 100)
    theta_range = tf.reshape(theta_range, [-1, 1])

    true_thetas_to_plot = tf.constant([[5.0], [10.0], [15.0], [20.0], [25.0]], dtype=tf.float32)

    for true_theta in true_thetas_to_plot:
        utilities = []
        for reported_theta in theta_range:
            # 确保输入是二维的
            utility = model.get_utility(tf.reshape(true_theta, [1, 1]), tf.reshape(reported_theta, [1, 1]))
            utilities.append(utility.numpy())
        utilities = np.array(utilities).flatten()

        plt.plot(theta_range.numpy().flatten(), utilities, label=f'True θ = {true_theta.numpy()[0]:.1f}')
        # 确保输入是二维的
        true_utility = model.get_utility(tf.reshape(true_theta, [1, 1]), tf.reshape(true_theta, [1, 1])).numpy()
        plt.plot(true_theta.numpy()[0], true_utility, 'o', markersize=8)

    plt.title('Vehicle Utility vs. Reported Type (Replication of Fig. 5)')
    plt.xlabel('Reported Type (θ_hat)')
    plt.ylabel('Utility of Vehicle')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 图 8: 学到的合约函数 x(θ) 和 R(θ)
    theta_range_sorted = tf.sort(generate_data(500, params), axis=0)
    # 确保输入是二维的
    x_learned, r_learned = model.get_contract(tf.reshape(theta_range_sorted, [-1, 1]))

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Learned Optimal Contract Functions (Replication of Fig. 8)', fontsize=16)

    axs[0].plot(theta_range_sorted.numpy(), x_learned.numpy())
    axs[0].set_title('Computation Resource vs. Vehicle Type')
    axs[0].set_xlabel('Type of Vehicle (θ)')
    axs[0].set_ylabel('Computation Resource x(θ)')

    axs[1].plot(theta_range_sorted.numpy(), r_learned.numpy())
    axs[1].set_title('Reward vs. Vehicle Type')
    axs[1].set_xlabel('Type of Vehicle (θ)')
    axs[1].set_ylabel('Reward R(θ)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
# --- 执行 ---
if __name__ == '__main__':
    trained_model, training_history, params = train()
    plot_results(training_history, trained_model, params)