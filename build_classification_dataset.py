import pandas as pd
import numpy as np
import os
from tqdm import tqdm


# =============================================================================
# 1. 数据增强与故障注入函数
# =============================================================================

def augment_signal(signal, noise_level=0.05, scale_range=(0.95, 1.05)):
    """对信号进行简单的数据增强：加噪和缩放"""
    # 加噪
    noise = np.random.randn(len(signal)) * np.std(signal) * noise_level
    augmented_signal = signal + noise
    # 缩放
    scale = np.random.uniform(scale_range[0], scale_range[1])
    augmented_signal = augmented_signal * scale
    return augmented_signal


def inject_imbalance_fault(signal, rpm=3000, severity=0.3):
    """注入不平衡故障（在转频上叠加正弦波）"""
    fs = 1 / (10 / len(signal))  # 假设采样频率，这里需要根据实际情况调整或估算
    f_rpm = rpm / 60.0  # 转频 (Hz)
    t = np.arange(len(signal)) / fs

    # 注入信号的强度应与原始信号的标准差相关
    amplitude = np.std(signal) * severity * (1 + np.random.rand())

    imbalance_wave = amplitude * np.sin(2 * np.pi * f_rpm * t + np.random.rand() * 2 * np.pi)
    return signal + imbalance_wave


def inject_misalignment_fault(signal, rpm=3000, severity=0.4):
    """注入不对中故障（在二倍转频上叠加正弦波）"""
    fs = 1 / (10 / len(signal))  # 估算采样频率
    f_2rpm = (rpm / 60.0) * 2  # 二倍转频 (Hz)
    t = np.arange(len(signal)) / fs

    amplitude = np.std(signal) * severity * (1 + np.random.rand())

    misalignment_wave = amplitude * np.sin(2 * np.pi * f_2rpm * t + np.random.rand() * 2 * np.pi)
    return signal + misalignment_wave


# =============================================================================
# 2. 主函数：创建数据集
# =============================================================================

def create_classification_dataset(
        raw_data_path: str = "data/processed_turbine_data_with_time.csv",
        output_filename: str = "data/turbine_classification_data.npz",
        window_size: int = 1024,
        stride: int = 512,
        samples_per_class: int = 5000
):
    """
    读取原始正常数据，通过增强和故障注入，创建一个有监督的分类数据集。
    """
    print(f"🔩 开始构建分类数据集...")

    # --- 加载原始数据 ---
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"错误：找不到原始数据文件 {raw_data_path}。请先运行 generate_dataset.py。")
        return

    # 我们以1号轴承的X轴振动为例
    # 在实际应用中，您可以选择多个或组合的信号
    vibration_signal = df['rotor_Vibration1X'].values
    print(f"   - 已加载原始信号，总长度: {len(vibration_signal)}")

    # --- 将长时序信号切分成窗口样本 ---
    windows = []
    for i in range(0, len(vibration_signal) - window_size + 1, stride):
        windows.append(vibration_signal[i:i + window_size])

    # 确保我们有足够的窗口来生成数据
    if len(windows) < samples_per_class:
        print(f"警告：原始信号切分出的窗口数 ({len(windows)}) 少于目标样本数 ({samples_per_class})。")
        print("建议生成一个更大的原始数据集。")
        samples_per_class = len(windows)

    windows = np.array(windows)
    np.random.shuffle(windows)
    print(f"   - 原始信号已切分为 {len(windows)} 个样本窗口")

    all_data = []
    all_labels = []

    # --- 生成各个类别的数据 ---
    print(f"   - 正在为每个类别生成 {samples_per_class} 个样本...")

    # 类别 0: 正常 (通过数据增强)
    print("     - 生成 类别 0: 正常...")
    for i in tqdm(range(samples_per_class)):
        base_signal = windows[i % len(windows)]  # 循环使用窗口
        augmented = augment_signal(base_signal)
        all_data.append(augmented)
        all_labels.append(0)

    # 类别 1: 不平衡故障
    print("     - 生成 类别 1: 不平衡故障...")
    for i in tqdm(range(samples_per_class)):
        base_signal = windows[i % len(windows)]
        faulty = inject_imbalance_fault(base_signal, severity=np.random.uniform(0.3, 0.6))
        all_data.append(faulty)
        all_labels.append(1)

    # 类别 2: 不对中故障
    print("     - 生成 类别 2: 不对中故障...")
    for i in tqdm(range(samples_per_class)):
        base_signal = windows[i % len(windows)]
        faulty = inject_misalignment_fault(base_signal, severity=np.random.uniform(0.4, 0.7))
        all_data.append(faulty)
        all_labels.append(2)

    # --- 保存最终的有监督数据集 ---
    X = np.array(all_data)
    y = np.array(all_labels)

    # 对数据进行标准化
    # 在实际应用中，通常会保存未标准化的数据，在加载时再进行标准化
    # 这里为了方便，我们直接处理
    X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

    print(f"\n💾 正在保存最终的有监督数据集到: {output_filename}")
    # 使用 .npz 格式保存，对于numpy数组更高效
    np.savez_compressed(output_filename, data=X, labels=y)

    print("\n✅ 有监督分类数据集构建完毕！")
    print(f"   - 总样本数: {len(y)}")
    print(f"   - 特征维度: {X.shape}")
    print(f"   - 标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")


if __name__ == "__main__":
    create_classification_dataset(
        samples_per_class=5000  # 您可以根据需要调整每个类别的样本量
    )