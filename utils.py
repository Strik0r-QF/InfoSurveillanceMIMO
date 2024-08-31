import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from datetime import datetime
import pandas as pd

# 设置 TeX
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['text.usetex'] = True

def plot_sequence(sequences,
                  labels=None,
                  xlabel=None,
                  ylabel=None,
                  filename=None):
    plt.figure(figsize=(12, 9))
    if labels is not None:
        for sequence, label in zip(sequences, labels):
            plt.plot(sequence, label=label, linewidth=2)
    else:
        for sequence in sequences:
            plt.plot(sequence, linewidth=2)
    if labels is not None:
        plt.legend(fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.grid(True)


    if filename is not None:
        plt.savefig("pic/"+filename)
    plt.show()

def plot_function(independent_varaiable,
                  functions,
                  labels=None,
                  xlabel=None,
                  ylabel=None,
                  filename=None):
    plt.figure(figsize=(12, 9))
    if labels is not None:
        for function, label in zip(functions, labels):
            y = function(independent_varaiable)
            plt.plot(independent_varaiable, y,
                     label=label, linewidth=2)
    else:
        for function in functions:
            y = function(independent_varaiable)
            plt.plot(independent_varaiable, y, linewidth=2)
    if labels is not None:
        plt.legend(fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)


    if filename is not None:
        plt.savefig("pic/" + filename)
    plt.show()


import numpy as np


def smooth(data, sm=1):
    """平滑数据的函数，同时返回平滑后的数据和窗口内的标准差"""
    if sm < 1:
        raise ValueError("Smoothing parameter 'sm' must be at least 1.")

    data = np.array(data)  # 确保 data 是 numpy 数组
    smoothed_data = np.zeros(len(data))
    std_devs = np.zeros(len(data))

    # 处理前 sm 个数据点，窗口大小从 1 逐渐增大
    for i in range(sm):
        window = data[:i + int(sm / 10)]
        smoothed_data[i] = np.mean(window)
        std_devs[i] = np.std(window)


    # 处理中间部分，窗口大小固定为 sm
    for i in range(sm, len(data)):
        window = data[i - sm:i]
        smoothed_data[i] = np.mean(window)
        std_devs[i] = np.std(window)

    return smoothed_data, std_devs


def plot_reward(rewards_list,
                labels=None,
                sm=10,
                title=None,
                filename=None):
    plt.figure(figsize=(12, 9))

    if labels is None:
        for rewards in rewards_list:
            smoothed_rewards, std_devs = smooth(rewards, sm=sm)
            plt.plot(smoothed_rewards, linewidth=2)

            plt.fill_between(
                range(len(smoothed_rewards)),
                smoothed_rewards - std_devs,
                smoothed_rewards + std_devs,
                alpha=0.2,
            )
    else:
        for rewards, label in zip(rewards_list, labels):
            smoothed_rewards, std_devs = smooth(rewards, sm=sm)
            plt.plot(smoothed_rewards,
                     label=label,
                     linewidth=3)

            plt.fill_between(
                range(len(smoothed_rewards)),
                smoothed_rewards - std_devs,
                smoothed_rewards + std_devs,
                alpha=0.2,
            )

    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)

    if title is not None:
        plt.title(title, fontsize=20)

    plt.grid(True)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    if filename is not None:
        plt.savefig("/pic/" + filename)

    plt.show()

def plot_smoothed_reward(rewards_list,
                labels=None,
                sm=10,
                title=None,
                filename=None):
    plt.figure(figsize=(12, 9))

    if labels is None:
        for rewards in rewards_list:
            smoothed_rewards = rewards.rolling(window=sm).mean()
            rewards_std = rewards.rolling(window=sm).std()
            plt.plot(smoothed_rewards, linewidth=2)

            plt.fill_between(
                range(len(smoothed_rewards)),
                smoothed_rewards - rewards_std,
                smoothed_rewards + rewards_std,
                alpha=0.2,
            )
    else:
        for rewards, label in zip(rewards_list, labels):
            smoothed_rewards = rewards.rolling(window=sm).mean()
            rewards_std = rewards.rolling(window=sm).std()
            plt.plot(smoothed_rewards, linewidth=2,
                    label=label)

            plt.fill_between(
                range(len(smoothed_rewards)),
                smoothed_rewards - rewards_std,
                smoothed_rewards + rewards_std,
                alpha=0.2,
            )
        plt.legend(fontsize=20)

    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)

    if title is not None:
        plt.title(title, fontsize=20)

    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    if filename is not None:
        plt.savefig("/pic/" + filename)

    plt.show()

def random_noise(size):
    sigma2 = 20e6 * 1.02 * 10e-12
    real = np.random.normal(0, sigma2, size=size)
    imag = np.random.normal(0, sigma2, size=size)
    return real + 1j * imag

def generate_complex_vector_with_trace_limit(n, T_max):
    """
    生成一个长度为 n 的复数列向量，使其自相关矩阵的迹不超过 T_max
    :param n: 向量长度
    :param T_max: 自相关矩阵的最大迹
    :return: 复数列向量
    """
    # Step 1: 随机生成一个复数列向量
    x = np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)

    # Step 2: 计算自相关矩阵的迹
    trace_x = np.trace(
        np.dot(x[:, np.newaxis], np.conjugate(x[:, np.newaxis]).T))

    # Step 3: 如果迹超过 T_max，调整复向量的幅度
    if trace_x > T_max:
        scale_factor = np.sqrt(T_max / trace_x)
        x = x * scale_factor

    return x

def generate_complex_channel_matrix(prev_H, num_rx, num_tx,
                                    alpha=0.9):
    """
    生成带有时间相关性的复信道矩阵
    :param prev_H: 上一个时刻的信道矩阵
    :param num_rx: 接收天线数量
    :param num_tx: 发送天线数量
    :param alpha: 时间相关性参数，取值范围在 (0, 1) 之间
    :return: 复信道矩阵
    """
    if prev_H is None:
        # 如果没有前一个信道矩阵，随机初始化一个
        real_part = np.random.normal(0, 1 / np.sqrt(2),
                                     (num_rx, num_tx))
        imag_part = np.random.normal(0, 1 / np.sqrt(2),
                                     (num_rx, num_tx))
        return real_part + 1j * imag_part

    # 生成新的信道矩阵
    real_part = np.random.normal(0, 1 / np.sqrt(2),
                                 (num_rx, num_tx))
    imag_part = np.random.normal(0, 1 / np.sqrt(2),
                                 (num_rx, num_tx))
    new_H = real_part + 1j * imag_part

    # 使用一阶自回归模型更新信道矩阵
    H = alpha * prev_H + np.sqrt(1 - alpha ** 2) * new_H
    return H

class MIMO:
    def __init__(self, num_rx=3, num_tx=3, max_power=10):
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.max_power = max_power

def current_time():
    # 获取当前时间
    now = datetime.now()
    # 格式化时间
    formatted_time = now.strftime('%Y%m%d-%H:%M')
    return formatted_time
