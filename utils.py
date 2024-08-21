import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from webencodings import labels

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
    """平滑数据的函数"""
    if sm < 1:
        raise ValueError("Smoothing parameter 'sm' must be at least 1.")

    if sm == 1:
        return np.array(data)  # 如果 sm 为 1，直接返回原始数据的数组形式

    data = np.array(data)  # 确保 data 是 numpy 数组
    z = np.ones(len(data))
    y = np.ones(sm) * 1.0

    numerator = np.convolve(y, data, mode="same")
    denominator = np.convolve(y, z, mode="same")
    smoothed_data = numerator / denominator

    return smoothed_data  # 返回为一维数组形式



def plot_reward(rewards_list,
                labels=None,
                sm=10,
                title=None,
                filename=None):
    plt.figure(figsize=(12, 9))

    if labels is None:
        for rewards in rewards_list:
            smoothed_rewards = smooth(rewards, sm=sm)
            plt.plot(smoothed_rewards, linewidth=2)

            # 添加阴影处理
            std = np.std(smoothed_rewards)
            mean = np.mean(smoothed_rewards)
            plt.fill_between(
                range(len(smoothed_rewards)),
                smoothed_rewards - std,
                smoothed_rewards + std,
                alpha=0.2,
            )
    else:
        for rewards, label in zip(rewards_list, labels):
            smoothed_rewards = smooth(rewards, sm=sm)
            plt.plot(smoothed_rewards,
                     label=label,
                     linewidth=2)
            # 添加阴影处理
            std = np.std(smoothed_rewards)
            mean = np.mean(smoothed_rewards)
            plt.fill_between(
                range(len(smoothed_rewards)),
                smoothed_rewards - std,
                smoothed_rewards + std,
                alpha=0.2,
            )

    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)

    # title
    if title is not None:
        plt.title(title, fontsize=20)

    plt.grid(True)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # save the picture
    if filename is not None:
        plt.savefig("/pic/" + filename)

    plt.show()
