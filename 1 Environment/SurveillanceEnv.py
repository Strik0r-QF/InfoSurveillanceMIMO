import gym
import numpy as np
from gym import spaces

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

def generate_complex_channel_matrix(prev_H, num_rx, num_tx, alpha=0.9):
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
        real_part = np.random.normal(0, 1 / np.sqrt(2), (num_rx, num_tx))
        imag_part = np.random.normal(0, 1 / np.sqrt(2), (num_rx, num_tx))
        return real_part + 1j * imag_part

    # 生成新的信道矩阵
    real_part = np.random.normal(0, 1 / np.sqrt(2), (num_rx, num_tx))
    imag_part = np.random.normal(0, 1 / np.sqrt(2), (num_rx, num_tx))
    new_H = real_part + 1j * imag_part

    # 使用一阶自回归模型更新信道矩阵
    H = alpha * prev_H + np.sqrt(1 - alpha ** 2) * new_H
    return H

class MIMO:
    def __init__(self, num_rx=3, num_tx=3, max_power=10):
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.max_power = max_power

class SurveillanceEnv(gym.Env):
    def __init__(self,
                 src=MIMO(0, 5),
                 dst=MIMO(5, 0),
                 eve=MIMO(5, 5), ):
        self.src = src
        self.dst = dst
        self.eve = eve

        self.action_space = spaces.Box(
            low=-self.eve.num_tx,
            high=self.eve.num_tx,
            shape=(2 * self.eve.num_tx,),  # 实部和虚部分开
            dtype=np.float32  # 确保使用 float32
        )

        # Observation Space
        # [H_SD, H_SE, H_ED, H_EE, signal_D, noise_D, signal_E, noise_E]
        self.observation_space = spaces.Box(
            low=-4,
            high=4,
            shape=(16, 5, 5),
            dtype=np.float32,
        )

        self.sigma2 = 20e6 * 1.02 * 10e-12

        self.reset()

    def reset(self):
        self.state = np.zeros(shape=(16, 5, 5))

        # 初始化前一时刻的信道矩阵
        self.prev_H_SD = None
        self.prev_H_ED = None
        self.prev_H_SE = None
        self.prev_H_EE = None

        return self.state

    def step(self, action):
        fwd_eve_real = action[:self.eve.num_tx]
        fwd_eve_imag = action[self.eve.num_tx:]
        fwd_eve = fwd_eve_real + 1j * fwd_eve_imag

        if all(fwd_eve == 0.+0.j):
            fwd_eve = np.array(
                [0.1 + 0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]
            )

        tr = np.matmul(fwd_eve, np.conjugate(fwd_eve).T)

        if tr > self.eve.max_power:
            scale_factor = np.sqrt(self.eve.max_power / tr)
            fwd_eve = fwd_eve * scale_factor

        fwd_src = generate_complex_vector_with_trace_limit(
            5, self.src.max_power
        )

        H_SD = generate_complex_channel_matrix(
            self.state[0] + 1j * self.state[1],
            self.dst.num_rx, self.src.num_tx,
            alpha=0.9
        )
        self.state[0] = H_SD.real
        self.state[1] = H_SD.imag

        H_SE = generate_complex_channel_matrix(
            self.state[2] + 1j * self.state[3],
            self.eve.num_rx, self.src.num_tx,
            alpha=0.9
        )
        self.state[2] = H_SE.real
        self.state[3] = H_SE.imag

        H_ED = generate_complex_channel_matrix(
            self.state[4] + 1j * self.state[5],
            self.dst.num_rx, self.eve.num_tx,
            alpha=0.9
        )
        self.state[4] = H_ED.real
        self.state[5] = H_ED.imag

        H_EE = generate_complex_channel_matrix(
            self.state[6] + 1j * self.state[7],
            self.eve.num_rx, self.eve.num_tx,
            alpha=0.9
        )
        self.state[6] = H_EE.real
        self.state[7] = H_EE.imag

        signal_dst = np.matmul(H_SD, fwd_src)
        cov_signal_dst = signal_dst @ np.conjugate(signal_dst).T
        self.state[8] = cov_signal_dst.real
        self.state[9] = cov_signal_dst.imag

        signal_dst_power = sum(
            fwd_eve * np.conjugate(fwd_eve)
        )

        noise_dst = np.matmul(H_SE, fwd_eve) + random_noise(self.dst.num_rx)
        cov_noise_dst = noise_dst @ np.conjugate(noise_dst).T
        self.state[10] = cov_noise_dst.real
        self.state[11] = cov_noise_dst.imag

        noise_dst_power = sum(
            noise_dst * np.conjugate(noise_dst)
        )

        signal_eve = np.matmul(H_SE, fwd_src)
        cov_signal_eve = signal_eve @ np.conjugate(signal_eve).T
        self.state[12] = cov_signal_eve.real
        self.state[13] = cov_signal_eve.imag

        signal_eve_power = sum(
            signal_eve * np.conjugate(signal_eve)
        )

        noise_eve = np.matmul(H_EE, fwd_eve) + random_noise(self.eve.num_rx)
        cov_noise_eve = noise_eve @ np.conjugate(noise_eve).T
        self.state[14] = cov_noise_eve.real
        self.state[15] = cov_noise_eve.imag

        noise_eve_power = sum(
            noise_eve * np.conjugate(noise_eve)
        )

        capacity_dst = np.log2(
            1 + signal_dst_power / noise_dst_power
        )

        capacity_eve = np.log2(
            1 + signal_eve_power / noise_eve_power
        )

        reward = capacity_dst if capacity_eve >= capacity_dst else 0

        done = False

        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

