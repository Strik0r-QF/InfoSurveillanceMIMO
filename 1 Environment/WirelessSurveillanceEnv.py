import gym
import numpy as np
from gym import spaces
from SurveillanceEnv import generate_complex_channel_matrix, MIMO
import random
import torch

class WirelessSurveillanceEnv(gym.Env):
    def __init__(self,
                 src=MIMO(0, 3),
                 dst=MIMO(1, 0),
                 eve=MIMO(3, 1), ):
        self.src = src
        self.dst = dst
        self.eve = eve

        # 动作空间：动作为一个连续值，表示噪声功率的增减幅度
        self.action_space = spaces.Box(
            low=np.array([-0.1]),  # 最小可减少的功率
            high=np.array([0.1]),  # 最大可增加的功率
            dtype=np.float32
        )

        # Observation space: Eve只能观察 [SNR_E]
        self.observation_space = spaces.Box(
            low=np.array([0]),
            high=np.array([np.inf])
        )

        self.sigma2 = 20e6 * 1.02 * 10e-12

        self.reset()

    def reset(self):
        self.state = np.zeros(shape=(6,))
        self.state[0] = np.random.uniform(low=0, high=10)
        self.eve_power = 0

        self.prev_H_SD = generate_complex_channel_matrix(
            None,
            self.dst.num_rx, self.src.num_tx,
            alpha=0.9
        )
        self.prev_H_ED = generate_complex_channel_matrix(
            None,
            self.dst.num_rx, self.eve.num_tx,
            alpha=0.9
        )
        self.prev_H_SE = generate_complex_channel_matrix(
            None,
            self.eve.num_rx, self.src.num_tx,
            alpha=0.9
        )
        self.prev_H_EE = generate_complex_channel_matrix(
            None,
            self.eve.num_rx, self.eve.num_tx,
            alpha=0.9
        )

        self.communicate = True
        self.count = 0

        return self._get_observation()

    def step(self, action):
        self.count += 1

        # 动作决定 eve_power 的增减幅度
        self.eve_power += action
        self.eve_power = np.clip(self.eve_power,
                                 0, self.eve.max_power)

        if not self.communicate:
            scale = np.random.uniform(1.05, 1.1)
        else:
            scale = np.random.uniform(0.95, 1.05)
        self.state[0] = np.clip(
            scale * self.state[0],
            0, 10,
        )

        H_SD = generate_complex_channel_matrix(
            self.prev_H_SD,
            self.dst.num_rx, self.src.num_tx,
            alpha=0.9
        )
        h_SD = np.linalg.norm(H_SD, ord='fro') ** 2
        self.state[2] = h_SD

        H_SE = generate_complex_channel_matrix(
            self.prev_H_SE,
            self.eve.num_rx, self.src.num_tx,
            alpha=0.9
        )
        h_SE = np.linalg.norm(H_SE, ord='fro') ** 2
        self.state[3] = h_SE

        H_ED = generate_complex_channel_matrix(
            self.prev_H_ED,
            self.dst.num_rx, self.eve.num_tx,
            alpha=0.9
        )
        h_ED = np.linalg.norm(H_ED, ord='fro') ** 2
        self.state[4] = h_ED

        H_EE = generate_complex_channel_matrix(
            self.prev_H_EE,
            self.eve.num_rx, self.eve.num_tx,
            alpha=0.9
        )
        h_EE = np.linalg.norm(H_EE, ord='fro') ** 2
        self.state[5] = h_EE

        SNR_E = h_SE * self.state[0] / (self.sigma2 + h_EE * self.eve_power)
        self.state[1] = SNR_E
        SNR_D = h_SD * self.state[0] / (self.sigma2 + h_ED * self.eve_power)

        capacity_E = np.log2(1 + SNR_E)
        capacity_D = np.log2(1 + SNR_D)
        self.communicate = capacity_D >= 2

        reward = capacity_D if self.communicate else 0
        done = (self.count == 500
                # or (self.state[0] == self.src.max_power
                #     and not self.communicate)
                )

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # Eve 只能观察 SNR_E
        return np.array([self.state[1]])

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    # def seed(self, seed=None):
    #     """设置环境的随机种子"""
    #     if seed is not None:
    #         random.seed(seed)
    #         np.random.seed(seed)
    #         # 如果你的环境中使用其他随机性库，例如 torch，也需要设置种子
    #         torch.manual_seed(seed)
