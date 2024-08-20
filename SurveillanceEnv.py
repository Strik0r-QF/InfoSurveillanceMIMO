import gym
import numpy as np
from EveActionSpace import EveActionSpace
from gym.spaces import Box, Dict
from gym import spaces

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

        self.action_space = EveActionSpace(self.eve)

        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (8, 5, 5),
            dtype = np.complex128,
        )

        self.reset()

    def reset(self):
        diag_elem = np.zeros(shape=self.src.num_tx)
        self.cov_src = np.diag(diag_elem)
        diag_elem = np.zeros(shape=self.eve.num_tx)
        self.cov_eve = np.diag(diag_elem)


