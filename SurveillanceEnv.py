import gym
import numpy as np
from EveActionSpace import EveActionSpace
from gym.spaces import Box, Dict

class MIMO:
    def __init__(self, num_rx=3, num_tx=3, max_power=10):
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.max_power = max_power

class SurveillanceEnv(gym.Env):
    def __init__(self,
                 src=MIMO(0, 3),
                 dst=MIMO(3, 0),
                 eve=MIMO(5, 5), ):
        self.src = src
        self.dst = dst
        self.eve = eve

        self.action_space = EveActionSpace(self.eve)

        self.observation_space = Dict({
            "HSD": Box(low=-np.inf, high=np.inf,
                       shape=(dst.num_rx, src.num_tx),
                       dtype=np.complex128),
            "HSE": Box(low=-np.inf, high=np.inf,
                       shape=(eve.num_rx, src.num_tx),
                       dtype=np.complex128),
            "HED": Box(low=-np.inf, high=np.inf,
                       shape=(dst.num_rx, eve.num_tx),
                       dtype=np.complex128),
            "HEE": Box(low=-np.inf, high=np.inf,
                       shape=(eve.num_rx, eve.num_tx),
                       dtype=np.complex128),
            "signal D": Box(low=-np.inf, high=np.inf,
                       shape=(dst.num_rx, dst.num_rx),
                       dtype=np.complex128),
            "noise D": Box(low=-np.inf, high=np.inf,
                       shape=(dst.num_rx, dst.num_rx),
                       dtype=np.complex128),
            "signal E": Box(low=-np.inf, high=np.inf,
                       shape=(eve.num_rx, dst.num_rx),
                       dtype=np.complex128),
            "noise E": Box(low=-np.inf, high=np.inf,
                       shape=(eve.num_rx, eve.num_rx),
                       dtype=np.complex128),
        })
