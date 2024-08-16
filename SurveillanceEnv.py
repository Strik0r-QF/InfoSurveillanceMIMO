import gym
import numpy as np
from EveActionSpace import EveActionSpace

class MIMO:
    def __init__(self, num_rx=3, num_tx=3, max_power=10):
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.max_power = max_power

class SurveillanceEnv(gym.Env):
    def __init__(self,
                 source=MIMO(0, 3),
                 destination=MIMO(3, 0),
                 eve=MIMO(5, 5),):
        self.source = source
        self.destination = destination
        self.eve = eve

        self.action_space = EveActionSpace(self.eve)