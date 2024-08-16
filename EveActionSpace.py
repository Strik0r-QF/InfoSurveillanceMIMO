import numpy as np
import gym
from SurveillanceEnv import MIMO


class EveActionSpace(gym.Space):
    def __init__(self, end_system: MIMO):
        self.end_system = end_system
        self.n = end_system.num_tx
        self.max_power = end_system.max_power
        super().__init__(shape=(self.n, self.n),
                         dtype=np.float64,)

    def sample(self):
        while True:
            diag_elem = np.random.uniform(low=0,
                                          high=self.max_power,
                                          size=self.n,)
            if np.sum(diag_elem) <= self.max_power:
                return np.diag(diag_elem)

    def contains(self, x):
        if isinstance(x, np.ndarray) and x.shape == (self.n, self.n):
            diag_elem = np.diag(x)
            result = (np.all(diag_elem >= 0)
                      and np.sum(diag_elem) <= self.max_power)
            return result
        return False

