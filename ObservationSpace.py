import gym
import numpy as np
from gym.spaces import Box, Dict
from SurveillanceEnv import MIMO

class ObervationSpace(gym.Space):
    def __init__(self, src=MIMO(0, 3),
                 dst=MIMO(3, 0),
                 eve=MIMO(5, 5)):
        self.matrix_shapes = [
            (dst.num_rx, src.num_tx), # HSD
            (eve.num_rx, src.num_tx), # HSE
            (dst.num_rx, eve.num_tx), # HED
            (eve.num_rx, eve.num_tx), # HEE
            (dst.num_rx, dst.num_rx), # signal D
            (dst.num_rx, dst.num_rx), # noise D
            (eve.num_rx, dst.num_rx), # signal E
            (eve.num_rx, eve.num_rx), # noise E
        ]