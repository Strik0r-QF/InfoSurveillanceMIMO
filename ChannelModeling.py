import numpy as np


def channel_matrix(num_rx: int, num_tx: int) -> np.ndarray:
    """
    生成信道矩阵, 假设信道矩阵中所有元素都服从复高斯分布 CN(0, 1)
    :param num_rx: 接收机天线数量
    :param num_tx: 发射机天线数量
    :return: 信道矩阵 H
    """
    real = np.random.normal(0, 1, size=[num_rx, num_tx])
    imag = np.random.normal(0, 1, size=[num_rx, num_tx])
    return real + 1j * imag



