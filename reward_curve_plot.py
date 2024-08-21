import numpy as np
import pandas as pd
from utils import *

csv_file = "2 LearningAlgorithms/epi_rewards-lr=1e-2.csv"
df = pd.read_csv(csv_file)

# print(df.head())

# 提取 'Reward' 列的数据并转换为 NumPy 数组
epi_rewards = df['Reward'].to_numpy()

plot_reward([epi_rewards], sm=550)
