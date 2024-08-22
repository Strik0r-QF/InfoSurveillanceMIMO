import numpy as np
import pandas as pd
from utils import *
from matplotlib import pyplot as plt

# dir = "results/8/"
# csv_files = [
#     dir + "8-4.csv",
#     dir + "8-5.csv",
#     dir + "8-7.csv",
#     dir + "8-6.csv",
#     # dir + "3-2.csv",
#     # dir + "3-3.csv",
# ]
# episode_reward_list = []
#
# for csv_file in csv_files:
#     df = pd.read_csv(csv_file)
#     episode_reward_list.append(df["Reward"].to_numpy())
#
# plot_reward(rewards_list=episode_reward_list,
#             labels=[
#                 r"lr = 0.05",
#                 r"lr = 0.01",
#                 r"lr = 0.005",
#                 r"lr = 0.001",
#                 # r"lr = 0.05",
#                 # r"lr = 0.01",
#             ],
#             sm=100,
#             title="Episode Rewards with Learning Rate",
#             )

csv_file = "results/8/8-5.csv"
df = pd.read_csv(csv_file)
series = df["Reward"]
rolling_mean = series.rolling(window=300).mean()

fig = plt.figure(figsize=(12, 9))
plt.plot(rolling_mean)
plt.show()
