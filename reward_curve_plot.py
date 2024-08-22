import numpy as np
import pandas as pd
from utils import *
from matplotlib import pyplot as plt

dir = "results/9/"
csv_files = [
    dir + "9-1.csv",
    # dir + "8-5.csv",
    # dir + "8-7.csv",
    # dir + "8-6.csv",
#     # dir + "3-2.csv",
#     # dir + "3-3.csv",
]
episode_reward_list = []
#
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    episode_reward_list.append(df["Reward"])
#
plot_smoothed_reward(rewards_list=episode_reward_list,
                     labels=[
                        # r"lr=0.05",
                        r"lr=0.01",
                        # r"lr=0.005",
                        # r"lr=0.001",
                     ],
                     sm=100)

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


