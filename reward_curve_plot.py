import numpy as np
import pandas as pd
from utils import *

dir = "results/7/"
csv_files = [
    # dir + "epi_rewards-lr=1e-3.csv",
    # "results/3/3-1.csv",
    dir + "7-2.csv",
]
episode_reward_list = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    episode_reward_list.append(df["Reward"].to_numpy()[:50])

plot_reward(rewards_list=episode_reward_list,
            labels=[
                r"Reward",
                # r"$\gamma=0.9$",
                # "lr=0.005",
                # "lr=0.01",
            ],
            sm=5,
            title="Episode Rewards with Learning Rate",
            )
