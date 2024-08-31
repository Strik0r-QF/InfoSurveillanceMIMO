from utils import *

dir = "2 LearningAlgorithms/SAC/results/"
csv_files = [
    dir + "episode_rewards_20240830-15:59.csv",
]
episode_reward_list = []
#
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    print(df.head())
    episode_reward_list.append(df["reward"])

plot_smoothed_reward(rewards_list=episode_reward_list,
                     # labels=[
                     #    # r"lr=0.05",
                     #    r"lr=0.01",
                     #    # r"lr=0.005",
                     #    # r"lr=0.001",
                     # ],
                     sm=500)

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


