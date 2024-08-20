from TD import *
import numpy as np
from SurveillanceEnv import *
import matplotlib.pyplot as plt
from sci_plot import *

# 设置随机种子
np.random.seed(810)
tf.random.set_seed(810)
random.seed(810)

# 环境初始化
env = SurveillanceEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# 定义 TD3 Agent
agent = TD3(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    actor_lr=0.001,
    critic_lr=0.001,
    replay_buffer=2500,
    batch_size=128,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_delay=2,
)

num_episodes = 100  # 设置训练的回合数
episode_rewards, surveillance_rate = train_td3(env, agent, num_episodes)

plot_sequence(sequence=episode_rewards,
              xlabel="Episode",
              ylabel="Reward",
              filename="TD3_rewards.png",)
