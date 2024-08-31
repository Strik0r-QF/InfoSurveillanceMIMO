from datetime import datetime
from WirelessSurveillanceEnv import *
from SAC import SAC
from utils import *

# 初始化环境和SAC代理

env = WirelessSurveillanceEnv()
sac_agent = SAC(env, gamma=0.5, tau=0.005, alpha=0.2,
                lr=5e-4, buffer_capacity=2000, batch_size=128)

# 开始训练
episode_rewards = sac_agent.train(num_episodes=2000)

episode_rewards.to_csv(f"results/episode_rewards_{current_time()}.csv")