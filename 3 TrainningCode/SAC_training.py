# import gym
# from WirelessSurveillanceEnv import *
# from SAC import *
# import pandas as pd
# from datetime import datetime
#
#
# def train_sac_agent(env, agent, replay_buffer, episodes=1000,
#                     batch_size=256):
#     for episode in range(episodes):
#         state = env.reset()
#         episode_reward = 0
#         done = False
#
#         while not done:
#             action = agent.select_action(state)
#             next_state, reward, done, _ = env.step(action)
#             replay_buffer.add(state, action, reward, next_state, done)
#             state = next_state
#             episode_reward += reward
#
#             # 如果replay buffer中有足够的样本，则开始训练
#             if len(replay_buffer) > batch_size:
#                 # 在每次训练之前重新实例化优化器，防止变量不一致的问题
#                 agent.critic_optimizer = tf.keras.optimizers.Adam(
#                     learning_rate=0.001)
#                 agent.actor_optimizer = tf.keras.optimizers.Adam(
#                     learning_rate=0.001)
#
#                 agent.train(replay_buffer, batch_size)
#
#         print(f"Episode {episode + 1}: Reward = {episode_reward}")
#
#     return episode_rewards
#
#
# # 创建环境和智能体
# env = WirelessSurveillanceEnv()
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# max_action = env.action_space.high[0]
#
#
#
# agent = SACAgent(state_dim, action_dim, max_action,
#                  lr=3e-4,
#                  discount=0.99,
#                  tau=0.001)
# replay_buffer = ReplayBuffer(max_size=2000)
#
# # 开始训练
# episode_rewards = train_sac_agent(env, agent, replay_buffer, episodes=1000, batch_size=256)
# series = pd.Series(episode_rewards)
# # 获取当前时间
# now = datetime.now()
#
# # 格式化时间
# formatted_time = now.strftime('%Y/%m/%d-%H:%M')
#
# series.to_csv(f'SAC_training_series-{formatted_time}.csv')