from WirelessSurveillanceEnv import *
from TD3 import *
import pandas as pd
from datetime import datetime

def training_td3_agent(episodes=500):
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            state = state.reshape(1, -1)
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state,
                                   done)

            state = next_state
            episode_reward += reward
            agent.learn()

            if done:
                print(f"Episode: {episode}, Reward: {episode_reward}")
                episode_rewards.append(episode_reward)
                break

    agent.actor_model.save_weights("actor_model.weights.h5")
    agent.critic_model_1.save_weights(
        "critic_model_1.weights.h5")
    agent.critic_model_2.save_weights(
        "critic_model_2.weights.h5")

    # 创建一个 DataFrame，其中包含 Episode、Step 和 Reward 列
    episode_rewards_df = pd.Series(episode_rewards)
    return episode_rewards_df

# Training
env = WirelessSurveillanceEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

agent = TD3Agent(state_dim, action_dim, action_bound,
                 gamma=0.9, tau=0.005, lr=1e-2,
                 replay_buffer=2000, batch_size=64)

rewards = training_td3_agent(episodes=500)
print(rewards.head())

# 获取当前时间
now = datetime.now()

# 格式化时间
formatted_time = now.strftime('%Y%m%d-%H:%M')

rewards.to_csv(f'TD3_rewards-{formatted_time}.csv')
