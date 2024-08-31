from SAC import *
from WirelessSurveillanceEnv import *
import numpy as np
import pandas as pd
from datetime import datetime

env = WirelessSurveillanceEnv()
agent = SAC(env)
agent.load_model('results/SAC_Final.pth')

# 预先创建一个列表来存储每一行的数据
data = []

count = 0
state = agent.env.reset()
episode_reward = 0

while True:
    count += 1
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    episode_reward += reward

    SNR_E = info['SNR_E']
    n_E = info['n_E']
    SNR_D = info['SNR_D']
    P_S = info['P_S']

    # 将每一行的数据存储在字典中，然后加入列表
    data.append({
        "time": count,
        "SNR_E": SNR_E[0],
        "SNR_D": SNR_D[0],
        "n_E": n_E[0],
        "P_S": P_S,
        "reward": reward[0],
    })

    state = next_state  # 更新 state

    if done:
        break



# 循环结束后，将列表转换为 DataFrame
dataframe = pd.DataFrame(data)
dataframe.to_csv(f'results/SAC_eval_result-{current_time()}.csv', index=False)

print(f"Total episode reward: {episode_reward}")
