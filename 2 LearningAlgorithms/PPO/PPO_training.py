from PPOAgent import *
from Memory import *
from WirelessSurveillanceEnv import *
import random
import pandas as pd
from datetime import datetime


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你使用的是 CUDA
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 的操作是可复现的
    torch.backends.cudnn.benchmark = False     # 确保 CUDA 的操作是可复现的

# Hyperparameters
input_dim = 1  # 你的状态维度
output_dim = 2  # 动作维度（增加功率或减少功率）
update_timestep = 2000



# 初始化环境和代理
env = WirelessSurveillanceEnv(
                src=MIMO(0, 1),
                dst=MIMO(1, 0),
                eve=MIMO(3, 2),
)

# 设置随机种子
seed = 303324
set_seed(seed)
env.seed(seed)

learning_rates = [
    0.1,
]
discounts = [
    # 0.9,
    # 0.7,
    0.5,
]
eps_clips = [
    0.15,
    # 0.5,
    # 0.7,
]
k_epochs = [
    # 4,
    # 8,
    16,
]
count = 0

for lr in learning_rates:
    for gamma in discounts:
        for eps_clip in eps_clips:
            for epochs in k_epochs:
                print(f"Config: {lr = }, {gamma = }, {eps_clip = }, {epochs = }")
                count += 1

                agent = PPOAgent(input_dim, output_dim, lr, gamma, eps_clip, epochs)
                memory = Memory()

                timestep = 0
                epi_rewards = []
                for i_episode in range(2000):
                    state = env.reset()
                    epi_reward = 0
                    while True:
                        timestep += 1

                        action, logprob = agent.select_action(state)
                        next_state, reward, done, info = env.step(action)

                        epi_reward += reward

                        memory.states.append(state)
                        memory.actions.append(action)
                        memory.logprobs.append(logprob)
                        memory.rewards.append(reward)
                        memory.is_terminals.append(done)

                        if timestep % update_timestep == 0:
                            agent.update(memory)
                            memory.clear_memory()
                            timestep = 0

                        state = next_state

                        if done:
                            break
                    print(f"Episode: {i_episode}, Reward: {epi_reward}")
                    epi_rewards.append(epi_reward)

                    # 保存模型
                    if i_episode % 100 == 0:  # 每 100 个 episode 保存一次
                        # 获取当前时间
                        now = datetime.now()

                        # 格式化时间
                        formatted_time = now.strftime('%Y%m%d-%H:%M')

                        torch.save(agent.policy_net.state_dict(),
                                   f'policy_net-{formatted_time}.pth')
                        torch.save(agent.value_net.state_dict(),
                                   f'value_net-{formatted_time}.pth')

                # 保存训练过程的奖励数据
                epi_rewards = np.array(epi_rewards)
                # 将 epi_rewards 转换为 Pandas DataFrame
                df = pd.DataFrame({
                    'Episode': np.arange(1, len(epi_rewards) + 1),
                    'Reward': epi_rewards})

                # 获取当前时间
                now = datetime.now()

                # 格式化时间
                formatted_time = now.strftime('%Y%m%d-%H:%M')

                # 将 DataFrame 保存为 CSV 文件
                df.to_csv(f'epi_rewards-{formatted_time}.csv', index=False)

                agent.policy_net.eval()
                agent.value_net.eval()

                dummy_input = torch.Tensor(np.array([3]))  # 这里假设输入的维度是 [batch_size, input_features]
