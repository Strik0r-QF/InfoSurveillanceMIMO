import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from WirelessSurveillanceEnv import *
import pandas as pd
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPOAgent:
    def __init__(self, input_dim, output_dim, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy_net = PolicyNetwork(input_dim, output_dim)
        self.value_net = ValueNetwork(input_dim)

        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(
                self.value_net.parameters()), lr=lr)
        self.policy_old = PolicyNetwork(input_dim, output_dim)
        self.policy_old.load_state_dict(self.policy_net.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        else:
            state = torch.FloatTensor([state])  # 确保 state 是一个列表或数组
        with torch.no_grad():
            action_probs = self.policy_old(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards),
                                       reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 将 memory.states 转换为 numpy.ndarray，然后再转换为 PyTorch 张量
        states_array = np.array(memory.states)
        old_states = torch.FloatTensor(states_array)

        # 将 memory.actions 直接转换为 PyTorch 张量，并指定数据类型为 long
        old_actions = torch.tensor(memory.actions, dtype=torch.long)

        # 将 memory.logprobs 中的张量堆叠成一个张量
        old_logprobs = torch.stack(memory.logprobs)

        for _ in range(self.k_epochs):
            logprobs = []
            state_values = []
            dist_entropy = []
            for state in old_states:
                action_probs = self.policy_net(state)
                action_dist = Categorical(action_probs)
                logprobs.append(action_dist.log_prob(old_actions))
                dist_entropy.append(action_dist.entropy())
                state_values.append(self.value_net(state))

            logprobs = torch.stack(logprobs)
            dist_entropy = torch.stack(dist_entropy)
            state_values = torch.cat(state_values).squeeze()

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(
                state_values, rewards) - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy_net.state_dict())


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# Hyperparameters
input_dim = 1  # 你的状态维度
output_dim = 2  # 动作维度（增加功率或减少功率）
lr = 0.01
gamma = 0.99
eps_clip = 0.2
k_epochs = 4
update_timestep = 2000

env = WirelessSurveillanceEnv()
agent = PPOAgent(input_dim, output_dim, lr, gamma, eps_clip, k_epochs)
memory = Memory()

timestep = 0
epi_rewards = []
for i_episode in range(2000):
    state = env.reset()
    epi_reward = 0
    for t in range(0, 500):
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

    if i_episode % 100 == 0:  # 每 100 个 episode 保存一次
        torch.save(agent.policy_net.state_dict(),
                   f'policy_net.pth')
        torch.save(agent.value_net.state_dict(),
                   f'value_net.pth')

epi_rewards = np.array(epi_rewards)
# 将 epi_rewards 转换为 Pandas DataFrame
df = pd.DataFrame({'Episode': np.arange(1, len(epi_rewards) + 1), 'Reward': epi_rewards})
# 将 DataFrame 保存为 CSV 文件
df.to_csv('epi_rewards-lr=1e-2.csv', index=False)