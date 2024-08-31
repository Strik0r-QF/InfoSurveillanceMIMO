import numpy as np
from torch import optim as optim, nn as nn
from torch.distributions import Categorical
from PolicyNetwork import *
from ValueNetwork import *


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
