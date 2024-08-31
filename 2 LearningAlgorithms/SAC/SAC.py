import torch.nn as nn
import torch.optim as optim
from collections import deque
from WirelessSurveillanceEnv import *
from utils import *

import torch

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()

# 检查 MPS 是否可用（仅适用于支持的 Apple 设备）
mps_available = False
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    mps_available = True

# 根据可用性选择设备
if cuda_available:
    device = torch.device("cuda")
    print("使用 GPU (CUDA)")
elif mps_available:
    device = torch.device("mps")
    print("使用 GPU (MPS)")
else:
    device = torch.device("cpu")
    print("使用 CPU")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.stack(action), reward, np.stack(next_state), done

    def size(self):
        return len(self.buffer)

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(GaussianPolicy, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, num_actions)
        self.fc_logstd = nn.Linear(256, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = torch.clamp(self.fc_logstd(x), min=-20, max=2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        return action, z, normal.log_prob(z)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC:
    def __init__(self, env,
                 gamma=0.99, tau=0.005, alpha=0.2,
                 lr=3e-4, buffer_capacity=100000, batch_size=256):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.policy = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0])
        self.q1 = QNetwork(env.observation_space.shape[0], env.action_space.shape[0])
        self.q2 = QNetwork(env.observation_space.shape[0], env.action_space.shape[0])
        self.target_q1 = QNetwork(env.observation_space.shape[0], env.action_space.shape[0])
        self.target_q2 = QNetwork(env.observation_space.shape[0], env.action_space.shape[0])

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Initialize target networks with the same weights as original networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # 确保网络移动到正确设备
        self.policy.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.target_q1.to(self.device)
        self.target_q2.to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size)

        # 将列表中的ndarray转换为一个单一的NumPy数组，然后再转换为PyTorch张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(
            self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(
            self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(
            self.device)

        with torch.no_grad():
            next_actions, next_z, next_log_pis = self.policy.sample(next_states)
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * (torch.min(target_q1, target_q2) - self.alpha * next_log_pis)

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = torch.mean((q1 - target_q) ** 2)
        q2_loss = torch.mean((q2 - target_q) ** 2)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, z, log_pis = self.policy.sample(states)
        q_new_actions = torch.min(self.q1(states, new_actions), self.q2(states, new_actions))
        policy_loss = torch.mean(self.alpha * log_pis - q_new_actions)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'target_q1_state_dict': self.target_q1.state_dict(),
            'target_q2_state_dict': self.target_q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.target_q1.load_state_dict(
            checkpoint['target_q1_state_dict'])
        self.target_q2.load_state_dict(
            checkpoint['target_q2_state_dict'])
        self.policy_optimizer.load_state_dict(
            checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(
            checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(
            checkpoint['q2_optimizer_state_dict'])

    def train(self, num_episodes):
        episode_rewards = pd.DataFrame(columns=['episode', 'reward'])

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward,
                                        next_state, done)

                state = next_state
                episode_reward += reward

                self.update()

                if done:
                    break

            print(f"Episode: {episode}, Reward: {episode_reward}")

            if episode % 100 == 0:
                formatted_time = current_time()
                self.save_model(f"results/SAC_{formatted_time}_epi{episode}.pth")

            # Append the new episode's reward data to the DataFrame
            new_reward_data = pd.DataFrame([{
                "episode": episode,
                "reward": float(
                    episode_reward.item() if hasattr(episode_reward,'item') else episode_reward),
            }])

            if episode_rewards.empty:
                episode_rewards = new_reward_data
            else:
                episode_rewards = pd.concat(
                    [episode_rewards, new_reward_data],
                    ignore_index=True)

        self.save_model(f"results/SAC_Final{current_time()}.pth")

        return episode_rewards



