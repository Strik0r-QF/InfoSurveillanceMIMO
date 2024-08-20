import matplotlib.pyplot as plt
from SurveillanceEnv import *
from DDPG import *
import matplotlib

# 设置随机种子
np.random.seed(810)
tf.random.set_seed(810)
random.seed(810)

# 环境初始化
env = SurveillanceEnv(
    src=MIMO(0, 5),
    dst=MIMO(5, 0),
    eve=MIMO(5, 5),
)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


num_episodes = 100
results = {}
surveillance_rate_dict = {}

# 设置不同学习率
learning_rates = [
    # 0.1,
    # 0.01,
    0.0001,
]

for lr in learning_rates:
    env.reset()
    print(f"Training with learning rate: {lr}")

    # 定义 DDPG Agent
    agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=lr,
        critic_lr=lr,
        discount=0.9,
        replay_buffer=2500,
        batch_size=128,
        tau=0.001,
    )

    # 训练 DDPG Agent
    episode_rewards, surveillance_rate = train_ddpg(env, agent, num_episodes)
    results[lr] = episode_rewards
    surveillance_rate_dict[lr] = surveillance_rate

# 保存模型
agent.actor.save_weights('final_actor.weights.h5')
agent.critic.save_weights('final_critic.weights.h5')


# 设置不同的折扣因子
# gammas = [
#     0.99,
#     0.9,
#     0.7,
# ]
# #
# for gamma in gammas:
#     print(f"Training with discount factor: {gamma}")
#     agent = DDPG(
#         state_dim,
#         action_dim,
#         max_action,
#         discount=gamma,
#         actor_lr=0.01,
#         critic_lr=0.01,
#         replay_buffer=2500,
#         batch_size=128,
#         tau=0.001,
#     )
#     episode_rewards, surveillance_rate = train_ddpg(env, agent, num_episodes)
#     results[gamma] = episode_rewards
#     surveillance_rate_dict[gamma] = surveillance_rate
#
# torch.save(agent.actor.state_dict(), 'final_actor.pth')
# torch.save(agent.critic.state_dict(), 'final_critic.pth')

# batch_sizes = [
#     32,
#     64,
#     128,
# ]

# opts = [
#     "adam",
# ]
#
# for opt in opts:
#     print(f"Training with optimizer: {opt}")
#     agent = DDPG(
#         state_dim,
#         action_dim,
#         max_action,
#         batch_size=128,
#         discount = 0.9,
#         actor_lr = 0.01,
#         critic_lr = 0.01,
#         replay_buffer=2000,
#         tau=0.001,
#         actor_optimizer= opt,
#         critic_optimizer= opt,
#     )
#     episode_rewards, surveillance_rate = train_ddpg(
#         env, agent, num_episodes
#     )
#     results[opt] = episode_rewards
#     surveillance_rate_dict[opt] = surveillance_rate
#
# torch.save(agent.actor.state_dict(), 'final_actor.pth')
# torch.save(agent.critic.state_dict(), 'final_critic.pth')




# 平滑数据的函数
def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            if isinstance(d, (float, int)):  # 如果d是浮点数或整数
                d = np.array([d])  # 转换为数组以继续处理

            z = np.ones(len(d))
            y = np.ones(sm) * 1.0

            numerator = np.convolve(y, d, mode="same")
            denominator = np.convolve(y, z, mode="same")
            d = numerator / denominator

            smooth_data.append(d)
    return smooth_data


# 设置字体和TeX
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['text.usetex'] = True

# 绘制训练曲线
plt.figure(figsize=(12, 9))
for lr, rewards in results.items():
    smoothed_rewards = smooth([rewards], sm=10)[0]
    plt.plot(smoothed_rewards, label=f'Learning rate: {lr}', linewidth=2)
    # 添加阴影处理
    std = np.std(smoothed_rewards)
    mean = np.mean(smoothed_rewards)
    plt.fill_between(
        range(len(smoothed_rewards)),
        smoothed_rewards - std,
        smoothed_rewards + std,
        alpha=0.2,
    )

plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.title('DDPG Training Rewards with Optimizer', fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("pic/DDPG_OptFi.png")
plt.show()
#
# # 设置字体和TeX
# matplotlib.rcParams['font.family'] = 'Times New Roman'
# matplotlib.rcParams['mathtext.default'] = 'regular'
# matplotlib.rcParams['text.usetex'] = True
#
# # 绘制训练曲线
# plt.figure(figsize=(12, 9))
# for lr, srate in surveillance_rate_dict.items():
#     smooth_srate = smooth([srate], sm=10)
#     plt.plot(smooth_srate, label=f'Discount Factor: {lr}', linewidth=2)
# plt.xlabel('Episode', fontsize=20)
# plt.ylabel('Surveillance Success Rate', fontsize=20)
# plt.title('Surveillance Success Rate with Discount Factor', fontsize=20)
# plt.grid(True)
# plt.legend(fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.savefig("pic/SurveillanceRate_DiscountFactorFi.png")
# plt.show()