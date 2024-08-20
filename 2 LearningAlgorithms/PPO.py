import gym
from SurveillanceEnv import *
from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC

# 创建环境
env = SurveillanceEnv()

# 包装环境
env = Monitor(env)  # 记录环境的额外信息
env = DummyVecEnv([lambda: env])  # 包装成vectorized环境以适应stable-baselines3

# 定义模型
model = SAC("MlpPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 保存模型
model.save("sac_surveillance")

# 测试模型
env = Monitor(env)  # 再次包装环境以记录测试信息
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()

# 关闭环境
env.close()
