# MIMO 系统的 Proactive-Jamming 监听

项目结构:
- `1 Environment` 文件夹存放了使用的环境, `WirelessSurveillanceEnv` 是当前正在使用的环境.
- `2 LearningAlgorithms` 文件夹存放了使用的学习方法, 目前正在使用的是 PPO 算法.
- `3 TrainingCode` 文件夹存放了训练代码 (适用于一些分离了算法实现与训练循环的情况, 如 DDPG). 目前没有活跃文件.
- `utils.py` 中存放了一些常用的函数, 目前主要是绘图函数.