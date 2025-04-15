# Independent RL 实现

## 算法特点
- 每个智能体独立学习和决策
- 将多智能体问题转化为多个单智能体问题
- 不考虑其他智能体的行为和策略

## 核心组件
- `IndependentRL.py`: 独立学习算法的主要实现
- `DDPG_agent.py`: 单智能体 DDPG 算法
- `NN_actor.py`: Actor 网络结构
- `NN_critic.py`: Critic 网络结构

## 优缺点
优点：
- 实现简单，训练稳定
- 易于并行化

缺点：
- 忽略智能体间的交互
- 难以学习协作行为


| 2025.2.18 updated.
<br>
1. reward独立.
2. 智能体独自决策，没有信息共享。  action = actor(obs);   Q = critic(obs, action). （应该没错）