# 强化学习与多智能体强化学习项目集
[English](./README_en.md) | 中文文档

![项目总状态](https://img.shields.io/badge/状态-维护模式-blue) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![强化学习](https://img.shields.io/badge/强化学习-基础到高级-orange) ![多智能体](https://img.shields.io/badge/多智能体-MADDPG实现-success)

本仓库包含强化学习（RL）和多智能体强化学习（MARL）相关的多个项目，既有经典算法的复现，也有个人的研究实现。通过这些项目，我希望构建从基础强化学习到多智能体强化学习的完整学习路径。

| 项目 | 状态 | 完成度 | 技术栈 |
|------|------|--------|--------|
| [RL_Learning-main](./RL_Learning-main/) | ![状态](https://img.shields.io/badge/状态-已完成-success) | ![完成度](https://img.shields.io/badge/完成度-90%25-green) | ![技术](https://img.shields.io/badge/技术-基础RL算法-blue) |
| [My_MADDPG_Continous](./My_MADDPG_Continous/) | ![状态](https://img.shields.io/badge/状态-已完成-success) | ![完成度](https://img.shields.io/badge/完成度-100%25-brightgreen) | ![技术](https://img.shields.io/badge/技术-连续MADDPG-blue) |
| [My_MPE](./My_MPE/) | ![状态](https://img.shields.io/badge/状态-已完成-success) | ![完成度](https://img.shields.io/badge/完成度-100%25-brightgreen) | ![技术](https://img.shields.io/badge/技术-离散MADDPG-blue) |
| [RL-basic-algorithm](./RL-basic-algorithm/) | ![状态](https://img.shields.io/badge/状态-暂停开发-orange) | ![完成度](https://img.shields.io/badge/完成度-40%25-yellow) | ![技术](https://img.shields.io/badge/技术-Q学习/DQN/PPO-blue) |
| [动手学强化学习](./动手学强化学习/) | ![状态](https://img.shields.io/badge/状态-参考实现-informational) | ![完成度](https://img.shields.io/badge/完成度-100%25-brightgreen) | ![技术](https://img.shields.io/badge/技术-DQN到DDPG-blue) |
| [pytorch-DRL-master](./pytorch-DRL-master/) | ![状态](https://img.shields.io/badge/状态-参考实现-informational) | ![完成度](https://img.shields.io/badge/完成度-100%25-brightgreen) | ![技术](https://img.shields.io/badge/技术-PyTorch/DRL-blue) |

## 学习路径与项目关联
本仓库中的项目构成了一条从基础强化学习到多智能体强化学习的完整学习路径：

1. **基础理论与算法** (RL_Learning-main)：掌握强化学习的数学基础和基本算法
2. **基础算法实现** (RL-basic-algorithm, 动手学强化学习)：动手实现基础强化学习算法
3. **深度强化学习** (pytorch-DRL-master)：学习基于深度学习的强化学习算法
4. **多智能体扩展** (My_MADDPG_Continous, My_MPE)：将单智能体算法扩展到多智能体场景

## 项目结构
### 一、RL_Learning-main：强化学习基础代码复现

复现西湖大学**赵世钰老师**的强化学习课程代码，包括值迭代、策略迭代、蒙特卡洛、时序差分、DQN、Reinforce等算法实现。这部分是理解强化学习基础算法的最佳起点。

#### 参考资源
- [赵老师强化学习课程](https://www.bilibili.com/video/BV1sd4y167NS)
- [强化学习的数学原理](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
#### 代码位置  [`赵老师强化学习代码仓库: ./RL_Learning-main`](./RL_Learning-main/scripts)

#### 更新日志

**2024.6.7**  
重大更新! 原作者render坐标与state设置不一致。坐标已统一修改为：  
![img.png](img.png)
> 原始代码来源: https://github.com/jwk1rose/RL_Learning  
> 本人正在重构代码，尽量分解为更多独立模块并添加详细注释。
>Refactoring the code of jwk1rose,I'm trying to divide it into as many sections as possible and write comments.

---
### 二、多智能体强化学习实现

在掌握了基础强化学习算法后，我们自然会思考：如何将这些方法扩展到多个智能体同时学习的场景？多智能体强化学习（MARL）正是解决这一问题的关键技术。以下是我在MARL领域的两个主要实现。

### 二.1、My_MADDPG_Continous：多智能体深度确定性策略梯度算法

个人基于最新版Pettingzoo中的MPE环境，实现的连续状态，连续动作下的MADDPG算法，支持连续动作空间的多智能体协作与竞争。

#### 实现进度
| 算法            | 状态   | 位置                  | 核心组件                           |
|----------------|--------|----------------------|----------------------------------|
| MADDPG         | ✅ 1.0 | `agents/*.py`        | MADDPG_agent, DDPG_agent, buffer |
| Independent RL | ⏳ 待完成 | `agents/independent/`| IndependentRL (计划中)          |
| Centralized RL | ⏳ 待完成 | `agents/centralized/`| CentralizedRL (计划中)          |
#### 代码位置  [`./My_MADDPG_Continous`](./My_MADDPG_Continous)

### 二.2、My_MPE

基于Pettingzoo的MPE环境，实现的离散动作空间下的MADDPG算法，支持多智能体协作与竞争。这部分探索了在离散动作空间中多智能体协作的可能性。
#### 代码位置  [`./My_MPE`](./My_MPE)

---
### 三、RL-basic-algorithm：基础强化学习算法实现
回到强化学习的基础，这个部分包含了我在学习过程中实现的各种基础算法。这些实现不仅帮助我深入理解了算法原理，也为后续的多智能体研究奠定了坚实基础。

包含Q-Learning、DQN、PPO等基础算法实现，以及一些复现的论文代码。这部分是我学习强化学习基础算法的实践记录，也是后续多智能体研究的基础。

#### 主要内容
- 基础算法: Q-Learning, DQN, PPO
- 论文复现: Team-Coordination on Graphs with Risky Edges (TCGRE)
- 图上的多智能体协调算法实现
#### 代码位置 [`./RL-basic-algorithm`](./RL-basic-algorithm/)

---
### 四、动手学强化学习
《动手学强化学习》书籍代码的复现与扩展，最终目标是扩展到MADDPG。这部分是我系统学习强化学习的记录，从基础算法到高级算法的实现。
#### 实现算法
- DQN (Deep Q-Network)
- Policy Gradient (REINFORCE)
- Actor-Critic
- DDPG (Deep Deterministic Policy Gradient)
#### 学习路径
这部分展示了从基础DQN到DDPG，再到MADDPG的学习路径，是理解多智能体强化学习的基础铺垫。
#### 代码位置 [`./动手学强化学习`](./动手学强化学习/)

#### 参考资源
- [动手学强化学习](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95)
- [HandsOnRL GitHub](https://github.com/peterwu4084/HandsOnRL/tree/main)


### 五、pytorch-DRL-master：PyTorch实现的深度强化学习
随着深度学习的发展，深度强化学习成为了当前研究的热点。这部分包含了基于PyTorch实现的各种深度强化学习算法，是我探索深度强化学习的重要工具。

基于PyTorch的深度强化学习算法实现集合，包含多种经典算法的高效实现。这部分提供了使用PyTorch框架实现强化学习算法的参考。
#### 实现算法
- DQN及其变种 (Double DQN, Dueling DQN)
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed DDPG)
- SAC (Soft Actor-Critic)
#### 代码位置 [`./pytorch-DRL-master`](./pytorch-DRL-master)


## 进行中的项目
- **MARL**: 基于深度强化学习的多智能体协作与协调
  - 探索不同通信机制对多智能体协作的影响
  - 研究异构智能体在复杂环境中的协作策略

- **图上的多智能体协调与决策**
  - 将多智能体强化学习与图神经网络结合
  - 研究大规模图结构上的多智能体协调问题
- **多智能体强化学习的应用**
  - 探索多智能体强化学习在工业、医疗等领域的应用
  - 研究多智能体强化学习在不同场景下的性能优化

## 联系方式
如有任何问题，请随时联系我。
ronchy_lu AT 163 dot com

Fight for MARL.
