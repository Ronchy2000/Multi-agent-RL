# 强化学习与多智能体强化学习项目集
[ 🇺🇸 English](./README_en.md) | 🇨🇳 中文文档

![项目总状态](https://img.shields.io/badge/状态-维护模式-blue) ![Python](https://img.shields.io/badge/Python-3.11.8%2B-blue) ![强化学习](https://img.shields.io/badge/强化学习-基础到高级-orange) ![多智能体](https://img.shields.io/badge/多智能体-MADDPG实现-success)

本仓库包含强化学习（RL）和多智能体强化学习（MARL）相关的多个项目，既有经典算法的复现，也有个人的研究实现。通过这些项目，我希望构建从基础强化学习到多智能体强化学习的完整学习路径。

| 项目 | 状态 | 完成度 | 技术栈 | 文档索引 |
|------|------|--------|--------|----------|
| [RL_Learning-main](./RL_Learning-main/) | ![状态](https://img.shields.io/badge/状态-已完成-success) | ![完成度](https://img.shields.io/badge/完成度-90%25-green) | ![技术](https://img.shields.io/badge/技术-基础RL算法-blue) | [已实现算法](./RL_Learning-main/README.md#已实现算法) |
| [动手学强化学习](./动手学强化学习/) | ![状态](https://img.shields.io/badge/状态-参考实现-informational) | ![完成度](https://img.shields.io/badge/完成度-100%25-brightgreen) | ![技术](https://img.shields.io/badge/技术-DQN到DDPG-blue) | [README](./动手学强化学习/README.md) |
| [MADDPG_Continous](./MADDPG_Continous/) | ![状态](https://img.shields.io/badge/状态-已完成-success) | ![完成度](https://img.shields.io/badge/完成度-100%25-brightgreen) | ![技术](https://img.shields.io/badge/技术-连续MADDPG-blue) | [中文文档](./MADDPG_Continous/README.md#项目特色) |
| [MATD3_Continous](./MATD3_Continous/) | ![状态](https://img.shields.io/badge/状态-已完成-success) | ![完成度](https://img.shields.io/badge/完成度-100%25-brightgreen) | ![技术](https://img.shields.io/badge/技术-连续MATD3-blue) | [中文文档](./MATD3_Continous/readme.md) |


## 学习路径与项目关联
本仓库中的项目构成了一条从基础强化学习到多智能体强化学习的完整学习路径：

1. **基础理论与算法** (RL_Learning-main)：掌握强化学习的数学基础和基本算法
2. **基础算法实现** (动手学强化学习)：动手实现基础强化学习算法
4. **多智能体扩展** (MADDPG_Continous, MATD3_Continous)：将单智能体算法扩展到多智能体场景

## 项目结构
### RL_Learning-main：强化学习基础代码复现

复现西湖大学**赵世钰老师**的强化学习课程代码，包括值迭代、策略迭代、蒙特卡洛、时序差分、DQN、Reinforce等算法实现。这部分是理解强化学习基础算法的最佳起点。

<div align="center">
  <img src="./RL_Learning-main/scripts/Chapter4_Value iteration and Policy iteration/plot_figure/policy_iteration.png" width="45%" alt="策略迭代可视化"/>
  <img src="./RL_Learning-main/scripts/Chapter4_Value iteration and Policy iteration/plot_figure/value_iteration.png" width="45%" alt="值迭代可视化"/>
  <p><strong>从左到右: 策略迭代算法、值迭代算法可视化</strong></p>
</div>

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
### 二、动手学强化学习
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

---
### 三、多智能体强化学习实现
> **本项目专为Predator-Prey追逃博弈任务优化！** 在`PettingZoo MPE`环境基础上重构修改，提供了完整的多智能体协作与对抗环境，适用于围捕控制、群体智能和策略博弈研究。

在掌握了基础强化学习算法后，我们自然会思考：如何将这些方法扩展到多个智能体同时学习的场景？多智能体强化学习（MARL）正是解决这一问题的关键技术。以下是我在MARL领域的两个主要实现。

#### 3.1 MADDPG_Continous：多智能体深度确定性策略梯度算法


个人基于最新版**Pettingzoo**`(pettingzoo==1.25.0)`中的MPE环境，实现的连续状态，连续动作下的MADDPG算法，支持连续动作空间的多智能体协作与竞争。

> MADDPG algorithm Reference: https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch

<div align="center">
  <img src="./MADDPG_Continous/plot/simple_tag_v3_demo_loop.gif" alt="智能体行为" width="45%"/>
  <p><strong>训练后的智能体行为展示：捕食者(红色)追逐猎物(绿色)的过程</strong></p>

  <img src="./MADDPG_Continous/plot/demo-rewards_plot_ma.png" alt="训练收敛结果" width="80%"/>
  <p><strong>MADDPG算法在simple_tag_v3环境中的奖励收敛曲线</strong></p>
</div>


#### 实现进度
| 算法            | 状态   | 位置                  | 核心组件                           |
|----------------|--------|----------------------|----------------------------------|
| MADDPG         | ✅ 1.0 | `agents/maddpg/`        | MADDPG_agent, DDPG_agent, buffer |
| Independent RL | ⏳ 待完成 | `agents/independent/`| IndependentRL (计划中)          |
| Centralized RL | ⏳ 待完成 | `agents/centralized/`| CentralizedRL (计划中)          |
#### 代码位置  [`./MADDPG_Continous`](./MADDPG_Continous)


#### 3.2 MATD3_Continous：多智能体双延迟深度确定性策略梯度算法

基于TD3算法的多智能体扩展版本(MATD3: Twin Delayed Deep Deterministic Policy Gradient)，相比MADDPG，通过双Q网络和目标策略平滑机制有效解决过估计问题，提供更稳定的训练和更优的策略。

> MATD3 algorithm Reference: https://github.com/wild-firefox/FreeRL/blob/main/MADDPG_file/MATD3_simple.py

<div align="center">
  <img src="./MATD3_Continous/plot/training_rewards_demo.png" alt="训练收敛结果" width="80%"/>
  <p><strong>MATD3算法在simple_tag_env环境中的奖励收敛曲线</strong></p>
</div>

#### MATD3 vs MADDPG
MATD3对标准MADDPG进行了以下关键增强：

1. **双Q网络设计**: 减少对动作值的过估计
2. **延迟策略更新**: 提高训练稳定性
3. **目标策略平滑**: 通过在目标动作中加入噪声防止过拟合
4. **自适应噪声调整**: 根据训练进度动态调整探索噪声

#### 代码位置  [`./MATD3_Continous`](./MATD3_Continous)



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



## Star History

<a href="https://www.star-history.com/#Ronchy2000/Multi-agent-RL&Date">

 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Ronchy2000/Multi-agent-RL&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Ronchy2000/Multi-agent-RL&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Ronchy2000/Multi-agent-RL&type=Date" />
 </picture>
</a>
