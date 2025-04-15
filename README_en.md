# Reinforcement Learning and Multi-Agent Reinforcement Learning Projects

🇺🇸 English | [🇨🇳 中文文档](./README.md)

![Project Status](https://img.shields.io/badge/status-maintenance-blue) ![Python](https://img.shields.io/badge/Python-3.11.8%2B-blue) ![RL](https://img.shields.io/badge/RL-basic%20to%20advanced-orange) ![MARL](https://img.shields.io/badge/MARL-MADDPG%20implemented-success)

This repository contains multiple projects related to Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL), including both reproductions of classic algorithms and personal research implementations. Through these projects, I aim to build a complete learning path from basic reinforcement learning to multi-agent reinforcement learning.

| Project | Status | Completion | Tech Stack | Documentation |
|------|------|--------|--------|----------|
| [RL_Learning-main](./RL_Learning-main/) | ![Status](https://img.shields.io/badge/status-completed-success) | ![Completion](https://img.shields.io/badge/completion-90%25-green) | ![Tech](https://img.shields.io/badge/tech-basic%20RL%20algorithms-blue) | [Implemented Algorithms](./RL_Learning-main/README.md#implemented-algorithms) |
| [Hands-on RL](./动手学强化学习/) | ![Status](https://img.shields.io/badge/status-reference-informational) | ![Completion](https://img.shields.io/badge/completion-100%25-brightgreen) | ![Tech](https://img.shields.io/badge/tech-DQN%20to%20DDPG-blue) | [README](./动手学强化学习/README.md) |
| [MADDPG_Continous](./MADDPG_Continous/) | ![Status](https://img.shields.io/badge/status-completed-success) | ![Completion](https://img.shields.io/badge/completion-100%25-brightgreen) | ![Tech](https://img.shields.io/badge/tech-continuous%20MADDPG-blue) | [README](./MADDPG_Continous/README.md) |

## Learning Path and Project Connections

The projects in this repository form a complete learning path from basic reinforcement learning to multi-agent reinforcement learning:

1. **Basic Theory and Algorithms** (RL_Learning-main): Master the mathematical foundations and basic algorithms of reinforcement learning
2. **Basic Algorithm Implementation** (Hands-on RL): Implement basic reinforcement learning algorithms
3. **Multi-Agent Extensions** (MADDPG_Continous): Extend single-agent algorithms to multi-agent scenarios

## Project Structure

### I. RL_Learning-main: Reproduction of Basic Reinforcement Learning Code

Reproduction of Professor Shiyu Zhao's reinforcement learning course code from Westlake University, including value iteration, policy iteration, Monte Carlo, temporal difference, DQN, Reinforce, and other algorithm implementations. This part is the best starting point for understanding basic reinforcement learning algorithms.

<div align="center">
  <img src="./RL_Learning-main/scripts/Chapter4_Value iteration and Policy iteration/plot_figure/policy_iteration.png" width="45%" alt="Policy Iteration Visualization"/>
  <img src="./RL_Learning-main/scripts/Chapter4_Value iteration and Policy iteration/plot_figure/value_iteration.png" width="45%" alt="Value Iteration Visualization"/>
  <p><strong>From left to right: Policy Iteration, Value Iteration Visualization</strong></p>
</div>

#### References
- [Professor Zhao's Reinforcement Learning Course](https://www.bilibili.com/video/BV1sd4y167NS)
- [Mathematical Foundation of Reinforcement Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)

#### Code Location
[Professor Zhao's RL Code Repository: ./RL_Learning-main](./RL_Learning-main/scripts)

#### Update Log
**2024.6.7**  
Major update! The original author's render coordinates were inconsistent with state settings. Coordinates have been unified:  
![img.png](img.png)
> Original code source: https://github.com/jwk1rose/RL_Learning  
> I am currently refactoring the code, trying to divide it into more independent modules and add detailed comments.

---

### II. Hands-on Reinforcement Learning

Reproduction and expansion of the code from the book "Hands-on Reinforcement Learning", with the ultimate goal of extending to MADDPG. This part records my systematic learning of reinforcement learning, from basic to advanced algorithm implementation.

#### Implemented Algorithms
- DQN (Deep Q-Network)
- Policy Gradient (REINFORCE)
- Actor-Critic
- DDPG (Deep Deterministic Policy Gradient)

#### Learning Path
This section demonstrates the learning path from basic DQN to DDPG, and then to MADDPG, laying the foundation for understanding multi-agent reinforcement learning.

#### Code Location
[./动手学强化学习](./动手学强化学习/)

#### References
- [Hands-on Reinforcement Learning](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95)
- [HandsOnRL GitHub](https://github.com/peterwu4084/HandsOnRL/tree/main)

---

### III. Multi-Agent Reinforcement Learning Implementation
> **This project is specially optimized for Predator-Prey pursuit games!** Built on a modified `PettingZoo MPE` environment, it provides a comprehensive multi-agent cooperative and competitive environment suitable for pursuit control, swarm intelligence, and strategy game research.

After mastering basic reinforcement learning algorithms, we naturally think: how can these methods be extended to scenarios where multiple agents learn simultaneously? Multi-agent reinforcement learning (MARL) is the key technology to solve this problem.

#### 3.1 MADDPG_Continous: Multi-Agent Deep Deterministic Policy Gradient Algorithm

Personal implementation of the MADDPG algorithm based on the latest version of the MPE environment in PettingZoo, supporting multi-agent cooperation and competition in continuous action spaces.

<div align="center">
  <img src="./MADDPG_Continous/plot/simple_tag_v3_demo_loop.gif" alt="MADDPG Demo" width="45%"/>
  <p><strong>MADDPG training results: Predators (red) chasing prey (green)</strong></p>
</div>

#### Implementation Progress
| Algorithm      | Status | Location              | Core Components                    |
|----------------|--------|----------------------|----------------------------------|
| MADDPG         | ✅ 1.0 | `agents/maddpg/`     | MADDPG_agent, DDPG_agent, buffer |
| Independent RL | ⏳ Planned | `agents/independent/`| IndependentRL (planned)        |
| Centralized RL | ⏳ Planned | `agents/centralized/`| CentralizedRL (planned)        |

#### Code Location
[./MADDPG_Continous](./MADDPG_Continous)

## Ongoing Projects
- **MARL**: Multi-agent cooperation and coordination based on deep reinforcement learning
  - Exploring the impact of different communication mechanisms on multi-agent cooperation
  - Researching cooperative strategies of heterogeneous agents in complex environments

- **Multi-agent Coordination and Decision-Making on Graphs**
  - Combining multi-agent reinforcement learning with graph neural networks
  - Researching multi-agent coordination problems on large-scale graph structures

- **Applications of Multi-Agent Reinforcement Learning**
  - Exploring applications of multi-agent reinforcement learning in industries, healthcare, etc.
  - Researching performance optimization of multi-agent reinforcement learning in different scenarios

## Contact

If you have any questions, please feel free to contact me.
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