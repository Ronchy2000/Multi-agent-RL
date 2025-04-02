# Reinforcement Learning and Multi-Agent Reinforcement Learning Projects

🇺🇸 English | [🇨🇳 中文文档](./README.md)

![Project Status](https://img.shields.io/badge/status-maintenance-blue) ![Python](https://img.shields.io/badge/Python-3.11.8%2B-blue) ![RL](https://img.shields.io/badge/RL-basic%20to%20advanced-orange) ![MARL](https://img.shields.io/badge/MARL-MADDPG%20implemented-success)

This repository contains multiple projects related to Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL), including both reproductions of classic algorithms and personal research implementations. Through these projects, I aim to build a complete learning path from basic reinforcement learning to multi-agent reinforcement learning.

| Project | Status | Completion | Tech Stack | Documentation |
|------|------|--------|--------|----------|
| [RL_Learning-main](./RL_Learning-main/) | ![Status](https://img.shields.io/badge/status-completed-success) | ![Completion](https://img.shields.io/badge/completion-90%25-green) | ![Tech](https://img.shields.io/badge/tech-basic%20RL%20algorithms-blue) | [Implemented Algorithms](./RL_Learning-main/README.md#implemented-algorithms) |
| [My_MADDPG_Continous](./My_MADDPG_Continous/) | ![Status](https://img.shields.io/badge/status-completed-success) | ![Completion](https://img.shields.io/badge/completion-100%25-brightgreen) | ![Tech](https://img.shields.io/badge/tech-continuous%20MADDPG-blue) | [README](./My_MADDPG_Continous/README.md) |
| [My_MPE](./My_MPE/) | ![Status](https://img.shields.io/badge/status-completed-success) | ![Completion](https://img.shields.io/badge/completion-100%25-brightgreen) | ![Tech](https://img.shields.io/badge/tech-discrete%20MADDPG-blue) | [README](./My_MPE/README.md) |
| [RL-basic-algorithm](./RL-basic-algorithm/) | ![Status](https://img.shields.io/badge/status-paused-orange) | ![Completion](https://img.shields.io/badge/completion-40%25-yellow) | ![Tech](https://img.shields.io/badge/tech-Q--learning%2FDQN%2FPPO-blue) | [README](./RL-basic-algorithm/README.md) |
| [动手学强化学习](./动手学强化学习/) | ![Status](https://img.shields.io/badge/status-reference-informational) | ![Completion](https://img.shields.io/badge/completion-100%25-brightgreen) | ![Tech](https://img.shields.io/badge/tech-DQN%20to%20DDPG-blue) | [README](./动手学强化学习/README.md) |
| [pytorch-DRL-master](./pytorch-DRL-master/) | ![Status](https://img.shields.io/badge/status-reference-informational) | ![Completion](https://img.shields.io/badge/completion-100%25-brightgreen) | ![Tech](https://img.shields.io/badge/tech-PyTorch%2FDRL-blue) | [README](./pytorch-DRL-master/README.md) |

## Learning Path and Project Connections

The projects in this repository form a complete learning path from basic reinforcement learning to multi-agent reinforcement learning:

1. **Basic Theory and Algorithms** (RL_Learning-main): Master the mathematical foundations and basic algorithms of reinforcement learning
2. **Basic Algorithm Implementation** (RL-basic-algorithm, 动手学强化学习): Implement basic reinforcement learning algorithms
3. **Deep Reinforcement Learning** (pytorch-DRL-master): Learn reinforcement learning algorithms based on deep learning
4. **Multi-Agent Extensions** (My_MADDPG_Continous, My_MPE): Extend single-agent algorithms to multi-agent scenarios

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
Major update! The original author's render coordinates were inconsistent with the state settings. The coordinates have been unified as:  
![img.png](img.png)

> Original code source: https://github.com/jwk1rose/RL_Learning  
> I am refactoring the code, trying to divide it into as many independent modules as possible and adding detailed comments.

---

### II. Multi-Agent Reinforcement Learning Implementation

After mastering basic reinforcement learning algorithms, we naturally think: how can these methods be extended to scenarios where multiple agents learn simultaneously? Multi-agent reinforcement learning (MARL) is the key technology to solve this problem. Below are my two main implementations in the MARL field.

### II.1 My_MADDPG_Continous: Multi-Agent Deep Deterministic Policy Gradient Algorithm

Personal implementation of the MADDPG algorithm for continuous states and actions based on the latest version of the MPE environment in PettingZoo, supporting multi-agent cooperation and competition in continuous action spaces.

<div align="center">
  <img src="./My_MADDPG_Continous/plot/simple_tag_v3_demo_loop.gif" alt="MADDPG Demo" width="45%"/>
  <p><strong>MADDPG training results: Predators (red) chasing prey (green)</strong></p>
</div>

#### Implementation Progress
| Algorithm      | Status | Location              | Core Components                    |
|----------------|--------|----------------------|----------------------------------|
| MADDPG         | ✅ 1.0 | `agents/*.py`        | MADDPG_agent, DDPG_agent, buffer |
| Independent RL | ⏳ Planned | `agents/independent/`| IndependentRL (planned)          |
| Centralized RL | ⏳ Planned | `agents/centralized/`| CentralizedRL (planned)          |

#### Code Location
[./My_MADDPG_Continous](./My_MADDPG_Continous)

### II.2 My_MPE

Implementation of the MADDPG algorithm for discrete action spaces based on the MPE environment in PettingZoo, supporting multi-agent cooperation and competition. This part explores the possibility of multi-agent cooperation in discrete action spaces.

#### Code Location
[./My_MPE](./My_MPE)

---

### III. RL-basic-algorithm: Basic Reinforcement Learning Algorithm Implementation

Going back to the basics of reinforcement learning, this part contains various basic algorithms I implemented during my learning process. These implementations not only helped me deeply understand the algorithm principles but also laid a solid foundation for subsequent multi-agent research.

This includes implementations of Q-Learning, DQN, PPO, and other basic algorithms, as well as some reproduced paper code. This part is a record of my practice in learning basic reinforcement learning algorithms and is the foundation for subsequent multi-agent research.

#### Main Content
- Basic Algorithms: Q-Learning, DQN, PPO
- Paper Reproduction: Team-Coordination on Graphs with Risky Edges (TCGRE)
- Multi-agent coordination algorithm implementation on graphs

#### Code Location
[./RL-basic-algorithm](./RL-basic-algorithm/)

---

### IV. 动手学强化学习 (Hands-on Reinforcement Learning)

Reproduction and extension of the code from the book "Hands-on Reinforcement Learning", with the ultimate goal of extending to MADDPG. This part is a record of my systematic learning of reinforcement learning, from basic algorithms to advanced algorithm implementation.

#### Implemented Algorithms
- DQN (Deep Q-Network)
- Policy Gradient (REINFORCE)
- Actor-Critic
- DDPG (Deep Deterministic Policy Gradient)

#### Learning Path
This part demonstrates the learning path from basic DQN to DDPG, and then to MADDPG, which is the foundation for understanding multi-agent reinforcement learning.

#### Code Location
[./动手学强化学习](./动手学强化学习/)

#### References
- [Hands-on Reinforcement Learning](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95)
- [HandsOnRL GitHub](https://github.com/peterwu4084/HandsOnRL/tree/main)

---

### V. pytorch-DRL-master: Deep Reinforcement Learning Implemented with PyTorch

With the development of deep learning, deep reinforcement learning has become a hot research topic. This part contains various deep reinforcement learning algorithms implemented based on PyTorch, which is an important tool for my exploration of deep reinforcement learning.

A collection of deep reinforcement learning algorithm implementations based on PyTorch, containing efficient implementations of various classic algorithms. This part provides a reference for implementing reinforcement learning algorithms using the PyTorch framework.

#### Implemented Algorithms
- DQN and its variants (Double DQN, Dueling DQN)
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed DDPG)
- SAC (Soft Actor-Critic)

#### Code Location
[./pytorch-DRL-master](./pytorch-DRL-master)

---

## Ongoing Projects

- **MARL**: Multi-agent cooperation and coordination based on deep reinforcement learning
  - Exploring the impact of different communication mechanisms on multi-agent cooperation
  - Studying cooperation strategies of heterogeneous agents in complex environments

- **Multi-agent Coordination and Decision-making on Graphs**
  - Combining multi-agent reinforcement learning with graph neural networks
  - Studying multi-agent coordination problems on large-scale graph structures

- **Applications of Multi-agent Reinforcement Learning**
  - Exploring applications of multi-agent reinforcement learning in industrial, medical, and other fields
  - Studying performance optimization of multi-agent reinforcement learning in different scenarios

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