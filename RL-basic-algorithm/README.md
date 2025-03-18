# RL-basic-algorithm 项目说明

## 项目概述

这是一个强化学习基础算法的实现和复现项目，主要包含了Q-Learning、DQN和PPO等经典强化学习算法，以及一些多智能体协作任务的复现实验。

## 项目结构

```
RL-basic-algorithm/
├── Deep-Q-Network.py        # DQN算法实现
├── PPO/                     # PPO算法相关实现
│   ├── PPO.py               # PPO核心算法
│   ├── common/              # 通用组件
│   ├── essential/           # 核心组件
│   ├── run_PPO.py           # PPO运行脚本
│   └── source link.md       # 参考资源链接
├── Q-Learning.py            # Q-Learning算法实现
├── README.md                # 项目说明文档
├── Reproduce/               # 复现实验
│   └── team-coordination-main/ # 团队协作任务复现
├── Scaling-Multi-agents-on-graphs/ # 多智能体在图上的扩展实验
│   ├── EnvironmentGraph.py  # 环境图实现
│   ├── grid_style_graph_generator.py # 网格风格图生成器
│   ├── gs_rg.py             # 图生成相关工具
│   ├── main.py              # 主程序
│   └── networkx_plot.py     # 网络可视化工具
├── graph_generator.py       # 图生成器
├── reproduce_TC with RL.py  # 使用RL复现团队协作
├── reproduce_Team_coordination_on_graph_with_RL.py # 在图上使用RL复现团队协作
├── reproduce_on_graph_from_TYQW.py # 从TYQW复现图上的实验
└── reproduce_on_line_graph_from_TYQW.py # 从TYQW复现线图上的实验
```

## 主要算法

### Q-Learning

Q-Learning是一种无模型的强化学习算法，通过学习动作-价值函数（Q函数）来优化策略。项目中的[Q-Learning.py]实现了基础的Q-Learning算法。

### DQN (Deep Q-Network)

DQN是Q-Learning的深度学习扩展版本，使用神经网络来近似Q函数。[Deep-Q-Network.py]提供了DQN的实现。

### PPO (Proximal Policy Optimization)

PPO是一种策略梯度方法，通过限制策略更新的步长来提高训练稳定性。[PPO/]目录包含了PPO算法的完整实现。

## 环境要求

主要依赖包括：
- numpy==1.24.2
- networkx==3.0
- matplotlib==3.7.0
- PyTorch (用于神经网络实现)

## 使用方法

1. 安装依赖：
```bash
pip install numpy networkx matplotlib torch
```

2. 运行基础算法示例：
```bash
python Q-Learning.py
python Deep-Q-Network.py
```

3. 运行PPO算法：
```bash
cd PPO
python run_PPO.py
```

4. 运行复现实验：
```bash
python reproduce_TC with RL.py
```

## 参考资源

- 部分代码参考了相关论文和开源项目，详细信息请参见[PPO/source link.md]
- 图生成器参考了小世界网络模型