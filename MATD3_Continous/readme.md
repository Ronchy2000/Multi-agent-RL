# MATD3 多智能体强化学习项目

本项目实现了基于 TD3 (Twin Delayed Deep Deterministic Policy Gradient) 算法的多智能体强化学习系统，专注于连续动作空间的多智能体协作与对抗任务。

## 项目结构

```
MATD3_Continous/
├── agents/                  # 智能体算法实现
│   └── legacy_mpe_official/ # 基于官方 MPE 环境的 MATD3 实现
│       ├── MATD3/           # MATD3 算法实现
│       └── ...
├── envs/                    # 环境实现
│   ├── custom_agents_dynamics.py  # 自定义智能体动力学
│   └── simple_tag_env.py    # 自定义追逐逃避环境
├── main/                    # 主程序脚本
│   └── td3_main/           # TD3 相关主程序
├── plot/                    # 数据可视化
│   ├── matd3_data/          # 训练数据存储
│   └── plot_rewards.py      # 奖励绘图脚本
├── logs/                    # 日志文件
└── utils/                   # 工具函数
    ├── conda-environment.yml    # Conda 环境配置
    ├── pip-requirements.txt     # Pip 依赖
    ├── logger.py                # 日志工具
    └── setupPettingzoo.py       # PettingZoo 环境设置
```

## 环境说明

本项目基于 PettingZoo 的 MPE (Multi-Particle Environment) 环境，主要实现了 simple_tag 追逐逃避任务：

- **追捕者 (Adversaries)**: 多个追捕者协作追捕逃避者
- **逃避者 (Good Agents)**: 尝试逃离追捕者

环境特点：
- 连续动作空间
- 部分可观测状态
- 多智能体协作与对抗

## 算法实现

项目实现了 MATD3 (Multi-Agent Twin Delayed Deep Deterministic Policy Gradient) 算法，这是 TD3 算法的多智能体扩展版本，主要特点：

- 双重 Q 网络减少过估计
- 延迟策略更新
- 目标策略平滑正则化
- 集中式训练，分布式执行 (CTDE) 范式

## 奖励设计

环境中的奖励函数设计如下：

### 逃避者奖励
```python
def agent_reward(self, agent, world):
    # 逃避者被捕获时获得负奖励
    rew = 0
    shape = True
    # 可选的奖励塑形
    # ...
    # 边界惩罚，防止逃避者离开环境
    def bound(x):
        if x < 0.9:
            return 0
        if x < 1.0:
            return (x - 0.9) * 10
        return min(np.exp(2 * x - 2), 10)
    
    for p in range(world.dim_p):
        x = abs(agent.state.p_pos[p])
        rew -= bound(x)
    
    return rew
```

### 追捕者奖励
```python
def adversary_reward(self, agent, world):
    # 追捕者捕获逃避者时获得正奖励
    rew = 0
    shape = True
    # 可选的奖励塑形
    # ...
    # 碰撞奖励
    if agent.collide:
        for ag in agents:
            for adv in adversaries: 
                if self.is_collision(ag, adv):
                    rew += 10
    # 边界惩罚
    # ...
    return rew
```

### 全局奖励
```python
def global_reward(self, world):
    # 鼓励追捕者合作追捕逃避者的全局奖励
    global_rew = 0.0
    # 计算碰撞奖励和距离惩罚
    # ...
    # 额外奖励：全部追捕者都接近逃避者时
    if all(np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos))) < 0.2 for adv in adversaries):
        global_rew += 10.0  # 围捕成功的额外奖励
    return global_rew
```

## 训练与评估

训练参数在 `main_parameters.py` 中设置：

```python
# 主要参数
parser.add_argument("--env_name", type=str, default="simple_tag_v3")
parser.add_argument("--episode_num", type=int, default=5000)
parser.add_argument("--episode_length", type=int, default=500)
parser.add_argument('--learn_interval', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--actor_lr', type=float, default=0.0002)
parser.add_argument('--critic_lr', type=float, default=0.002)
```

## 数据可视化

训练过程中的奖励数据保存在 CSV 文件中，可以使用 `plot/plot_rewards.py` 脚本进行可视化：

```python
# 使用方法
python plot/plot_rewards.py
```

该脚本会生成包含所有智能体奖励曲线的图表，并保存为 PNG 文件。

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- PettingZoo 1.22.0+
- Pandas 2.2.1 (注意：2.2.3 版本可能会出错)
- Matplotlib

可以使用提供的环境配置文件进行安装：
```bash
# 使用 Conda
conda env create -f utils/conda-environment.yml

# 或使用 Pip
pip install -r utils/pip-requirements.txt
```

## 注意事项

- 所有智能体（追捕者和逃避者）都通过 Actor 网络进行控制（全学习模式）
- 训练数据保存在 `plot/matd3_data/` 目录下
- 可视化脚本需要注意 CSV 文件中的列名格式
