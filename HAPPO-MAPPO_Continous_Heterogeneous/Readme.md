# HAPPO for Heterogeneous Multi-Agent Reinforcement Learning

本项目实现了HAPPO（Heterogeneous-Agent Proximal Policy Optimization）算法，专门针对异构多智能体环境设计，如PettingZoo的Simple_tag_v3环境。

## 🎯 核心特性

- ✅ **异构智能体支持**: 不同智能体可以有不同的观察空间维度
- ✅ **HAPPO算法**: 实现了完整的HAPPO训练逻辑，包括保护因子机制
- ✅ **结构化模型保存**: 单文件保存所有智能体的独立Actor模型
- ✅ **连续动作空间**: 支持连续控制任务
- ✅ **可视化评估**: 支持训练后的策略可视化评估

## 📋 目录结构

```
HAPPO-MAPPO_Continous_Heterogeneous/
├── HAPPO.py              # HAPPO核心算法实现
├── HAPPO_main.py         # 训练主脚本
├── HAPPO_evaluate.py     # 评估脚本
├── replay_buffer.py      # 经验回放缓冲区
├── normalization.py      # 奖励归一化工具
├── models/               # 训练好的模型保存目录
├── data/                 # 训练数据保存目录
└── Readme.md            # 本文件
```

## 🏗️ HAPPO异构多智能体架构

### 1. 基本架构设计

HAPPO为每个智能体创建独立的Actor和Critic网络：

```
异构智能体架构:
├── adversary_0: Actor_0 + Critic_0 (追捕者)
├── adversary_1: Actor_1 + Critic_1 (追捕者)  
├── adversary_2: Actor_2 + Critic_2 (追捕者)
└── agent_0:     Actor_3 + Critic_3 (逃跑者)
```

**关键特点：**
- 🎭 **Actor异构性**: 每个智能体的Actor适配不同的观察空间维度
- 🧠 **Critic独立性**: 每个智能体有独立的价值网络，便于个性化价值评估
- 🔄 **HAPPO更新**: 随机顺序更新智能体，使用保护因子防止策略崩塌

### 2. 网络输入输出规格

以Simple_tag_v3环境为例：

```python
# Actor网络（异构输入，同构输出）
adversary_0 Actor: 12维观察 → 5维连续动作
adversary_1 Actor: 12维观察 → 5维连续动作  
adversary_2 Actor: 12维观察 → 5维连续动作
agent_0 Actor:     10维观察 → 5维连续动作

# Critic网络（同构输入，标量输出）
所有Critic: 46维全局状态(12+12+12+10) → 1维价值估计
```

### 3. 训练数据流程

**数据收集阶段:**
```python
# 1. 环境交互，获取异构观察
obs_dict = {
    'adversary_0': [12维观察],  # 追捕者观察
    'adversary_1': [12维观察], 
    'adversary_2': [12维观察],
    'agent_0':     [10维观察]   # 逃跑者观察（维度更小）
}

# 2. 数据预处理
# - Actor使用: 各智能体保持原始观察维度
# - ReplayBuffer: 填充到最大维度12进行统一存储
# - Critic使用: 拼接为46维全局状态
obs_padded = [12, 12, 12, 12]  # agent_0填充2个零
global_state = [46维]          # 真实维度拼接
```

**HAPPO训练更新:**
```python
for agent_id in random_shuffled_agents:  # 🔀 随机智能体顺序
    # 🎯 提取该智能体的真实观察
    obs_real = extract_real_obs(obs_padded, agent_id)
    
    # 🎭 Actor更新（使用个性化观察）
    actor_loss = PPO_loss(obs_real, actions, advantages * protection_factor)
    
    # 🧠 Critic更新（使用全局状态）
    critic_loss = MSE_loss(global_state, value_targets)
    
    # 🛡️ 更新保护因子
    protection_factor *= exp(new_logprob - old_logprob)
```

## 💾 结构化模型保存

### 保存格式

HAPPO使用**结构化单文件保存**，将所有智能体的Actor模型打包在一个`.pth`文件中：

```python
# 保存的模型文件结构
{
    'format': 'hetero_per_agent_actor_v2',
    'agents': ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0'],
    'actor_state_dict_by_agent': {
        'adversary_0': {完整的Actor_0参数字典},
        'adversary_1': {完整的Actor_1参数字典}, 
        'adversary_2': {完整的Actor_2参数字典},
        'agent_0':     {完整的Actor_3参数字典}
    }
}
```

### 保存优势

- 🎯 **原子性**: 保证所有智能体模型的版本一致性
- 📦 **便于管理**: 单文件包含完整的策略组合
- 🔒 **避免丢失**: 不会出现某个智能体模型文件缺失的问题
- 🚀 **部署友好**: 一次加载获得完整的多智能体系统

### 文件命名规则

```
模型文件名格式:
HAPPO_actor_env_{环境名}_number_{编号}_seed_{随机种子}_step_{训练步数}k.pth

示例:
HAPPO_actor_env_simple_tag_v3_number_1_seed_23_step_10k.pth
```

## 🚀 快速开始

### 环境要求

```bash
# 安装依赖
pip install torch numpy pettingzoo[mpe] tensorboard
```

### 训练模型

```bash
# 使用默认参数训练Simple_tag_v3环境
python HAPPO_main.py

# 自定义参数训练
python HAPPO_main.py \
    --env_name simple_tag_v3 \
    --max_train_steps 200000 \
    --lr 5e-4 \
    --batch_size 32 \
    --evaluate_freq 5000 \
    --seed 23
```

### 评估训练结果

```bash
# 评估特定步数的模型
python HAPPO_evaluate.py \
    --env_name simple_tag_v3 \
    --timestamp 2025-09-24_20-20 \
    --model_step 10 \
    --evaluate_episodes 5 \
    --render_mode human

# 无渲染评估（更快）
python HAPPO_evaluate.py \
    --env_name simple_tag_v3 \
    --timestamp 2025-09-24_20-20 \
    --model_step 10 \
    --render_mode None
```

## 📊 训练参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `max_train_steps` | 200000 | 总训练步数 |
| `episode_limit` | 100 | 每回合最大步数 |
| `batch_size` | 32 | 批次大小（回合数） |
| `mini_batch_size` | 8 | 小批次大小 |
| `lr` | 5e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `epsilon` | 0.2 | PPO裁剪参数 |
| `K_epochs` | 15 | PPO更新轮数 |

### 评估参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `timestamp` | - | 模型时间戳目录 |
| `model_step` | - | 加载的模型步数（k为单位） |
| `evaluate_episodes` | 5 | 评估回合数 |
| `render_mode` | "human" | 渲染模式 |

## 🎮 支持的环境

### Simple_tag_v3 (默认)
- **场景**: 追捕逃跑游戏
- **智能体**: 3个追捕者 + 1个逃跑者
- **异构性**: 追捕者12维观察，逃跑者10维观察
- **目标**: 追捕者合作捕获逃跑者

### Simple_spread_v3
- **场景**: 协作导航
- **智能体**: N个智能体（同构）
- **目标**: 覆盖所有landmark

### Simple_adversary_v3
- **场景**: 对抗通信
- **智能体**: 对抗智能体 + 协作智能体
- **目标**: 信息传递与干扰

## 📈 训练监控

### 终端输出
```bash
total_steps:5000    evaluate_reward:-78.54
total_steps:10000   evaluate_reward:25.52
```

### 数据文件
- **训练曲线**: `data/happo_rewards_{env_name}_n{number}_s{seed}_{timestamp}.csv`
- **模型检查点**: `models/{timestamp}/HAPPO_actor_env_*.pth`

## 🔧 模型加载详解

### 自动格式检测

评估脚本会自动检测模型格式：

```python
# 异构模型（推荐）
if 'actor_state_dict_by_agent' in model_data:
    # 逐个加载每个智能体的Actor
    for agent_id, state_dict in model_data['actor_state_dict_by_agent'].items():
        agents[agent_id].actor.load_state_dict(state_dict)

# 兼容旧格式
else:
    # 共享模型复制到所有智能体
    for agent_id in all_agents:
        agents[agent_id].actor.load_state_dict(model_data)
```

### 加载验证

```bash
# 成功加载示例输出
✅ 成功加载智能体 adversary_0 的Actor模型
✅ 成功加载智能体 adversary_1 的Actor模型  
✅ 成功加载智能体 adversary_2 的Actor模型
✅ 成功加载智能体 agent_0 的Actor模型
总共加载了 4 个智能体的模型
```

## 🧪 实验建议

### 超参数调优
- **学习率**: 建议范围 [1e-5, 1e-3]，异构环境可能需要较小学习率
- **批次大小**: 根据内存调整，建议 [16, 64]
- **PPO轮数**: 困难任务建议减少到5-10轮，防止过拟合

### 训练技巧
- **奖励塑形**: 对于稀疏奖励环境，考虑添加中间奖励
- **课程学习**: 从简单场景开始，逐渐增加难度
- **种子实验**: 使用多个随机种子验证结果稳定性

## 📝 算法原理

### HAPPO vs MAPPO

| 特性 | MAPPO | HAPPO |
|-----|-------|-------|
| 智能体更新 | 同时更新 | 随机顺序更新 |
| 策略保护 | 无 | 保护因子机制 |
| 异构支持 | 有限 | 原生支持 |
| 训练稳定性 | 中等 | 更稳定 |

### 保护因子机制

HAPPO的核心创新是保护因子（Protection Factor），防止先更新的智能体策略被后更新的智能体破坏：

```python
# 更新前记录旧策略
old_logprob = policy.log_prob(actions)

# 更新智能体策略
update_agent_policy()

# 计算新策略
new_logprob = policy.log_prob(actions)

# 更新保护因子
protection_factor *= exp(new_logprob - old_logprob)
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发建议
- 遵循现有代码风格
- 添加必要的注释和文档
- 测试新功能的兼容性

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🙏 致谢

- [HAPPO论文](https://arxiv.org/abs/2109.11251)
- [PettingZoo环境](https://pettingzoo.farama.org/)
- [PyTorch深度学习框架](https://pytorch.org/)

---

如有问题或建议，请提交Issue或联系维护者。
