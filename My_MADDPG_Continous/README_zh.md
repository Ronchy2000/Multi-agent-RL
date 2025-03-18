# 多智能体深度强化学习MADDPG算法

![项目状态](https://img.shields.io/badge/状态-不再维护-red) ![MADDPG](https://img.shields.io/badge/MADDPG-已实现-success)![Python](https://img.shields.io/badge/python-3.12%2B-blue)

> 当前状态：MADDPG算法已在 `/agents/*.py` 中实现

## 🚀 实现进度
| 算法            | 状态   | 位置                  | 核心组件                           |
|----------------|--------|----------------------|----------------------------------|
| MADDPG         | ✅ 1.0 | `agents/*.py`        | MADDPG_agent, DDPG_agent, buffer |
| Independent RL | ⏳ 待完成 | `agents/independent/`| IndependentRL (计划中)          |
| Centralized RL | ⏳ 待完成 | `agents/centralized/`| CentralizedRL (计划中)          |

> 注意：MADDPG模块目前位于agents根目录（buffer.py, DDPG_agent.py等），但功能完整可用！

## 🏗️ 项目结构
```tree
My_MADDPG_Continous/
├── agents/                   # 核心实现
│   ├── MADDPG_agent.py       # 多智能体控制器
│   ├── DDPG_agent.py         # 基础DDPG实现
│   ├── buffer.py             # 经验回放缓冲区
│   └── (NN_actor|NN_critic).py  # 神经网络模块
├── envs/                     # 自定义环境
│   ├── custom_agents_dynamics.py  # 扩展物理引擎
│   └── simple_tag_env.py           # 修改版tag环境
├── utils/                    # 工具模块
│   ├── runner.py             # 训练运行器
│   └── logger.py             # 训练日志系统
├── main_train.py             # 统一训练入口
├── main_evaluate.py          # 统一评估入口
└── main_parameters.py        # 统一参数配置
```

## 🛠️ 快速开始

### 环境配置
```bash
# 创建并激活虚拟环境（推荐）
略
# 安装核心依赖
pip install -r requirements.txt

# 安装PettingZoo（仅需一次）
python utils/setupPettingzoo.py
```

### 🖥️ 运行配置
```bash
# 启动Visdom可视化服务器（新终端）
python -m visdom.server
# 或指定端口
python -m visdom.server -port 8097

# 访问训练仪表盘：
# http://localhost:8097
```

## 🔄 训练流程
1. **参数配置**   
在 [`main_parameter.py`](main_parameters.py) 中设置环境参数：
```python
env_name = 'simple_tag_v3'  # 可选：simple_adversary_v3/simple_spread_v3
episode_num = 2000         # 总训练回合数
# 训练参数
batch_size = 1024          # 经验回放批次大小
actor_lr = 0.01           # Actor网络学习率
critic_lr = 0.01          # Critic网络学习率
```
2. **启动Visdom服务器**
```bash
# 在单独的终端中启动Visdom可视化服务器
python -m visdom.server
# 或指定端口
python -m visdom.server -port 8097

# 访问训练仪表盘：
# http://localhost:8097
```
3. **运行训练脚本**
```bash
# 使用默认参数训练
python main_train.py
```
4. **在 `http://localhost:8097` 监控训练进度**

5. **评估训练模型**
```bash
# 渲染训练好的模型策略
python main_evaluate.py
```

### 🌐 环境定制
[`simple_tag_env.py`](envs/simple_tag_env.py) 扩展了PettingZoo的MPE环境：
- 在 [`custom_agents_dynamics.py`](envs/custom_agents_dynamics.py) 中自定义智能体动力学
- 修改的奖励函数
- 可调节的智能体物理参数

## 📦 数据管理
### 模型存储
训练模型自动保存在：
```tree
./models/
└── maddpg_models/          # MADDPG检查点目录
    ├── {timestamp}_agent_0_actor.pth    # Actor网络参数
    ├── {timestamp}_agent_0_critic.pth   # Critic网络参数
    └── ...                             # 其他智能体网络
```

### 可视化系统
训练指标可视化：
```tree
plot/
├── data/                   # 序列化训练指标
│   └── plot_data_20240515.pkl  # PyTorch张量存储
└── plot_rewards.py         # 可视化工具
```

### 日志系统
实现于 [`logger.py`](utils/logger.py)：
- 记录训练元数据（设备、时长）
- 序列化超参数
- 生成训练报告

```tree
logs/
├── training_log.json       # 可读训练报告
└── plot_data_20240515.pkl  # 原始指标数据
```

## 🔧 高级配置

### 自定义环境
[`simple_tag_env.py`](envs/simple_tag_env.py) 扩展了PettingZoo的MPE环境：
- 在 [`custom_agents_dynamics.py`](envs/custom_agents_dynamics.py) 中修改智能体物理特性
- 可调节的智能体物理参数：
  - 世界大小：2.5单位
  - 时间步长：0.1秒
  - 阻尼系数：0.2
  - 碰撞参数：
    - 接触力：1e2（控制碰撞强度）
    - 接触边界：1e-3（控制碰撞柔软度）

## 🤝 贡献
如遇到任何问题，欢迎提交Issue或Pull Request。
