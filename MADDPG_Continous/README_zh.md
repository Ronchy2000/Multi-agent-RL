[🇨🇳 中文文档](README_zh.md) | [🇺🇸 English](README.md)

# 多智能体深度强化学习MADDPG算法 - Predator-Prey追逃博弈

![项目状态](https://img.shields.io/badge/状态-重构完成-green) ![MADDPG](https://img.shields.io/badge/MADDPG-已实现-success)![Python](https://img.shields.io/badge/python-3.11.8%2B-blue)

 **本项目专为Predator-Prey追逃博弈任务优化！** 在`PettingZoo MPE`环境基础上重构修改，提供了完整的多智能体协作与对抗环境，适用于围捕控制、群体智能和策略博弈研究。

## 📈 训练效果
<div align="center">
  <img src="./plot/simple_tag_v3_demo_loop.gif" alt="智能体行为" width="45%"/>
  <p><strong>训练后的智能体行为展示：捕食者(红色)追逐猎物(绿色)的过程</strong></p>

  <img src="./plot/demo-rewards_plot_ma.png" alt="训练收敛结果" width="80%"/>
  <p><strong>MADDPG算法在simple_tag_v3环境中的奖励收敛曲线</strong></p>
</div>

> **⚠️ 重要提示**：使用前请查看🔍 [**已知问题与解决方案KNOWN_ISSUES.md**](KNOWN_ISSUES.md)文档，了解常见问题的解决方法，特别是Windows系统的渲染卡死问题和PettingZoo版本兼容性问题。

> **奖励函数修改**：官方的奖励配置无法训练出好的效果，需要修改追捕者的奖励函数

> 当前状态：MADDPG算法已在 `/agents/maddpg/*.py` 中实现

## 🚀 实现进度
| 算法            | 状态   | 位置                  | 核心组件                           |
|----------------|--------|----------------------|----------------------------------|
| MADDPG         | ✅ 1.0 | `agents/maddpg/`   | MADDPG_agent, DDPG_agent, buffer |
| Independent RL | ⏳ 待完成 | `agents/independent/`| IndependentRL (计划中)          |
| Centralized RL | ⏳ 待完成 | `agents/centralized/`| CentralizedRL (计划中)          |

> 注意：MADDPG模块目前位于agents根目录（buffer.py, DDPG_agent.py等），但功能完整可用！

## 🏗️ 项目结构
```tree
MADDPG_Continous/
├── agents/                   # 核心实现
│   ├── maddpg/              # MADDPG算法实现
│   │   ├── MADDPG_agent.py  # 多智能体控制器
│   │   ├── DDPG_agent.py    # 基础DDPG实现
│   │   ├── buffer.py        # 经验回放缓冲区
│   │   └── NN_(actor|critic).py # 神经网络模块
│   ├── Independent/         # 独立RL实现(计划中)
│   └── Centralized/         # 中央化RL实现(计划中)
├── envs/                     # 自定义环境
│   ├── custom_agents_dynamics.py  # 扩展物理引擎
│   └── simple_tag_env.py          # 修改版tag环境
├── utils/                    # 工具模块
│   ├── runner.py             # 训练运行器
│   └── logger.py             # 训练日志系统
├── main_train.py             # 统一训练入口
├── main_evaluate.py          # 统一评估入口
├── main_evaluate_save_render2gif.py # 渲染并保存GIF
└── main_parameters.py        # 统一参数配置
```

## 🛠️ 快速开始

### 环境配置

> 相关配置需求在utils/文件夹下。

# 创建并激活虚拟环境（推荐）
1. 使用conda-environment.yml创建新环境
```bash
# 注意：将"MPE"替换为您喜欢的环境名称
conda env create -f utils/conda-environment.yml -n MPE
# 激活刚创建的环境
conda activate MPE
```
2. pip安装核心依赖
```bash
pip install -r utils/pip-requirements.txt
```
3. 从PyTorch官网安装对应版本的PyTorch
```bash
# 请访问 https://pytorch.org 选择适合您系统的安装命令
# 例如：
pip3 install torch torchvision torchaudio
```
4. 安装PettingZoo 1.24.4版本
```bash
# 重要说明：本项目需要PettingZoo 1.24.4版本，但官方PyPI仓库最新版本仅为1.24.3
# 必须从GitHub源码安装才能获取1.24.4版本，安装命令为：
pip install "pettingzoo[mpe] @ git+https://github.com/Farama-Foundation/PettingZoo.git"
# 或者，您可以直接运行提供的安装脚本：
python utils/setupPettingzoo.py
```

### 🖥️ 运行配置
> **注意：** 为简化使用，当前版本已不再依赖Visdom进行可视化，您可跳过下述visdom配置，但保留相关配置供需要时参考。

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
[`envs/simple_tag_env.py`](envs/simple_tag_env.py) 扩展了PettingZoo的MPE环境：
- 在 [`envs/custom_agents_dynamics.py`](envs/custom_agents_dynamics.py) 中自定义智能体动力学
- 修改的奖励函数，专为Predator-Prey任务优化
- 可调节的智能体物理参数：
  - 世界大小：2.5单位（可根据追逃需求自定义）
  - 时间步长：0.1秒（影响动作响应速度）
  - 阻尼系数：0.2（影响智能体的惯性）
  - 碰撞参数：
    - 接触力：1e2（控制碰撞强度，影响围捕效果）
    - 接触边界：1e-3（控制碰撞柔软度）

#### 🔄 自定义追逃场景
您可以轻松配置自己的追逃环境：
- 自定义Predator数量、速度和加速度
- 配置Evader的逃跑策略和敏捷度
- 设计围捕奖励机制，鼓励协作或竞争行为
- 实现复杂地形和障碍物（通过自定义碰撞处理）


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


## 🐛 已知问题与解决方案
我们整理了一份详细的已知问题及其解决方案文档，包括：
- **Windows系统渲染无响应问题**：修复PettingZoo的渲染问题
- **PettingZoo版本兼容性问题**：本项目需要1.24.4版本
- **Visdom服务器连接问题**：确保可视化服务正常运行
- **奖励函数修改**：官方的奖励配置无法训练出好的效果，需要修改追捕者的奖励函数
👉 **[点击查看完整的已知问题与解决方案文档](KNOWN_ISSUES.md)**

如果您遇到文档中未提及的问题，请在Issues中提交，我们将尽快解决。

## 🤝 贡献
本项目的主要贡献在于：
- 针对Predator-Prey追逃博弈任务的环境适配与优化
- 改进的奖励函数设计，解决官方环境训练效果不佳的问题
- 灵活的围捕控制参数配置，支持多种追逃场景

如遇到任何问题，欢迎提交Issue或Pull Request。若您有兴趣扩展更多追逃博弈场景，欢迎您的贡献！
