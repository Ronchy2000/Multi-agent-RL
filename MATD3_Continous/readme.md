[🇨🇳 中文文档](readme.md) | [🇺🇸 English](readme_en.md)

# 多智能体深度强化学习MATD3算法 - Predator-Prey追逃博弈

>**本项目专为Predator-Prey追逃博弈任务优化！** 基于TD3算法的多智能体扩展版本(MATD3:Twin Delayed Deep Deterministic Policy Gradient)，在`PettingZoo MPE`环境基础上重构修改，提供了完整的多智能体协作与对抗环境，专注于连续动作空间的多智能体协作与对抗任务;适用于围捕控制、群体智能和策略博弈研究.

> MATD3算法优势：相比MADDPG，通过双Q网络和目标策略平滑机制有效解决过估计问题，提供更稳定的训练和更优的策略。

> Reference: https://github.com/wild-firefox/FreeRL/blob/main/MADDPG_file/MATD3_simple.py

## 📈 训练效果
<div align="center">
  <img src="./plot/simple_tag_v3_matd3_demo.gif" alt="MATD3算法表现" width="50%"/>
  <p><strong>MATD3算法在simple_tag_v3环境中的表现（追逃博弈 pursuit-evasion game）</strong></p>
</div>

<div align="center">
  <img src="./plot/training_rewards_demo.png" alt="训练收敛结果" width="80%"/>
  <p><strong>MATD3算法在simple_tag_v3环境中的奖励收敛曲线</strong></p>
</div>

> **⚠️ 重要提示**：使用前请查看🔍 [**已知问题与解决方案KNOWN_ISSUES.md**](KNOWN_ISSUES.md)文档，了解常见问题的解决方法，特别是Windows系统的渲染卡死问题和PettingZoo版本兼容性问题。

> **奖励函数优化**：官方的奖励配置无法训练出良好的围捕行为，本项目专门优化了追捕者的奖励函数，实现更高效的协作围捕

## 🚀 实现进度
| 算法          | 状态   | 位置                | 核心组件                         |
|--------------|--------|-------------------|----------------------------------|
| MATD3        | ✅ 1.0 | `agents/`   | MATD3_agent, buffer, networks    |


## 项目结构

```tree
MATD3_Continous/
├── agents/                   # 智能体算法实现
│   ├── buffer.py            # 经验回放缓冲区
│   ├── MATD3_agent.py       # MATD3智能体控制器
│   ├── MATD3_runner.py      # 训练与评估运行器
│   ├── NN_actor_td3.py      # Actor网络结构
│   ├── NN_critic_td3.py     # Critic网络结构(双Q网络)
│   └── TD3_agent.py         # 基础TD3实现
├── envs/                     # 环境实现
│   ├── custom_agents_dynamics.py  # 自定义智能体动力学
│   └── simple_tag_env.py    # 修改版追逃环境
├── main/                     # 主程序脚本
│   ├── main_evaluate.py     # 评估脚本
│   ├── main_parameters.py   # 参数配置
│   └── main_train.py        # 训练入口
├── plot/                     # 数据可视化
│   ├── matd3_data/          # 训练数据存储
│   ├── plot_rewards.py      # 奖励绘图脚本
│   ├── README.md            # 绘图说明
│   └── training_rewards_demo.png  # 样例训练曲线
├── logs/                     # 日志文件
│   └── log_td3_main/        # TD3训练日志
└── utils/                    # 工具函数
    ├── conda-environment.yml  # Conda环境配置(Windows和Intel芯片的macOS)
    ├── linux_environment.yml  # Linux环境配置
    ├── logger.py            # 日志工具
    ├── mac_arm_M4_environment.yml  # Mac M系列芯片环境配置
    ├── pip-requirements.txt  # pip依赖
    ├── pip-requirements_mac_arm_M4.txt  # Mac M系列芯片专用依赖
    └── setupPettingzoo.py   # PettingZoo环境设置
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


## 🛠️ 快速开始

### 环境配置

> 相关配置需求在utils/文件夹下。

### Linux环境（ubuntu）
1. 使用linux_environment.yml创建新环境
```bash
# 注意：将"MPE"替换为您喜欢的环境名称
conda env create -f utils/linux_environment.yml -n MPE
# 激活刚创建的环境
conda activate MPE
```
2. pip安装核心依赖
```bash
pip install -r utils/pip-requirements.txt
```
### Mac M系列芯片环境
1. 使用mac_arm_M4_environment.yml创建新conda环境
```bash
# 注意：将"MPE"替换为您喜欢的环境名称
conda env create -f utils/mac_arm_M4_environment.yml -n MPE
# 激活刚创建的环境
conda activate MPE
```
2. pip安装Mac M芯片专用依赖
```bash
pip install -r utils/pip-requirements_mac_arm_M4.txt
```

### Windows创建并激活虚拟环境（推荐）
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
### 手动安装依赖
> 上述虚拟环境创建成功后，您需要手动安装以下依赖：
3. 从PyTorch官网安装对应版本的PyTorch
```bash
# 请访问 https://pytorch.org 选择适合您系统的安装命令
# 例如：
pip3 install torch torchvision torchaudio
```

4. 2025.4.26 update: 安装`PettingZoo 1.25.0`版本，官方PyPI仓库最新版本更新为为1.25.0，内容与1.24.4相同。MPE被拆分出PettingZoo, **警告可忽略**，`MPE2`详情可见:https://github.com/Farama-Foundation/MPE2
```bash
pip install pettingzoo==1.25.0
```

4. ~~安装PettingZoo 1.24.4版本~~
```bash
# 重要说明：本项目需要PettingZoo 1.24.4版本，但官方PyPI仓库最新版本仅为1.24.3
# 必须从GitHub源码安装才能获取1.24.4版本，安装命令为：
# pip install "pettingzoo[mpe] @ git+https://github.com/Farama-Foundation/PettingZoo.git"
或者，您可以直接运行提供的安装脚本安装pettingzoo1.25.0：
python utils/setupPettingzoo.py
```

### 🖥️ 运行配置
> **注意：** 当前版本采用本地数据存储模式，无需额外配置可视化服务器。训练数据将保存在plot/matd3_data/目录下。

## 🔄 训练流程
1. **参数配置**   
在 `main_parameters.py` 中设置环境和算法参数：
```python
env_name = 'simple_tag_v3'  # 可选：simple_adversary_v3/simple_spread_v3
episode_num = 5000         # 总训练回合数
# 训练参数
batch_size = 128          # 经验回放批次大小
actor_lr = 0.0002         # Actor网络学习率
critic_lr = 0.002         # Critic网络学习率
```

2. **运行训练脚本**
```bash
# 使用默认参数训练
cd main
python main_train.py
```

3. **查看训练进度**
训练数据将实时保存到CSV文件中，可使用plot_rewards.py脚本进行可视化：
```bash
python plot/plot_rewards.py
```

4. **评估训练模型**
```bash
# 渲染训练好的模型策略
cd main
python main_evaluate.py
```

### 🌐 环境特性与优化
本项目基于PettingZoo的MPE环境进行了大量优化：

- **TD3增强的策略稳定性**: 相比MADDPG，MATD3通过双Q网络和目标策略平滑有效解决过估计问题
- **围捕行为的奖励优化**: 通过精心设计的奖励函数，实现更具协作性的围捕策略
- **物理参数优化**: 
  - 世界大小：2.5单位（可根据追逃需求自定义）
  - 时间步长：0.1秒（影响动作响应速度）
  - 阻尼系数：0.2（影响智能体的惯性）

#### 🌟 MATD3 vs MADDPG
MATD3对标准MADDPG进行了以下关键增强：

1. **双Q网络设计**: 减少对动作值的过估计
2. **延迟策略更新**: 提高训练稳定性
3. **目标策略平滑**: 通过在目标动作中加入噪声防止过拟合
4. **自适应噪声调整**: 根据训练进度动态调整探索噪声

这些优化使MATD3在追逃博弈场景中展现出更强大的性能和更快的收敛速度。

## 📦 数据管理
### 模型存储
训练模型自动保存在：
```tree
./main/models/
└── matd3_models/           # MATD3检查点目录
    ├── {timestamp}_agent_0_actor.pth    # Actor网络参数
    ├── {timestamp}_agent_0_critic_1.pth # 第一个Critic网络参数
    ├── {timestamp}_agent_0_critic_2.pth # 第二个Critic网络参数
    └── ...                             # 其他智能体网络
```

### 可视化系统
训练指标可视化：
```tree
plot/
├── matd3_data/             # 训练数据存储
│   └── rewards_{timestamp}.csv   # CSV格式奖励记录
└── plot_rewards.py         # 可视化工具
```

## 🤝 贡献
本项目的主要贡献在于：
- TD3算法在多智能体场景下的扩展与优化
- 针对Predator-Prey追逃博弈任务的环境适配与优化
- 改进的奖励函数设计，实现高效的围捕协作行为
- 稳定的训练框架，支持各种复杂追逃场景

如遇到任何问题，欢迎提交Issue或Pull Request。若您有兴趣扩展更多追逃博弈场景或改进算法，我们欢迎您的贡献！