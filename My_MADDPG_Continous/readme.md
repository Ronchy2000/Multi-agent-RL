# This is Ronchy's MADDPG project.
| This repo is no longer actively maintained, but feel free to use it - it should still work!

## How to use? 
1. Run `setupPettingzoo.py` to install neccesary packages.
2. Trainning script is given as `main_train.py`, then the NN model will be stored in `.\models\`.
3. Then, before run the `main_train.py`, please run `python -m visdom.server` in your terminal. 
   (then you can run `main_train.py` successfully!
4. Furthermore, you can run the `main_evaluate.py` to evaluate the models you trained.
5. What's more, **parameters** are setting in `main_parameter.py`


Pull requests issue please, if you meet any bugs.


# This is Ronchy's Multi-Agent RL project
| 2025.2.17 updated.

本项目实现了三种多智能体强化学习算法：MADDPG、Independent RL 和 Centralized RL。

## 算法实现
### 1. MADDPG (Multi-Agent DDPG)
- 位置：`agents/maddpg/`
- 特点：采用集中训练、分散执行的架构
- 适用：多智能体合作与竞争场景

### 2. Independent RL
- 位置：`agents/independent/`
- 特点：每个智能体独立学习，不考虑其他智能体的行为
- 适用：智能体间交互较少的场景

### 3. Centralized RL
- 位置：`agents/centralized/`
- 特点：完全集中式训练和执行
- 适用：需要全局最优解的场景

## 如何使用
1. 运行 `setupPettingzoo.py` 安装必要的包
2. 训练脚本为 `main_train.py`，模型将保存在 `./models/`
3. 运行训练前，请先在终端执行 `python -m visdom.server`
4. 使用 `main_evaluate.py` 评估训练好的模型
5. 参数配置在 `main_parameter.py` 中设置

## 项目结构
```tree
My_MADDPG_Continous/
├── agents/
│   ├── maddpg/                 # 现有的MADDPG实现
│   │   ├── DDPG_agent.py
│   │   ├── MADDPG_agent.py
│   │   ├── NN_actor.py
│   │   └── NN_critic.py
│   ├── independent/            # Independent RL实现
│   │   ├── DDPG_agent.py
│   │   ├── IndependentRL.py
│   │   ├── NN_actor.py
│   │   └── NN_critic.py
│   └── centralized/           # Centralized RL实现
│       ├── DDPG_agent.py
│       ├── CentralizedRL.py
│       ├── NN_actor.py
│       └── NN_critic.py
├── main_train.py              # 统一的训练入口
├── main_evaluate.py           # 统一的评估入口
└── algorithms/                # 算法选择和配置
    ├── algorithm_config.py    # 算法配置文件
    └── algorithm_factory.py   # 工厂模式创建算法实例