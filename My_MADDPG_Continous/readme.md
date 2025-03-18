[中文文档](README_zh.md) | [View in Chinese](README_zh.md)

# This is Ronchy's MADDPG project.
| This repo is no longer actively maintained, but feel free to use it - it should still work!

> Current Status: MADDPG implemented in `/agents/*.py`

## Implementation Status
| Algorithm       | Status  | Location          |
|-----------------|---------|-------------------|
| MADDPG          | ✅ 1.0 | `/agents/*.py`    |
| Independent RL  | ⏳ Todo | `agents/independent/` |
| Centralized RL  | ⏳ Todo | `agents/centralized/` |

> Note: MADDPG modules are currently in root agents directory (buffer.py, DDPG_agent.py, etc.), but fully functional!

## Project Structure (Key Files)
```tree
agents/
├── MADDPG_agent.py    # Multi-agent controller
├── DDPG_agent.py      # Single-agent DDPG
├── NN_actor.py        # Actor networks
├── NN_critic.py       # Critic networks
└── buffer.py          # replay buffer
```

## 🛠️ Getting Started
### Prerequisites
```bash
# 1. Create and activate virtual environment (recommended)
***
# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install PettingZoo(only required once)
python utils/setupPettingzoo.py
```

### 🖥️ Runtime Setup
```bash
# Start Visdom visualization server (in separate terminal)
python -m visdom.server
or
python -m visdom.server -port 8097

# Access training dashboard at:
# http://localhost:8097
```

## How to use? 
1. Run `utils/setupPettingzoo.py` to install neccesary packages.
2. Trainning script is given as `./main_train.py`, then the NN model will be stored in `./models/`.
3. Then, before run the `./main_train.py`, please run `python -m visdom.server` in your terminal. 
   (then you can run `./main_train.py` successfully!
4. Furthermore, you can run the `./main_evaluate.py` to evaluate the models you trained.
5. What's more, **parameters** are setting in `./main_parameter.py`


Pull requests issue please, if you meet any bugs.
