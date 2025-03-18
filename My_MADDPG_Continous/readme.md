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
My_MADDPG_Continous/
├── agents/                   # Core implementations
│   ├── MADDPG_agent.py       # Multi-agent controller
│   ├── DDPG_agent.py         # Base DDPG implementation
│   ├── buffer.py             # Experience replay buffer
│   └── (NN_actor|NN_critic).py  # Neural network modules
├── envs/                     # Custom environments
│   ├── custom_agents_dynamics.py  # Extended physics engine
│   └── simple_tag_env.py           # Modified tag environment
├── utils/                    # Utility modules
│   ├── runner.py             # Training runner
│   └── logger.py             # Training logger
│── main_train.py             # Unified training entry
│── main_evaluate.py          # Unified evaluate model entry
└── main_parameters.py        # Unified parameters entry
```

## 🛠️ Getting Started
### Prerequisites
```bash
# 1. Create and activate virtual environment (recommended)
      # omit...
# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install PettingZoo(only required once)
python utils/setupPettingzoo.py
```


## 🔄 Training Pipeline
1. **Parameter Customization**  
Configure environment parameters in [`main_parameter.py`](main_parameters.py)
```python
   env_name = 'simple_tag_v3'  # Options: simple_adversary_v3/ simple_spread_v3
   episode_num = 2000         # Total training episodes
   # Training parameters
   batch_size = 1024          # Experience replay batch  size
   actor_lr = 0.01            # Actor network learning   rate
   critic_lr = 0.01           # Critic network learning  rate
```
2. **Start Visdom server**
```bash
# Start Visdom visualization server (in separate terminal)
python -m visdom.server
or
python -m visdom.server -port 8097

# Access training dashboard at:
# http://localhost:8097
```
3. **Run training script**:
```bash
   # Train with custom parameters
   python main_train.py
```
4. **Monitor training progress at `http://localhost:8097`**
5. **Evaluate trained models**
```bash
   python main_evaluate.py
```

### 🌐 Environment Customization
The [`simple_tag_env.py`](envs/simple_tag_env.py)  extends PettingZoo's MPE environment with:
- Custom agent dynamics in [`custom_agents_dynamics.py`](envs/custom_agents_dynamics.py)
- Modified reward functions
- Adjustable agent physics parameters

## 🤝 Contributing
Pull requests issue please, if you meet any bugs.
