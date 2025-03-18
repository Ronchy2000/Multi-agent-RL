[🇨🇳 中文文档](README_zh.md) | [🇺🇸 English](README.md)

# This is Ronchy's MADDPG project.

![Status](https://img.shields.io/badge/status-archived-red) ![MADDPG](https://img.shields.io/badge/MADDPG-implemented-success) ![Python](https://img.shields.io/badge/python-3.11.8%2B-blue)

> **⚠️ Important Note**: Before using, please check the 🔍 [**Known Issues & Solutions**](KNOWN_ISSUES.md) document to understand common problems and their solutions, especially Windows rendering issues and PettingZoo version compatibility.

> **Note**: This repo is no longer actively maintained, but feel free to use it - it should still work!
>
> Current Status: MADDPG implemented in `/agents/*.py`

## 🚀 Implementation Status
| Algorithm       | Status  | Location                | Components                          |
|-----------------|---------|-------------------------|------------------------------------|
| MADDPG          | ✅ 1.0  | `agents/*.py`           | MADDPG_agent, DDPG_agent, buffer   |
| Independent RL  | ⏳ WIP  | `agents/independent/`   | IndependentRL (planned)            |
| Centralized RL  | ⏳ WIP  | `agents/centralized/`   | CentralizedRL (planned)            |

> **Note**: MADDPG modules are currently in the root agents directory (buffer.py, DDPG_agent.py, etc.), but are fully functional!

## 🏗️ Project Structure (Key Files)
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
└── main_parameters.py        # Unified parameters config
```
## Other Common Issues
For other common issues and their solutions, please check the Issues section of this repository.


## 🛠️ Getting Started
### Prerequisites
```bash
# 1. Create and activate virtual environment (recommended)
# Note: Replace "MPE" with your preferred environment name
conda env create -f utils/conda-environment.yml -n MPE  
#then, activate env.
conda activate MPE

# 2. Install core dependencies
pip install -r utils/pip-requirements.txt

# 3. Install PyTorch
# Visit https://pytorch.org/ to select the appropriate installation command for your system
# For example:
pip3 install torch torchvision torchaudio

# 4. Install PettingZoo 1.24.4
# Important: This project requires PettingZoo 1.24.4, but the official PyPI repository only offers version 1.24.3
# You must install from GitHub source to get version 1.24.4 using:
pip install "pettingzoo[mpe] @ git+https://github.com/Farama-Foundation/PettingZoo.git"

# Alternatively, you can use the provided installation script:
python utils/setupPettingzoo.py
```


## 🔄 Training Pipeline
1. **Parameter Customization**  
Configure environment parameters in [`main_parameter.py`](main_parameters.py)
``` bash
   env_name = 'simple_tag_v3'  # Options: simple_adversary_v3/ simple_spread_v3
   episode_num = 2000         # Total training episodes
   # Training parameters
   batch_size = 1024          # Experience replay batch  size
   actor_lr = 0.01            # Actor network learning   rate
   critic_lr = 0.01           # Critic network learning  rate
```
2. **Start Visdom server**
```python
# Start Visdom visualization server (in separate terminal)
   python -m visdom.server
   or
   python -m visdom.server -port 8097

# Access training dashboard at:
# http://localhost:8097
```
3. **Run training script**:
```python
# Train with custom parameters
   python main_train.py
```
4. **Monitor training progress at `http://localhost:8097`**
5. **Evaluate trained models**
```python
   python main_evaluate.py
```

### 🌐 Environment Customization
The [`simple_tag_env.py`](envs/simple_tag_env.py)  extends PettingZoo's MPE environment with:
- Custom agent dynamics in [`custom_agents_dynamics.py`](envs/custom_agents_dynamics.py)
- Modified reward functions
- Adjustable agent physics parameters:
- World size: 2.5 units
- Time step: 0.1s
- Damping coefficient: 0.2
- Collision parameters:
  - Contact force: 1e2 (controls collision intensity)
  - Contact margin: 1e-3 (controls collision softness)

## 📦 Data Management
### Model Storage
Trained models are automatically saved with timestamps:
```tree
./models/
└── maddpg_models/          # MADDPG checkpoint directory
    ├── {timestamp}_agent_0_actor.pth    # Actor network parameters
    ├── {timestamp}_agent_0_critic.pth   # Critic network parameters
    └── ...  
```
### Visualization Pipeline
```tree
plot/
├── data/                   # Serialized training metrics
│   └── plot_data_20240515.pkl  # PyTorch tensor storage
└── plot_rewards.py         # Visualization toolkit
```
### Logging System
Implemented in [logger.py](utils/logger.py) :
- Records training metadata (device, duration)
- Serializes hyperparameters
- Generates human-readable training reports

```tree
logs/
├── training_log.json       # Human-readable training report
└── plot_data_20240515.pkl  # Raw metrics for post-analysis
```

## 🐛 Known Issues & Solutions
We have compiled a detailed document of known issues and their solutions, including:
- **Windows Rendering Unresponsiveness**: Fixes for PettingZoo rendering issues
- **PettingZoo Version Compatibility**: This project requires version 1.24.4
- **Visdom Server Connection Issues**: Ensuring visualization services run properly

👉 **[Click to view the complete Known Issues & Solutions document](KNOWN_ISSUES.md)**

If you encounter issues not mentioned in the document, please submit them in the Issues section and we will address them promptly.


## 🤝 Contributing
If you encounter any issues, please submit a Pull Request or open an Issue.
