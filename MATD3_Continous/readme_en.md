[🇨🇳 Chinese](readme.md) | [🇺🇸 English](readme_en.md)

# Multi-Agent Deep Reinforcement Learning MATD3 Algorithm - Predator-Prey Game

>**This project is specially optimized for Predator-Prey pursuit games!** Based on the multi-agent extension of TD3 algorithm (MATD3: Twin Delayed Deep Deterministic Policy Gradient), rebuilt on the `PettingZoo MPE` environment, it provides a complete multi-agent cooperative and competitive environment, focusing on multi-agent tasks in continuous action spaces; suitable for pursuit control, swarm intelligence, and strategic game research.

> MATD3 Algorithm Advantage: Compared to MADDPG, it effectively solves overestimation problems through double Q-networks and target policy smoothing mechanisms, providing more stable training and better policies.

> Reference: https://github.com/wild-firefox/FreeRL/blob/main/MADDPG_file/MATD3_simple.py

## 📈 Training Results
<div align="center">
  <img src="./plot/simple_tag_v3_matd3_demo.gif" alt="MATD3 performance" width="50%"/>
  <p><strong>MATD3 performance in simple_tag_v3 (pursuit-evasion game)</strong></p>
</div>

<div align="center">
  <img src="./plot/training_rewards_demo.png" alt="Training Convergence" width="80%"/>
  <p><strong>Reward convergence curve of MATD3 algorithm in simple_tag_v3 environment</strong></p>
</div>

> **⚠️ Important Note**: Before using, please check the 🔍 [**Known Issues & Solutions**](KNOWN_ISSUES.md) document to understand common problems and their solutions, especially Windows rendering issues and PettingZoo version compatibility.

> **Reward Function Optimization**: The official reward configuration cannot train good capture behaviors. This project specifically optimizes the predator's reward function to achieve more efficient cooperative capture. For more details please [replace official files with our settings.](./envs)

## 🚀 Implementation Progress
| Algorithm    | Status | Location          | Core Components                    |
|--------------|--------|-------------------|----------------------------------|
| MATD3        | ✅ 1.0 | `agents/`         | MATD3_agent, buffer, networks    |


## Project Structure

```tree
MATD3_Continous/
├── agents/                   # Agent algorithm implementation
│   ├── buffer.py            # Experience replay buffer
│   ├── MATD3_agent.py       # MATD3 agent controller
│   ├── MATD3_runner.py      # Training and evaluation runner
│   ├── NN_actor_td3.py      # Actor network structure
│   ├── NN_critic_td3.py     # Critic network structure (double Q-networks)
│   └── TD3_agent.py         # Base TD3 implementation
├── envs/                     # Environment implementation
│   ├── custom_agents_dynamics.py  # Custom agent dynamics
│   └── simple_tag_env.py    # Modified pursuit environment
├── main/                     # Main program scripts
│   ├── main_evaluate.py     # Evaluation script
│   ├── main_parameters.py   # Parameter configuration
│   └── main_train.py        # Training entry
├── plot/                     # Data visualization
│   ├── matd3_data/          # Training data storage
│   ├── plot_rewards.py      # Reward plotting script
│   ├── README.md            # Plotting instructions
│   └── training_rewards_demo.png  # Sample training curve
├── logs/                     # Log files
│   └── log_td3_main/        # TD3 training logs
└── utils/                    # Utility functions
    ├── conda-environment.yml  # Conda environment configuration(Windows & macOS with intel chip)
    ├── linux_environment.yml  # Linux Environment configuration(Linux)
    ├── logger.py            # Logging tool
    ├── mac_arm_M4_environment.yml  # Mac M Series Chip environment configuration(macOS M Series Chip)
    ├── pip-requirements.txt  # pip dependencies
    ├── pip-requirements_mac_arm_M4.txt  # Mac M Series Chip environment configuration(macOS M Series Chip)
    └── setupPettingzoo.py   # PettingZoo environment setup
```

## Environment Description

This project is based on PettingZoo's MPE (Multi-Particle Environment), mainly implementing the simple_tag pursuit-evasion task:

- **Pursuers (Adversaries)**: Multiple pursuers cooperatively chase the evader
- **Evaders (Good Agents)**: Try to escape from pursuers

Environment features:
- Continuous action space
- Partially observable states
- Multi-agent cooperation and competition

## Algorithm Implementation

The project implements the MATD3 (Multi-Agent Twin Delayed Deep Deterministic Policy Gradient) algorithm, which is a multi-agent extension of the TD3 algorithm, with the following key features:

- Double Q-networks to reduce overestimation
- Delayed policy updates
- Target policy smoothing regularization
- Centralized Training with Decentralized Execution (CTDE) paradigm


## 🛠️ Quick Start

### Environment Setup

> Configuration requirements are in the utils/ folder.

### Linux Environment
1. Create a new environment using linux_environment.yml
```bash
# Note: Replace "MPE" with your preferred environment name
conda env create -f utils/linux_environment.yml -n MPE
# Activate the newly created environment
conda activate MPE
```
2. Install core dependencies with pip
```bash
pip install -r utils/pip-requirements.txt
```
### Mac M Series Chip Environment
1. Create a new environment using mac_arm_M4_environment.yml
```bash
# Note: Replace "MPE" with your preferred environment name
conda env create -f utils/mac_arm_M4_environment.yml -n MPE
# Activate the newly created environment
conda activate MPE
```
2. Install Mac M chip-specific dependencies
```bash
pip install -r utils/pip-requirements_mac_arm_M4.txt
```

### Windows Environment
1. Create and activate virtual environment (recommended)
```bash
# Note: Replace "MPE" with your preferred environment name
conda env create -f utils/conda-environment.yml -n MPE  
# Activate the newly created environment
conda activate MPE
```
2. Install core dependencies
```bash
pip install -r utils/pip-requirements.txt
```

### Other Prerequisites
> Then, install other prerequisites after creating new environment.
3. Install PyTorch
```bash
# Visit https://pytorch.org/ to select the appropriate installation command for your system
# For example:
pip3 install torch torchvision torchaudio
```
4. 2025.4.26 update: Install PettingZoo 1.25.0 version, the official PyPI repository has been updated to 1.25.0, with the same content as 1.24.4. MPE has been separated from PettingZoo, **warnings can be ignored**, see MPE2 for details: https://github.com/Farama-Foundation/MPE2
```bash
pip install pettingzoo==1.25.0
```
Alternatively, you can use the provided installation script to install PettingZoo 1.25.0:
```bash
python utils/setupPettingzoo.py
```

4. ~~Install PettingZoo version 1.24.4~~
```bash
# Important note: This project requires PettingZoo version 1.24.4, but the latest version in the official PyPI repository may not be fully compatible
# It is recommended to install from GitHub source code with the command:
#pip install "pettingzoo[mpe] @ git+https://github.com/Farama-Foundation/PettingZoo.git"
# Alternatively, you can run the provided installation script:
python utils/setupPettingzoo.py
```

### 🖥️ Runtime Configuration
> **Note:** The current version uses local data storage mode, requiring no additional visualization server configuration. Training data will be saved in the plot/matd3_data/ directory.

## 🔄 Training Process
1. **Parameter Configuration**   
Set environment and algorithm parameters in `main_parameters.py`:
```python
env_name = 'simple_tag_v3'  # Options: simple_adversary_v3/simple_spread_v3
episode_num = 5000         # Total training episodes
# Training parameters
batch_size = 128          # Experience replay batch size
actor_lr = 0.0002         # Actor network learning rate
critic_lr = 0.002         # Critic network learning rate
```

2. **Run training script**
```bash
# Train with default parameters
cd main
python main_train.py
```

3. **View training progress**
Training data will be saved to CSV files in real-time. You can visualize it using the plot_rewards.py script:
```bash
python plot/plot_rewards.py
```

4. **Evaluate trained models**
```bash
# Render trained model policies
cd main
python main_evaluate.py
```

### 🌐 Environment Features and Optimizations
This project has made numerous optimizations to the PettingZoo MPE environment:

- **TD3-Enhanced Policy Stability**: Compared to MADDPG, MATD3 effectively solves overestimation problems through double Q-networks and target policy smoothing
- **Capture Behavior Reward Optimization**: Through carefully designed reward functions, more cooperative capture strategies are achieved
- **Physics Parameter Optimization**: 
  - World size: 2.5 units (customizable for pursuit-evasion needs)
  - Time step: 0.1 seconds (affects action response speed)
  - Damping coefficient: 0.2 (affects agent inertia)

#### 🌟 MATD3 vs MADDPG
MATD3 enhances standard MADDPG with these key improvements:

1. **Double Q-Network Design**: Reduces overestimation of action values
2. **Delayed Policy Updates**: Improves training stability
3. **Target Policy Smoothing**: Prevents overfitting by adding noise to target actions
4. **Adaptive Noise Adjustment**: Dynamically adjusts exploration noise based on training progress

These optimizations enable MATD3 to demonstrate stronger performance and faster convergence in pursuit-evasion game scenarios.

## 📦 Data Management
### Model Storage
Training models are automatically saved in:
```tree
./main/models/
└── matd3_models/           # MATD3 checkpoint directory
    ├── {timestamp}_agent_0_actor.pth    # Actor network parameters
    ├── {timestamp}_agent_0_critic_1.pth # First critic network parameters
    ├── {timestamp}_agent_0_critic_2.pth # Second critic network parameters
    └── ...                             # Other agent networks
```

### Visualization System
Training metrics visualization:
```tree
plot/
├── matd3_data/             # Training data storage
│   └── rewards_{timestamp}.csv   # Reward records in CSV format
└── plot_rewards.py         # Visualization tool
```

## 🤝 Contributing
This project's main contributions include:
- Extension and optimization of TD3 algorithm in multi-agent scenarios
- Environment adaptation and optimization specifically for Predator-Prey pursuit games
- Improved reward function design, achieving efficient cooperative capture behavior
- Stable training framework supporting various complex pursuit scenarios

If you encounter any issues, please submit an Issue or Pull Request. If you're interested in extending more pursuit game scenarios or improving the algorithm, your contributions are especially welcome!