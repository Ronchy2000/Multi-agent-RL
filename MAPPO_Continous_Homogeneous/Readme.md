# MAPPO for Homogeneous Multi-Agent Reinforcement Learning

This MAPPO implementation is designed for homogeneous agents, like the Simple_spread_v3 environment.

## Features

- ✅ **Homogeneous Agent Support**: All agents share the same observation and action space dimensions
- ✅ **Reproducible Training**: Built-in seed management for reproducible experiments
- ✅ **Continuous Action Space**: Supports continuous control tasks
- ✅ **Visualization Support**: Training and evaluation with rendering support

## Quick Start

### Training

```bash
python MAPPO_main.py
```

### Evaluation

```bash
python MAPPO_evaluate.py
```

### Seed Configuration

The implementation includes comprehensive seed support for reproducibility:

- **PyTorch Seed**: For neural network initialization and training
- **NumPy Seed**: For data processing operations
- **Environment Seed**: For PettingZoo environment randomness

To set a fixed seed, modify the seed parameter in the main script:

```python
# In MAPPO_main.py or MAPPO_evaluate.py
runner = Runner_MAPPO_MPE(args, number=2, seed=23)  # Set seed=23 for reproducibility
```

Set `seed=None` for random initialization in each run.