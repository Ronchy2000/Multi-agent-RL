# Known Issues & Solutions | 已知问题与解决方案

[🇺🇸 English](#english) | [🇨🇳 中文](#chinese)

<a id="english"></a>
## English

This document lists known issues in the project and their solutions.

### Table of Contents
- [Rendering Issues](#rendering-issues)
- [PettingZoo Version Compatibility](#pettingzoo-version-compatibility)
- [Other Common Issues](#other-common-issues)

### Rendering Issues

#### Windows Rendering Unresponsiveness

**Issue**: When using PettingZoo's MPE environment on Windows systems, the rendering window may become unresponsive.

**Solution**:
1. Replace the official `simple_env.py` file with our fixed version:
```bash
# Copy the fixed renderer to your PettingZoo installation path
cp envs/simple_env_fixed_render.py <YOUR_PETTINGZOO_PATH>/pettingzoo/mpe/_mpe_utils/simple_env.py
```

But, I suggest you find the `simple_env.py` file in your PettingZoo installation path and replace it with the fixed version `simple_env_fixed_render.py`. **Copy and paste the code into the file manually.**

2. The key fix is adding proper event handling to ensure it works on all platforms:
```python
# Add event handling to fix rendering issues on Windows
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        pygame.quit()
        return
    if event.type == pygame.WINDOWCLOSE:
        pygame.quit()
        return
pygame.event.pump()  # Ensure the event system runs properly
```
### PettingZoo Version Compatibility
**Issue**: This project requires PettingZoo 1.24.4, but the official PyPI repository only offers version 1.24.3.

**Solution**:
Install version 1.24.4 from GitHub source:
```bash
pip install "pettingzoo[mpe] @ git+https://github.com/Farama-Foundation/PettingZoo.git"
```

Or use the provided installation script:
```python
python utils/setupPettingzoo.py
```

### Other Common Issues
#### Visdom Server Connection Issues

**Issue**: Unable to connect to the Visdom server.

**Solution**:

1. Ensure the Visdom server is running: python -m visdom.server
2. Check if the port is in use, try specifying another port: python -m visdom.server -port 8098
3. Make sure the firewall is not blocking the Visdom service


#### Reward Function Modification
**Issue**: The default reward configuration cannot train good policies, especially for the adversary agents.

**Solution**:
Modify the reward function in the `simple_tag.py` file (located in the PettingZoo MPE library):
1. Set the `shape` parameter to `True` (default is `False`)
2. Add boundary penalties for adversaries
3. Adjust collision reward values

Key modifications example:
```python
    def reward(self, agent, world): # 单个agentd 的 reward
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        # print(f"main_reward{main_reward}")
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = True  # Ronchy 改为True
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            # for adv in adversaries:
            #     rew += 0.1 * np.sqrt(
            #         np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
            #     )
            pass  #Ronchy 修改
        # agent.collide default value is True
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 0  # default value = 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True  #Ronchy 改为True，default: False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                # print("rew_a"   # a 只有一个，所以min无所谓
                #     ,[np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                #     for a in agents])
                rew -= 0.1 * min(  # a 只有一个，所以min无所谓
                    np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents
                )
        if agent.collide:  # 与逃跑者相碰
            for ag in agents:
                for adv in adversaries: 
                    if self.is_collision(ag, adv):
                        rew += 10
        # TODO: 追捕者也要加边界惩罚！
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        
        return rew

```



# Known Issues & Solutions | 已知问题与解决方案
[🇺🇸 English](#english) | 🇨🇳 [中文](#chinese)

<a id="chinese"></a>
## 中文

本文档列出了项目中已知的问题及其解决方案。

### 目录
- [渲染问题](#渲染问题)
- [PettingZoo版本兼容性](#pettingzoo版本兼容性)
- [其他常见问题](#其他常见问题)

### 渲染问题

#### Windows系统渲染无响应

**问题描述**：在Windows系统上，使用PettingZoo的MPE环境时，渲染窗口可能会变得无响应。

**解决方案**：
1. 使用我们修复后的`simple_env.py`文件替换官方版本：
```bash
# 将修复后的渲染器复制到您的PettingZoo安装路径中
cp envs/simple_env_fixed_render.py <YOUR_PETTINGZOO_PATH>/pettingzoo/mpe/_mpe_utils/simple_env.py
```

2. 修复的关键在于添加了适当的事件处理，确保在所有平台上都能正常工作：
```python
# 添加事件处理, 解决windows渲染报错
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        pygame.quit()
        return
    if event.type == pygame.WINDOWCLOSE:
        pygame.quit()
        return
pygame.event.pump()  # 确保事件系统正常运行
```

### PettingZoo版本兼容性
#### 问题描述 

本项目需要PettingZoo 1.24.4版本，但官方PyPI仓库最新版本仅为1.24.3
#### 解决方案
从GitHub源码安装1.24.4版本：
```bash
pip install "pettingzoo[mpe] @ git+https://github.com/Farama-Foundation/PettingZoo.git"
```
或使用提供的安装脚本：
```bash
python utils/setupPettingzoo.py
```

### 其他常见问题

#### Visdom服务器连接问题

**问题描述**：无法连接到Visdom服务器。

**解决方案**：
1. 确保Visdom服务器已启动：`python -m visdom.server`
2. 检查端口是否被占用，可以尝试指定其他端口：`python -m visdom.server -port 8098`
3. 确保防火墙未阻止Visdom服务


#### 奖励函数修改
**问题描述**：官方的奖励配置无法训练出好的效果，特别是追捕者的奖励函数需要修改。

**解决方案**：
修改`simple_tag.py`文件中的奖励函数（位于PettingZoo的MPE库文件中）：
1. 将`shape`参数设置为`True`（默认为`False`）
2. 为追捕者添加边界惩罚
3. 调整碰撞奖励值

关键修改示例：
```python
    def reward(self, agent, world): # 单个agentd 的 reward
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        # print(f"main_reward{main_reward}")
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = True  # Ronchy 改为True
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            # for adv in adversaries:
            #     rew += 0.1 * np.sqrt(
            #         np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
            #     )
            pass  #Ronchy 修改
        # agent.collide default value is True
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 0  # default value = 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True  #Ronchy 改为True，default: False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                # print("rew_a"   # a 只有一个，所以min无所谓
                #     ,[np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                #     for a in agents])
                rew -= 0.1 * min(  # a 只有一个，所以min无所谓
                    np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents
                )
        if agent.collide:  # 与逃跑者相碰
            for ag in agents:
                for adv in adversaries: 
                    if self.is_collision(ag, adv):
                        rew += 10
        # TODO: 追捕者也要加边界惩罚！
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        
        return rew
```

