from pettingzoo.mpe import simple_v3


### AEC - Agent Environment Cycle
# 串行执行：在 AEC 模式下，智能体是按顺序一个一个地进行动作和观察的。这意味着每个时间步只有一个智能体在执行动作，其他智能体在等待。

### Parallel
# 在 Parallel 模式下，所有智能体同时进行动作和观察。这意味着每个时间步所有智能体都在执行动作，没有等待时间。
from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()

