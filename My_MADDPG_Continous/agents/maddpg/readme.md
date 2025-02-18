2025.2.18

TODO:

1. 经典MADDPG：Critic是集中式的（全局输入），但每个智能体可独立更新Critic。  （需找经典代码）~


# 一、独立Critic网络的核心意义
1. 异构奖励函数支持
竞争场景：若智能体间存在利益冲突（如对抗游戏），每个Critic需学习不同的Q函数以反映各自奖励目标。
例如：足球游戏中，进攻方Critic需评估射门收益，防守方Critic需评估拦截收益。
混合协作场景：部分智能体可能有辅助性奖励（如无人机编队中的领航者与跟随者）。

2. 策略独立性
策略空间差异：即使输入相同，不同智能体的Actor网络输出动作分布不同，Critic需独立评估各自策略的全局影响。
非对称学习速率：独立Critic允许智能体以不同速度学习，避免共享网络导致的策略耦合震荡。
3. 实现灵活性
扩展性：支持未来扩展至异构观测/动作空间（如部分智能体为连续控制，其他为离散决策）。
调试便利：独立网络便于单独监控和调整特定智能体的学习过程。

# 二、输入相同时的Critic差异性来源
即使Critic输入相同（所有Agent的obs+actions），以下因素仍会导致各Critic输出不同：

1. 网络参数独立性
初始随机化：独立网络参数初始值不同，导致梯度更新路径分化。
优化过程差异：不同Critic的优化器状态（如动量）独立积累。
2. 目标Q值差异
奖励函数不同：若 r_i ≠ r_j，目标Q值 target_q = r_i + γQ' 直接不同。
下一状态动作差异：不同智能体的目标Actor生成的动作策略不同（如进攻者选择突破，防守者选择拦截）。
3. 环境动力学影响
状态转移差异：不同智能体对环境的改变方式不同（如机器人推箱子任务中，不同推法导致不同后续状态）。
# 三、独立Critic的代价与优化
1. 计算开销分析
训练速度：独立Critic的并行计算可通过GPU批处理缓解，实际影响有限。
内存占用：网络参数数量与智能体数量线性增长，可通过网络结构简化（如共享隐层）优化。
2. 优化策略
参数共享试探：在同构完全协作场景中，可尝试同类智能体共享Critic。
```
{
# 示例：追击者共享Critic
class SharedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(global_input_dim, 64)
}
```
        
# 初始化时分配共享实例
chaser_critic = SharedCritic()
for agent in chaser_agents:
    agent.critic = chaser_critic}
# 初始化时分配共享实例
chaser_critic = SharedCritic()
for agent in chaser_agents:
    agent.critic = chaser_critic
分布式训练：利用多GPU或Ray框架实现并行更新。
# 四、场景驱动的设计选择

|场景类型|推荐架构|理由|
|---|---|---|
完全协作+同构|	共享Critic（同类智能体）	|减少冗余计算，利用环境对称性<br>
竞争/混合奖励|	独立Critic|	反映不同奖励函数和策略目标
异构观测/动作空间|	独立Critic|	适应不同输入输出维度
初步算法验证|	独立Critic|	实现简单，避免共享逻辑复杂性

# 五、代码实现对比解析
### 用户代码1（混合MADDPG/DDPG）
https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/algorithms/maddpg.py <br>
1. Critic输入：<br>
    - MADDPG模式：全局obs+actions → 输入相同但Critic独立。<br>
    - DDPG模式：仅自身obs+action → 输入不同。<br>
2. 设计意图：兼容独立训练（DDPG）与协作训练（MADDPG），牺牲效率换取灵活性。
### 用户代码2（标准MADDPG）
https://github.com/starry-sky6688/MADDPG/blob/master/maddpg/maddpg.py
1. Critic输入：强制全局obs+actions → 输入相同但Critic独立。
2. 设计意图：严格遵循CTDE范式，适合同构协作场景，扩展性较弱但结构清晰。

# 六、总结
1. 必要性：独立Critic是处理异构奖励、策略差异和环境非平稳性的核心设计，即使输入相同，各Critic仍需独立更新以捕捉不同策略的全局影响。
2. 效率权衡：通过参数共享试探和分布式训练可缓解计算开销，但在多数复杂场景中，独立Critic的收益远大于其成本。
3. 实践建议：优先采用独立Critic实现，待任务明确后针对性优化（如同类共享）。