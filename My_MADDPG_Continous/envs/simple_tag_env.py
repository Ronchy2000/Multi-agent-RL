# from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3

from pettingzoo.mpe.simple_tag_v3 import parallel_env
import numpy as np

class CustomEnv(parallel_env):
    def __init__(self, _detection_range = 10.0):
        super().__init__()
        self.detection_range = _detection_range

    def observation(self, agent, world):
        # 获取所有实体的位置（不包括边界地标）
        entity_pos = []
        for entity in world.landmarks:  # 即地标, 障碍物
            if not entity.boundary:
                if self.detection_range < 0:  #若参数为负数，没有探测范围限制。
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                else:  #若参数为正数，则有探测范围限制。
                    dist = np.linalg.norm(entity.state.p_pos - agent.state.p_pos)
                    if dist <= self.detection_range:
                        entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                    else:
                        entity_pos.append(0)  

        # 获取其他所有代理的位置
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        # 返回一个只包含位置的数据（注意我们可以只保留位置相关的信息）
        return np.concatenate([agent.state.p_pos] + entity_pos + other_pos)
    

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        print("Using rewrite method_reward.\n")
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
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew