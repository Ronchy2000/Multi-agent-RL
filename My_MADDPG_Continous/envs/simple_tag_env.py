# from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3

# from pettingzoo.mpe.simple_tag_v3 import raw_env

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

import pygame  #Ronchy: 用于渲染动画环境

'''
继承 raw_env, 修改部分功能。
'''

class Custom_raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_good, num_adversaries, num_obstacles)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_tag_v3"
        # Ronchy添加轨迹记录
        self.history_positions = {agent.name: [] for agent in world.agents}
        self.max_history_length = 50  # 最大轨迹长度



    """
    rewrite `_execute_world_step` method in:
        simple_env -> class SimpleEnv()
    """
    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        # Ronchy add: 
        def handle_collisions(): # 在 添加碰撞检查 (避免重叠，无弹开效果)
            for i, agent1 in enumerate(self.world.agents):
                for agent2 in self.world.agents[i+1:]:
                    if self.scenario.is_collision(agent1, agent2):
                        delta_pos = agent1.state.p_pos - agent2.state.p_pos
                        dist = np.sqrt(np.sum(np.square(delta_pos)))
                        dist_min = agent1.size + agent2.size

                        if dist < dist_min:
                            # 仅将重叠的智能体移动到刚好接触的位置
                            overlap = dist_min - dist
                            direction = delta_pos / (dist + 1e-8)  # 避免除零
                            move_dist = overlap / 2
                            agent1.state.p_pos += direction * move_dist
                            agent2.state.p_pos -= direction * move_dist
                # 检查与障碍物的碰撞
                for landmark in self.world.landmarks:
                    if landmark.collide:
                        delta_pos = agent1.state.p_pos - landmark.state.p_pos
                        dist = np.sqrt(np.sum(np.square(delta_pos)))
                        dist_min = agent1.size + landmark.size

                        if dist < dist_min:
                            overlap = dist_min - dist
                            direction = delta_pos / (dist + 1e-8)
                            # 只移动智能体，障碍物不动
                            agent1.state.p_pos += direction * overlap
        
        # Ronchy 多次迭代以处理复杂的碰撞情况
        for _ in range(3):  # 通常3次迭代足够处理大多数情况
            handle_collisions()
        #--------#################-----------------#
        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    """
    rewrite step method in: 
        simple_env -> class SimpleEnv()
    """ 
    def step(self, action): 
        # print("Using rewrited step method.")
        #  如果有任何智能体的 terminated 状态为 True，它们将从 self.env.agents 中移除
        if ( 
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()

            # Ronchy记录轨迹
            for agent in self.world.agents:
                self.history_positions[agent.name].append(agent.state.p_pos.copy())
                if len(self.history_positions[agent.name]) > self.max_history_length:
                    self.history_positions[agent.name].pop(0)
            
            self.steps += 1

            self.check_capture_condition(threshold=0.1)  #围捕标志——半径

            # 如果达到最大步数，标记 truncation 为 True
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
        
    def check_capture_condition(self,threshold = 0.1): # agent.size = 0.075 if agent.adversary else 0.05
        """
        检查所有围捕者是否都进入逃跑者的指定范围内。
        Args:
            threshold (float): 围捕者和逃跑者之间的最大允许距离。
        """
        agents = self.scenario.good_agents(self.world)  # 逃跑者
        adversaries = self.scenario.adversaries(self.world)  # 围捕者
        # 假设只有一个逃跑者，计算所有围捕者与该逃跑者的距离
        for agent in agents:  
            captured = all(  # Return True if all elements of the iterable are true.
                np.linalg.norm(agent.state.p_pos - adv.state.p_pos) < threshold
                for adv in adversaries
            )
            if captured:
                # 如果所有围捕者都在范围内，标记所有智能体为终止状态
                for a in self.agents:
                    self.terminations[a] = True
        

env = make_env(Custom_raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05  # 智能体的大小，判断是否碰撞的界定。
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):  # main_reward 也是一个数值，而不是元组
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        # print(f"main_reward{main_reward}")
        return main_reward

    # 逃跑者reward设置
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
                    rew -= 0  # 即，不学习逃跑策略。 default value = 10  

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # 设置环境边界。
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

    # 围捕者reward设置
    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True  #Ronchy 改为True，default: False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            '''
            'Cooperative control for multi-player pursuit-evasion games with reinforcement learning'中的奖励设置
            '''
            for adv in adversaries:
                for agent in agents:# 逃跑者目前只有一个
                    dist = np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
                    collision = 1 if self.is_collision(agent, adv) else 0  
                    rew += collision - dist
            for agent in agents:
                speed_agent = np.sqrt(np.sum(np.square(agent.state.p_vel)))
                rew -= speed_agent
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )

    # Ronchy: 重写render函数
    def draw(self):
       # 清空画布
       self.screen.fill((255, 255, 255))
       # 计算动态缩放
       all_poses = [entity.state.p_pos for entity in self.world.entities]
       cam_range = np.max(np.abs(np.array(all_poses)))
       scaling_factor = 0.9 * self.original_cam_range / cam_range
       # 绘制轨迹
       for agent in self.world.agents:
           if len(self.history_positions[agent.name]) >= 2:
               points = []
               for pos in self.history_positions[agent.name]:
                   x, y = pos
                   y *= -1
                   x = (x / cam_range) * self.width // 2 * 0.9
                   y = (y / cam_range) * self.height // 2 * 0.9
                   x += self.width // 2
                   y += self.height // 2
                   points.append((int(x), int(y)))
               
               # 绘制渐变轨迹
               for i in range(len(points) - 1):
                   alpha = int(255 * (i + 1) / len(points))
                   color = (0, 0, 255, alpha) if agent.adversary else (255, 0, 0, alpha)
                   line_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                   pygame.draw.line(line_surface, color, points[i], points[i + 1], 2)
                   self.screen.blit(line_surface, (0, 0))
       # 绘制实体
       for entity in self.world.entities:
           x, y = entity.state.p_pos
           y *= -1
           x = (x / cam_range) * self.width // 2 * 0.9
           y = (y / cam_range) * self.height // 2 * 0.9
           x += self.width // 2
           y += self.height // 2
           
           radius = entity.size * 350 * scaling_factor
           
           if isinstance(entity, Agent):
               color = (0, 0, 255) if entity.adversary else (255, 0, 0)
               pygame.draw.circle(self.screen, color, (int(x), int(y)), int(radius))
               pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), int(radius), 1)
           else:  # Landmark
               pygame.draw.circle(self.screen, (128, 128, 128), (int(x), int(y)), int(radius))
    
#=======================================================================
if __name__ =="__main__":
    print("Custom_raw_env",Custom_raw_env)