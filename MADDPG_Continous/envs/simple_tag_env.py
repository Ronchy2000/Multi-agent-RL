# from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3

# from pettingzoo.mpe.simple_tag_v3 import raw_env

import numpy as np
import gymnasium
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .custom_agents_dynamics import CustomWorld # Ronchy自定义的World类，用于测试自定义的智能体动力学模型

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
        max_cycles=50,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        world_size = 2.5, # Ronchy添加 world_size参数 ,地图大小 world_size x world_size
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
        world = scenario.make_world(num_good, num_adversaries, num_obstacles, _world_size = world_size) # Ronchy添加 world_size参数
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.world_size = world_size  # Ronchy添加 world_size参数, 地图大小 world_size x world_size
        self.metadata["name"] = "simple_tag_v3"
        # Ronchy添加轨迹记录
        self.history_positions = {agent.name: [] for agent in world.agents}
        # self.max_history_length = 500  # 最大轨迹长度

        # 重载 simple_env.py中的代码
        pygame.font.init()
        self.game_font = pygame.font.SysFont('arial', 16)  # 使用系统字体

        self.max_force = 1.0  # 最大力
        self.capture_threshold = self.world_size * 0.2 # 围捕阈值: 使用世界大小的20%作为默认捕获范围
        # 重载continuous_actions空间
        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:  #每个智能体都有自己的观测空间和动作空间
            if agent.movable: 
                if self.continuous_actions == True:
                    space_dim = self.world.dim_p  # dim_p: default 2  -> position dimensionality  
                elif self.continuous_actions == False:
                    space_dim = self.world.dim_p * 2 + 1  # default: 5  # 1个维度表示静止，4个维度表示4个方向的运动，离散值
            else:
                space_dim = 1 # 1个维度表示静止
            # 通信动作
            if agent.silent == False:  #Scenario类中默认为True
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c
            obs_dim = len(self.scenario.observation(agent, self.world))  # 观测维度， Scenario类中 observation()创建观测空间
            state_dim += obs_dim  # 所有智能体的观测空间累积，就是 状态空间维数

            if self.continuous_actions: # continuous actions
                self.action_spaces[agent.name] = gymnasium.spaces.Box(
                    low= -1.0, high=1.0, shape=(space_dim,), dtype=np.float32 # 限制在[-1,1]之间，这个是控制输入的限幅
                )
            else:  # discrete actions
                self.action_spaces[agent.name] = gymnasium.spaces.Discrete(space_dim)
            # 定义单个agent的观测空间
            self.observation_spaces[agent.name] = gymnasium.spaces.Box(
                low = -np.float32(np.inf), # 最低限制
                high = +np.float32(np.inf), # 最高限制
                shape = (obs_dim,),
                dtype = np.float32,
            )
        # 定义多智能体状态空间 公用1个状态空间。
        self.state_space = gymnasium.spaces.Box(
            low = -np.float32(np.inf),
            high = +np.float32(np.inf),
            shape = (state_dim,),
            dtype = np.float32,
        )


    def reset(self, seed=None, options=None):
        # 重置环境状态并清空轨迹记录
        super().reset(seed=seed, options=options)
        # 清空轨迹
        self.history_positions = {agent.name: [] for agent in self.world.agents}

    def reset_world(self, world, np_random):
        # 清除历史轨迹
        self.history_positions = {agent.name: [] for agent in self.world.agents}  
        # 调用Scenario的reset_world方法
        super().scenario.reset_world(world, np_random)

    """
    rewrite `_execute_world_step` method in:
        simple_env <- class SimpleEnv()
    """
    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            mdim = self.world.dim_p if self.continuous_actions else self.world.dim_p * 2 + 1  # 连续 2，离散 5
            # print(f"_execute_world_step : mdim:{mdim}") # mdim: 2
            if agent.movable: # default: True
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])  # phisical action  mdim: 2  ,此处的Scenario_action是二维列表了.[action[0:2], acrionp[2:]],[[物理动作]，[通信动作]]
                    action = action[mdim:] # communication action
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent: # default: True
                scenario_action.append(action)
            # set action for agent  action_spaces[agent.name]已经被划分成 scenario_action和action了，所以此处action_spaces[agent.name]不再使用
            self._set_action(scenario_action, agent, self.action_spaces[agent.name], time = None)

        self.world.step() #core.py  在world实例中 执行动力学

        # Ronchy add: 
        # def handle_collisions(): # 在 添加碰撞检查 (避免重叠，无弹开效果)
        #     for i, agent1 in enumerate(self.world.agents):
        #         for agent2 in self.world.agents[i+1:]:
        #             if self.scenario.is_collision(agent1, agent2):
        #                 delta_pos = agent1.state.p_pos - agent2.state.p_pos
        #                 dist = np.sqrt(np.sum(np.square(delta_pos)))
        #                 dist_min = agent1.size + agent2.size

        #                 if dist < dist_min:
        #                     # 仅将重叠的智能体移动到刚好接触的位置
        #                     overlap = dist_min - dist
        #                     direction = delta_pos / (dist + 1e-8)  # 避免除零
        #                     move_dist = overlap / 2
        #                     agent1.state.p_pos += direction * move_dist
        #                     agent2.state.p_pos -= direction * move_dist
        #         # 检查与障碍物的碰撞
        #         for landmark in self.world.landmarks:
        #             if landmark.collide:
        #                 delta_pos = agent1.state.p_pos - landmark.state.p_pos
        #                 dist = np.sqrt(np.sum(np.square(delta_pos)))
        #                 dist_min = agent1.size + landmark.size

        #                 if dist < dist_min:
        #                     overlap = dist_min - dist
        #                     direction = delta_pos / (dist + 1e-8)
        #                     # 只移动智能体，障碍物不动
        #                     agent1.state.p_pos += direction * overlap
        
        # # Ronchy 多次迭代以处理复杂的碰撞情况
        # for _ in range(3):  # 通常3次迭代足够处理大多数情况
        #     handle_collisions()
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

    # set env action for a particular agent
    # self._set_action(scenario_action, agent, self.action_spaces[agent.name])
    # scenario_action 物理动作实参
    def _set_action(self, action, agent, action_space, time=None):
        """
        pettiongzoo中的 agent的动作 被分为 action.u 和 action.c 两部分,
        分别代表physical action和communication action。
        默认值：
        action维数为5, 第0位没有用
        第1,2位表示x方向加速和减速
        第3,4位表示y方向加速和减速
        """
        #此处是指agent.action = agent.action.u -> scenarios_action + agent.action.c -> communication_action
        agent.action.u = np.zeros(self.world.dim_p) # default:2  phisiacal action, 加速度的维数
        agent.action.c = np.zeros(self.world.dim_c) # default:2  communication action的维数
        if agent.movable:
            agent.action.u = np.zeros(self.world.dim_p) #  default:2
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                # Note: this ordering preserves the same movement direction as in the discrete case
                # print("_set_action: action",action)
                agent.action.u[0] = action[0][0] # Force in x direction
                agent.action.u[1] = action[0][1]  # Force in y direction
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
        
        # Ronchy 添加力限幅
        #self.max_force = 1.0  # 根据需求调整
        agent.action.u = np.clip(agent.action.u, -self.max_force, self.max_force)

        #     # Ronchy 修改加速度逻辑
        #     sensitivity = 1.0  # default: 5.0
        #     if agent.accel is not None:
        #         sensitivity = agent.accel
        #     agent.action.u *= sensitivity
        #     action = action[1:]

        # if not agent.silent:  # 默认为True，这里被跳过。
        #     # communication action
        #     if self.continuous_actions:
        #         agent.action.c = action[0]
        #     else:
        #         agent.action.c = np.zeros(self.world.dim_c)
        #         agent.action.c[action[0]] = 1.0
        #     action = action[1:]
        # make sure we used all elements of action
        # assert len(action) == 0 # Ronchy: 保证action被完全使用。如果 action 数组不为空，说明有动作维度没有被正确处理，程序会抛出 AssertionError

    """
    rewrite step method in: 
        simple_env <- class SimpleEnv()

        simple_tag_env.step(action)   # 在算法上直接调用env.step(action)即可
            -> _execute_world_step()  
            -> _set_action(action) # 把合外力变成加速度。（原版是 乘以 sensitivity; 即agent.accel）
            -> world.step()  # 调用 core.py 中的 World.step()  # 实现agent 的动力学
        -simple_tag_env.step() ：
            - 环境层面的步进
            - 处理动作的接收和预处理
            - 管理奖励、状态转换等高层逻辑
            - 处理轨迹记录、终止条件等
        - core.py 中的 World.step() ：  # 需要在Scenario类中重载
            - 物理引擎层面的步进
            - 实现具体的动力学计算
            - 处理力的应用和状态积分
            - 更新物理状态（位置、速度等）
    """ 
    def step(self, action):   # 环境层面的步进
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
                # if len(self.history_positions[agent.name]) > self.max_history_length: # 限制轨迹长度
                #     self.history_positions[agent.name].pop(0)

            self.steps += 1

            self.check_capture_condition(threshold=self.capture_threshold)  #围捕标志——半径

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
    """
    rewrite step method in: 
        simple_env <- class SimpleEnv()
    """ 
    # Ronchy: 重载render函数
    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True
    
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        # 添加事件处理, 解决windows渲染报错
        """
        Mac 上不需要特别处理事件是因为 macOS 的窗口管理系统和事件处理机制与 Windows 不同。Mac 系统有更好的窗口管理机制，即使不处理事件队列也不会导致程序无响应。
         这样的设计可以：
                1. 在 Windows 上避免无响应
                2. 在 Mac 上也能正常工作
                3. 提供更好的用户体验（比如正确响应窗口关闭按钮）
                所以建议保留这段事件处理代码，这样你的程序在任何平台上都能正常工作。
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.WINDOWCLOSE:
                pygame.quit()
                return
        pygame.event.pump()  # 确保事件系统正常运行
        #--------
        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return
    
    def draw(self):
        # 清空画布
        self.screen.fill((255, 255, 255))
        # 计算动态缩放
        all_poses = [entity.state.p_pos for entity in self.world.entities]
    #     cam_range = np.max(np.abs(np.array(all_poses)))
        cam_range = self.world_size  # 使用环境实际大小
        scaling_factor = 0.7 * self.original_cam_range / cam_range
        # 绘制坐标轴
        self.draw_grid_and_axes()
        # 在逃跑者位置绘制capture_threshold 圆圈
        for agent in self.scenario.good_agents(self.world):
            x, y = agent.state.p_pos
            y *= -1
            # 使用与实体相同的坐标转换逻辑
            x = (x / cam_range) * self.width // 2 * 0.9
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            
            # 创建透明surface来绘制捕获圈
            circle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            threshold_pixels = int(self.capture_threshold / cam_range * self.width // 2 * 0.9)  # 与实体渲染使用相同的缩放逻辑
            pygame.draw.circle(circle_surface, (0, 200, 0, 50), (int(x), int(y)), threshold_pixels, 2)  # 最后一个参数2是线宽
            self.screen.blit(circle_surface, (0, 0))
    
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
                    pygame.draw.line(line_surface, color, points[i], points[i + 1], 4)  # 最后一位是线宽
                    self.screen.blit(line_surface, (0, 0))
        # 绘制实体
        for entity in self.world.entities:
            x, y = entity.state.p_pos
            y *= -1
            x = (x / cam_range) * self.width // 2 * 0.9
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2

            # radius = entity.size * 140 * scaling_factor
            # # 修改为：根据世界到屏幕的实际转换比例计算
            world_to_screen_scale = (self.width / (2 * self.world_size)) * 0.9
            radius = entity.size * world_to_screen_scale

            if isinstance(entity, Agent):
             # 设置透明度：例如，transparent_alpha=128 (半透明)
                transparent_alpha = 200  # 透明度，范围从0（完全透明）到255（完全不透明）
                color = (0, 0, 255, transparent_alpha) if entity.adversary else (255, 0, 0, transparent_alpha)
                # 创建透明度支持的Surface
                agent_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

                # pygame.draw.circle(self.screen, color, (int(x), int(y)), int(radius))
                pygame.draw.circle(agent_surface, color, (int(x), int(y)), int(radius))

                agent_surface.set_alpha(transparent_alpha)  # 设置透明度
                self.screen.blit(agent_surface, (0, 0))

                # pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), int(radius), 1) # 绘制边框
                pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), int(radius), 1) # 绘制边框
            else:  # Landmark
                pygame.draw.circle(self.screen, (128, 128, 128), (int(x), int(y)), int(radius))
        pygame.display.flip()
    
    """绘制坐标轴"""
    def draw_grid_and_axes(self):
        cam_range = self.world_size  # 使用环境实际大小
        # 计算屏幕边界位置
        margin = 40  # 边距
        plot_width = self.width - 2 * margin
        plot_height = self.height - 2 * margin
     
        # 绘制边框
        pygame.draw.rect(self.screen, (0, 0, 0), 
                        (margin, margin, plot_width, plot_height), 1)
     
        # 绘制网格线
        grid_size = 0.5  # 网格间隔
        for x in np.arange(-self.world_size, self.world_size + grid_size, grid_size):
            screen_x = int((x + self.world_size) / (2 * self.world_size) * plot_width + margin)
            pygame.draw.line(self.screen, (220, 220, 220),
                            (screen_x, margin),
                            (screen_x, margin + plot_height), 1)
            # 绘制刻度
            if abs(x) % 1.0 < 0.01:  # 整数位置
                pygame.draw.line(self.screen, (0, 0, 0),
                               (screen_x, margin + plot_height),
                               (screen_x, margin + plot_height + 5), 1)
                text = self.game_font.render(f"{x:.0f}", True, (0, 0, 0))
                self.screen.blit(text, (screen_x - 5, margin + plot_height + 10))
     
        for y in np.arange(-self.world_size, self.world_size + grid_size, grid_size):
            screen_y = int((-y + self.world_size) / (2 * self.world_size) * plot_height + margin)
            pygame.draw.line(self.screen, (220, 220, 220),
                            (margin, screen_y),
                            (margin + plot_width, screen_y), 1)
            # 绘制刻度
            if abs(y) % 1.0 < 0.01:  # 整数位置
                pygame.draw.line(self.screen, (0, 0, 0),
                               (margin - 5, screen_y),
                               (margin, screen_y), 1)
                text = self.game_font.render(f"{y:.0f}", True, (0, 0, 0))
                text_rect = text.get_rect()
                self.screen.blit(text, (margin - 25, screen_y - 8))

    def check_capture_condition(self,threshold = None): # agent.size = 0.075 if agent.adversary else 0.05
        """
        检查所有围捕者是否都进入逃跑者的指定范围内。
        Args:
            threshold (float): 围捕者和逃跑者之间的最大允许距离。
        """
        if threshold is None:
            threshold = self.world_size * 0.2 # 使用世界大小的20%作为默认捕获范围
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
    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2, _world_size=2.5):
        # world = World() # core.py 中的World类
        world = CustomWorld() # Ronchy: 使用自定义的World类,重载了动力学逻辑
        # set any world properties first
        world.world_size =  _world_size # Ronchy添加世界大小
        world.dim_c = 0  # Ronchy set 0, communication channel dimensionality,default 2
        world.dim_p = 2  # position dimensionality, default 2
        """
        time_step = 0.1  这个是在core.py中的World类中定义的,名称为 dt = 0.1
        agent的运动都在core.py中的World类中的step()方法中进行
        """
        world.dt = 0.1 # time_step, default 0.1
        world.damping = 0.2  # 阻尼系数 0.25是默认值

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
            base_size = _world_size * 0.1 # 基础大小为世界大小的10%
            # agent.size = 0.25 if agent.adversary else 0.15  # 智能体的半径，判断是否碰撞的界定
            agent.size = base_size if agent.adversary else base_size*0.6  # 智能体的半径，判断是否碰撞的界定
            agent.initial_mass = 1.6 if agent.adversary else 0.8  # 智能体的质量 kg
            agent.accel = None # 不使用该参数
            # agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.max_speed = 1.0 if agent.adversary else 1.0
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
            # agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p) # default: 2
            agent.state.p_pos = np_random.uniform(-world.world_size * 0.9, +world.world_size * 0.9, world.dim_p) #  # 留出10%边界
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                # landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p) # default: 1.8
                landmark.state.p_pos = np_random.uniform(-world.world_size * 0.8, +world.world_size * 0.8, world.dim_p) # 留出20%边界
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
        # 设置环境边界, 我在world中限制了边界，pos无法超出，该奖励需要测试，看是否还需要
        # def bound(x):
        #     if x < 2.4:
        #         return 0
        #     if x < 2.5:
        #         return (x - 2.4) * 10
        #     return min(np.exp(2 * x - 5), 10)

        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= bound(x)

        # return rew
        # 修改为动态边界（假设边界为 world.world_size 的 96% 开始衰减）
        # 边界惩罚的新定义
        def bound(x):
            boundary_start = world.world_size * 0.96 
            full_boundary = world.world_size
            if x < boundary_start:
                return 0
            if x < full_boundary:
                return (x - boundary_start) * 10
            return min(np.exp(2 * x - 2 * full_boundary), 10)
        
        # ==== 必须添加实际边界计算 ====
        for p in range(world.dim_p):  # 遍历每个坐标轴 (x, y)
            x = abs(agent.state.p_pos[p])  # 获取坐标绝对值
            rew -= bound(x)  # 应用边界惩罚函数
            
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
                    rew += 10 * collision - dist / world.world_size  # 增大碰撞，归一化距离计算奖励
            for agent in agents:  # 逃跑者
                speed_agent = np.sqrt(np.sum(np.square(agent.state.p_vel)))
                rew -= 0.5 * speed_agent / agent.max_speed  # 逃跑者速度惩罚归一化

        def bound(x):
            boundary_start = world.world_size * 0.96 
            full_boundary = world.world_size
            if x < boundary_start:
                return 0
            if x < full_boundary:
                return (x - boundary_start) * 10
            return min(np.exp(2 * x - 2 * full_boundary), 10)
        
        # ==== 必须添加实际边界计算 ====
        for p in range(world.dim_p):  # 遍历每个坐标轴 (x, y)
            x = abs(agent.state.p_pos[p])  # 获取坐标绝对值
            rew -= bound(x)  # 应用边界惩罚函数
            
        return rew

    def observation(self, agent, world):  # 返回值，自动适配智能体的观测空间维数
        """
            智能体及地标的观测空间
            TODO:需要按需重载。
        """
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:  # default: False ,要执行的代码块
                # 归一化相对位置
                relative_entity_pos = (entity.state.p_pos - agent.state.p_pos) / world.world_size
                entity_pos.append(relative_entity_pos)  #表示地标相对于智能体的位置。这样返回的每个地标位置将是一个 2D 向量，表示该地标与智能体之间的相对位置。
        # communication of all other agents
        comm = [] # default: self.c = None
        other_pos = [] # 其他智能体（包扩逃跑者）相对于智能体的位置——相对位置
        other_vel = [] # 逃跑者的速度
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c / world.world_size)  # default: self.c = None
            # 归一化相对位置
            relative_pos = (other.state.p_pos - agent.state.p_pos) / world.world_size
            other_pos.append(relative_pos)
            if not other.adversary:  # adversary，追逐者的值是True，逃跑者的值是False
                # 归一化速度
                norm_vel = other.state.p_vel / other.max_speed
                other_vel.append(norm_vel)
        # 自身状态归一化
        norm_self_vel = agent.state.p_vel / world.world_size
        norm_self_pos = agent.state.p_pos / world.world_size
        return np.concatenate(
            [norm_self_vel]
            + [norm_self_pos]
            + entity_pos
            + other_pos
            + other_vel
        )
    
    
    
#=======================================================================
if __name__ =="__main__":
    print("Custom_raw_env",Custom_raw_env)
    # 创建测试环境
    # env = make_env(Custom_raw_env)
    # parallel_env = parallel_wrapper_fn(env) # 启用并行环境
    # parallel_env.reset()

    num_good = 1
    num_adversaries = 3
    num_obstacles = 0

        # 创建并初始化环境
    env = Custom_raw_env(
        num_good=num_good, 
        num_adversaries=num_adversaries, 
        num_obstacles=num_obstacles, 
        continuous_actions=True,  # 设置为 True 使用连续动作空间
        render_mode="None"
    )

    env.reset()

    # 打印环境和智能体信息
    print("环境初始化完成。")
    print(f"环境名称: {env.metadata['name']}")
    print(f"智能体数量: {len(env.agents)}")

   # 遍历每个智能体并打印其初始状态
    for agent_name in env.agents:
        # 获取当前智能体的观察空间和动作空间
        obs_space = env.observation_space(agent_name)
        action_space = env.action_space(agent_name)

        # 获取当前智能体的观测
        observation = env.observe(agent_name)

        # 获取当前智能体的动作空间范围（低和高值）
        action_low = action_space.low
        action_high = action_space.high

        # 打印信息
        print(f"\n==== {agent_name} ====")
        print(f"观测空间维度: {obs_space.shape}")
        print(f"动作空间维度: {action_space.shape}")
        print(f"动作空间的低值: {action_low}")
        print(f"动作空间的高值: {action_high}")

        # 打印智能体的初始观测
        print(f"初始观测: {observation}")
        
        # 如果你想测试环境的一个动作，可以给智能体一个随机动作，并打印
        random_action = action_space.sample()  # 从动作空间中采样一个随机动作
        print(f"随机选择的动作: {random_action}")