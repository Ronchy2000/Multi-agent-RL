import time
from typing import Optional, Union, List, Tuple

import gymnasium as gym
# import gym
import numpy as np
from gymnasium import spaces
# from gym import spaces
from gymnasium.core import RenderFrame, ActType, ObsType
# from gym.core import RenderFrame, ActType, ObsType

#随机数生成器将产生相同的随机数序列, 这在需要可重复结果的情况下非常有用
np.random.seed(1)
import render


def arr_in_list(array, _list):
    for element in _list:
        if np.array_equal(element, array):
            return True
    return False


class GridEnv(gym.Env):

    def __init__(self, size: int, target: Union[list, tuple, np.ndarray], forbidden: Union[list, tuple, np.ndarray],
                 render_mode: str):
        """
        GridEnv 的构造函数
        :param size: grid_world 的边长
        :param target: 目标点的pos
        :param forbidden: 不可通行区域 二维数组 或者嵌套列表 如 [[1,2],[2,2]]
        :param render_mode: 渲染模式 video表示保存视频
        """
        # 初始化可视化
        self.agent_location = np.array([0, 0])
        self.time_steps = 0
        self.size = size
        self.render_mode = render_mode
        self.render_ = render.Render(target=target, forbidden=forbidden, size=size)
        # 初始化起点 障碍物 目标点
        self.forbidden_location = []
        for fob in forbidden:
            self.forbidden_location.append(np.array(fob))
        self.target_location = np.array(target)
        # 初始化 动作空间 观测空间
        self.action_space, self.action_space_size = spaces.Discrete(5,seed = 42), spaces.Discrete(5).n  #seed = 42, “42 是 “生命、宇宙和一切终极问题的答案”
        print("self.action_space:{}, self.action_space_size:{}".format(self.action_space, self.action_space_size))  #从0开始索引

        # self.reward_list = [0, 1, -10, -10]
        # self.reward_list = [0, 1, -1, -10]
        self.reward_list = [-1, 0, -1, -10]  #forbidden area:-10 ;  撞墙:-10
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low = 0, high = size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(low = 0, high = size - 1, shape=(2,), dtype=int),
                "barrier": spaces.Box(low = 0, high = size - 1, shape=(2,), dtype=int),
            }
        )
        # action to pos偏移量 的一个map
        #坐标系  ------>    x > 0
        #       |
        #       | y>0
        #       v

        self.action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }
        # Rsa表示 在 指定 state 选取指定 action 得到Immediate reward的概率
        self.Rsa = None
        # Psa表示 在 指定 state 选取指定 action 跳到下一个state的概率
        self.Psa = None
        self.psa_rsa_init()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.agent_location = np.array([0, 0])
        observation = self.get_obs()
        info = self.get_info()
        return observation, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:  #  -> 是函数的返回类型注解，

        reward = self.reward_list[self.Rsa[self.pos2state(self.agent_location), action].tolist().index(1)]
        direction = self.action_to_direction[action]
        self.render_.upgrade_agent(self.agent_location, direction, self.agent_location + direction)
        self.agent_location = np.clip(self.agent_location + direction, 0, self.size - 1)
        terminated = np.array_equal(self.agent_location, self.target_location)
        observation = self.get_obs()
        info = self.get_info()
        return observation, reward, terminated, False, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.render_mode == "video":
            self.render_.save_video('image/' + str(time.time()))

        self.render_.show_frame(0.3)
        return None
    def render_clear(self):
        self.render_.close_frame()
        return None

    def plot_title(self,title = "title"):
        self.render_.plot_title(title)

    def get_obs(self) -> ObsType:
        return {"agent": self.agent_location, "target": self.target_location, "barrier": self.forbidden_location}

    def get_info(self) -> dict:
        return {"time_steps": self.time_steps}


    def state2pos(self, state: int) -> np.ndarray:
        """
        用于将状态（state）转换为位置（pos）。这在一些环境中是很常见的.
        比如在一个二维的格子世界中，我们可能会将每个格子看作一个状态，然后用一个整数来表示这个状态。而这个函数就是用来将这个整数转换回对应的格子位置。
        :param state: state number
        :return: 二维列表，表示agent的位置：x行x列
        """
        return np.array((state // self.size, state % self.size))

    def pos2state(self, pos: np.ndarray) -> int:
        """
        假设 self.size 是 4，那么位置 (1, 1) 对应的状态就是 1 * 4 + 1 = 5
        :param pos:  pos[0] 和 pos[1] 分别表示位置的行和列。
        :return: state number
        """
        return pos[0] * self.size + pos[1]

    def psa_rsa_init(self):
        """
        初始化网格世界的 psa 和 rsa
        赵老师在b站评论区回答过 关于 rsa设计的问题
        原问题是；
        B友：老师您好，在spinning up 7.2.5里有写到
        Reward depends on the current state of the world, the action just taken, and the next state of the world.
        但您提到Rewad depends on the state and action, but not the next state.不知道reward 和 next state的关系是怎样的？

        答案如下：
        赵老师：这是一个很细小、但是很好的问题，说明你思考了。也许其他人也会有这样的疑问，我来详细解答一下。
        1）从贝尔曼公式和数学的角度来说，r是由p(r|s,a)决定的，所以从数学的角度r依赖于s,a，而不依赖于下一个状态s’。这是很简明的。
        2）举例，如果在target state刚好旁边是墙，agent试图撞墙又弹回来target state，这时候不应该给正r，而应该是给负r，因为r依赖于a而不是下一个状态。
        3）但是r是否和s’无关呢？实际是有关系的，否则为什么每次进到target state要得到正r呢？不过，这也可以等价理解成是在之前那个状态采取了好的动作才得到了正r。
        总结：r确实和s’有关，但是这种关系被设计蕴含到了条件概率p(r|s,a)中去。
        故而这里的rsa蕴含了next_state的信息
        :return:
        """
        """
        Psa 和 Rsa 是三维的，因为它们需要表示在每个状态下，对于每个可能的动作，都有一个概率分布或奖励值。
        对于 Psa，第一维表示当前状态，第二维表示当前动作，第三维表示下一个状态。Psa[state, action, next_state] 表示在 state 状态下，执行 action 动作后，转移到 next_state 的概率。
        对于 Rsa，第一维表示当前状态，第二维表示当前动作，第三维表示奖励值。Rsa[state, action, reward] 表示在 state 状态下，执行 action 动作后，得到 reward 奖励的概率。
        这样的三维结构可以很好地表示状态、动作和奖励之间的关系，是强化学习中常用的数据结构。
        """
        """
        Psa 和 Rsa 中的概率体现在它们的值上。具体来说：
        对于 Psa，Psa[state, action, next_state] 的值表示在 state 状态下，执行 action 动作后，转移到 next_state 的概率。例如，如果 Psa[1, 2, 3] = 0.5，那么这就表示在状态 1 下执行动作 2 后，转移到状态 3 的概率是 0.5。
        对于 Rsa，Rsa[state, action, reward] 的值表示在 state 状态下，执行 action 动作后，得到 reward 奖励的概率。例如，如果 Rsa[1, 2, 3] = 0.5，那么这就表示在状态 1 下执行动作 2 后，得到奖励 3 的概率是 0.5。
        这两个矩阵的每个元素都是一个概率值，这就是概率在数据结构中的体现。
        """

        state_size = self.size ** 2
        self.Psa = np.zeros(shape=(state_size, self.action_space_size, state_size), dtype=float)
        self.Rsa = np.zeros(shape=(self.size ** 2, self.action_space_size, len(self.reward_list)), dtype=float)
        # 填充Psa、Rsa矩阵
        for state_index in range(state_size):
            for action_index in range(self.action_space_size):
                pos = self.state2pos(state_index) # 二维列表
                next_pos = pos + self.action_to_direction[action_index] # action_index：0~4，将动作映射给pos，得到next_pos

                if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] > self.size - 1 or next_pos[1] > self.size - 1:  #如果“撞墙”了，超出地图边界
                    self.Psa[state_index, action_index, state_index] = 1
                    self.Rsa[state_index, action_index, 3] = 1
                else:
                    self.Psa[state_index, action_index, self.pos2state(next_pos)] = 1
                    if np.array_equal(next_pos, self.target_location): #如果到达target area
                        self.Rsa[state_index, action_index, 1] = 1
                    elif arr_in_list(next_pos, self.forbidden_location): #如果进入forbidden_area
                        self.Rsa[state_index, action_index, 2] = 1 #
                    else:
                        self.Rsa[state_index, action_index, 0] = 1
        #print("self.Psa:{}\n self.Rsa:{}".format(self.Psa,self.Rsa))
    def close(self):
        pass


if __name__ == "__main__":
    # grid = GridEnv(size=5, target=[1, 3], forbidden=[[2, 2],[2,0],[4,2],[3,4]], render_mode='')
    grid = GridEnv(size=2, target=[1, 1], forbidden=[], render_mode='')
    grid.render()
