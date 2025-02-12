"""
该文件定义了自定义的环境，用于测试自定义的智能体动力学模型

继承自core.py

"""
import numpy as np
from pettingzoo.mpe._mpe_utils.core import EntityState, AgentState, Action, Entity, Landmark, Agent
from pettingzoo.mpe._mpe_utils.core import World

class CustomWorld(World):
    def __init__(self):
        super().__init__() # 调用父类的构造函数
        self.dt = 0.1 # 时间步长
        self.damping = 0.2 # 阻尼系数
    
    """ 
        重载底层动力学逻辑
        主要是integrate_state()函数
    """
    def step(self):
        # set actions for scripted agents
        print("Using world -> step()")
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force) # 加入噪声
        # apply environment forces
        p_force = self.apply_environment_force(p_force) # 碰撞力计算 collide为True时
        # integrate physical state
        self.integrate_state(p_force) # 动力学逻辑
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent) # 更新 communication action 后的状态
        
    # integrate physical state
    #函数功能：动力学逻辑。更新实体的位置和速度
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_pos += entity.state.p_vel * self.dt  # 更新位置
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping) # 更新速度
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(
                            np.square(entity.state.p_vel[0])
                            + np.square(entity.state.p_vel[1])
                        )
                        * entity.max_speed
                    )


    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
