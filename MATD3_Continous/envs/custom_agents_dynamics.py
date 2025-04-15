"""
该文件定义了自定义的环境，用于测试自定义的智能体动力学模型

继承自core.py

"""
import numpy as np
from pettingzoo.mpe._mpe_utils.core import EntityState, AgentState, Action, Entity, Landmark, Agent
from pettingzoo.mpe._mpe_utils.core import World

class CustomWorld(World):
    def __init__(self, world_size = 2.5 ): #
        super().__init__() # 调用父类的构造函数
        self.world_size = world_size # Ronchy 添加世界大小
        self.dt = 0.1 # 时间步长
        self.damping = 0.2 # 阻尼系数
        # contact response parameters
        self.contact_force = 1e2 # 控制碰撞强度（默认1e2，值越大反弹越强）
        self.contact_margin = 1e-3 # 控制碰撞"柔软度"（默认1e-3，值越小越接近刚体）
        """
        常见问题示例
        实体重叠穿透	contact_force太小	增大contact_force至1e3或更高
        碰撞后震荡	damping太低	增大阻尼系数（如0.5）
        微小距离抖动	contact_margin不合理	调整到1e-2~1e-4之间
        """
    """ 
        重载底层动力学逻辑
        主要是integrate_state()函数
    """
    def step(self):
        # set actions for scripted agents
        # print("Using world -> step()") # 重载成功！
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
            # 速度阻尼衰减
            entity.state.p_vel *= (1 - self.damping)  # 正确应用阻尼
             # 动力学 -> 运动学
            if p_force[i] is not None:
                acceleration = p_force[i] / entity.mass # F = ma
                entity.state.p_vel += acceleration * self.dt # v = v_0 + a * t

            # 速度限幅
            if entity.max_speed is not None:
                speed = np.linalg.norm(entity.state.p_vel)  # 计算向量模长
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel * (entity.max_speed / speed)  # 向量缩放         
            
            # 更新位置
            entity.state.p_pos += entity.state.p_vel * self.dt  # 更新位置
            # 限制位置在世界大小范围内
            # entity.state.p_pos = np.clip(entity.state.p_pos, -self.world_size, self.world_size) # Ronchy 添加世界大小限制
             

    # get collision forces for any contact between two entities
    # TODO: 碰撞逻辑待细化
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos))) #用norm更简洁
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size  # 两个实体的半径之和
        # softmax penetration
        k = self.contact_margin 
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k  #渗透深度， 当 dist < dist_min 时产生虚拟渗透量
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
