# -*- coding: utf-8 -*-
#
# @Time : 2024-10-30 16:04:46
# @Author : Ronchy Lu
# @Email : rongqi1949@gmail.com
# @File : MADDPG.py
# @Software: PyCharm
# @Description: None
from utils import *
from NN_Module import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DDPG:
    """The DDPG Algorithm.

    1. Each instance of DDPG corresponds an agent.
    2. Each instance of DDPG consists of an actor (policy network) and a critic (value network).
    3. Each instance of DDPG contains a set of target networks for its actor and critic (affected by the double DQN strategy).
    4. Network update function is contained in the Center Controller class, the MADDPG class, so that we can achieve the Centralized Training and Decentralized Execution method easily.

    Attributes:
        actor:              The Actor (Policy Network).
        target_actor:       The Target Actor (Target Policy Network).
        critic:             The Critic (value Network).
        target_critic:      The Target Critic (Target Value Network).
        actor_optimizer:    The optimizer of Actor.
        critic_optimizer:   The optimizer of Critic.
    """

    def __init__(self, state_dim, action_dim, critic_dim, hidden_dim, actor_lr, critic_lr, device):
        """Initialize a DDPG instance for an agent.

        Args:
            state_dim:      The dimension of the state, which is also the input dimension of Actor and a part of Critics' input.
            action_dim:     The dimension of the action, which is also a part of Critics' input.
            critic_dim:     The dimension of the Critics' input (Critic Dimension = State Dimensions + Action Dimension).
            hidden_dim:     The dimension of the hidden layer of the networks.
            actor_lr:       The learning rate for the Actor.
            critic_lr:      The learning rate for the Critic.
            device:         The device to compute.
        """
        # Set Actor with Target Network
        self.actor = SimpleNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = SimpleNet(state_dim, action_dim, hidden_dim).to(device)
        # Set Critic with Target Network
        self.critic = SimpleNet(critic_dim, 1, hidden_dim).to(device)
        self.target_critic = SimpleNet(critic_dim, 1, hidden_dim).to(device)
        # Load parameters from Actor and Critic to their target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, explore=False):
        """Take action from the Actor (Policy Network).

        1. State -> the Actor -> Action Value
        2. Choose action according to the value of explore.
            - explore == True: Choose an action based on the gumbel trick.
            - explore == False: Choose the action from the Actor, and transform the value to a one-hot vector.

        Args:
            state:      The partial observation of the agent.
            explore:    The strategy to choose action.

        Returns: The action that the Actor has chosen.

        """
        # Choose an action from actor network (deterministic policy network).
        action = self.actor(state)
        # Exploration and Exploitation
        if explore:
            action = gumbel_softmax(action)
        else:
            action = trans2onehot(action)
        # TODO: Find out why the action need to be transferred to the CPU
        # return action.detach().cpu().numpy()[0]
        return action.detach().cpu().numpy()[0]

    @staticmethod
    def soft_update(net, target_net, tau):
        """Soft update function, which is used to update the parameters in the target network.

        Args:
            net:            The original network.
            target_net:     The target network.
            tau:            Soft update parameter.

        Returns: None

        """
        # Update target network's parameters using soft update strategy
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

#===================================================================================================
#===================================================================================================
class MADDPG:
    """The Multi-Agent DDPG Algorithm.

    1. The instance of MADDPG is the Center Controller in the algorithm.
    2. The instance of MADDPG contains a list of DDPG instances, which are corresponded with the agents in the environment one by one.

    Attributes:
        agents:     A list of DDPG instances, which are corresponded with the agents in the environment one by one.
        device:     The device to compute.
        gamma:      The gamma parameter in TD target.
        tau:        The tau parameter for soft update.
        critic_criterion: The loss function for the Critic networks.
    """

    def __init__(self, state_dims, action_dims, critic_dim, hidden_dim, actor_lr, critic_lr, device, gamma, tau):
        """Initialize a MADDPG instance as the Center Controller.

        Args:
            state_dims: A list of dimensions of each agent's observation.
            action_dims: A list of dimensions of each agent's action.
            critic_dim: The dimension of the Critic networks' input.
            hidden_dim: The dimension of the networks' hidden layers.
            actor_lr: The learning rate for the Actor.
            critic_lr: The learning rate for the Critic.
            device: The device to compute.
            gamma: The gamma parameter in TD target.
            tau: The tau parameter for soft update.
        """
        # TODO: Should we use dict to combine the DDPG instance with agents?
        self.agents = [
            DDPG(state_dim, action_dim, critic_dim, hidden_dim, actor_lr, critic_lr, device)
            for state_dim, action_dim in zip(state_dims, action_dims)
        ]
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = nn.MSELoss()

    @property
    def policies(self):
        """A list of Actors for the agents."""
        return [agent.actor for agent in self.agents]

    @property
    def target_policies(self):
        """A list of Target Actors for the agents."""
        return [agent.target_actor for agent in self.agents]

    def take_action(self, states, explore):
        """Take actions from the Actors (Policy Networks).

        Args:
            states: A list of observations from all the agents.
            explore: The strategy to choose action (Exploration or Exploitation).

        Returns: A list of actions in one-hot vector form.

        """
        states = [torch.tensor(np.array([state]), dtype=torch.float, device=self.device) for state in states]
        return [agent.take_action(state, explore) for agent, state in zip(self.agents, states)]

    def update(self, sample, agent_idx):
        """Update parameters for the agent whose index is agent_idx.

        Args:
            sample: A batch of transitions used to update all the parameters.
            agent_idx: The index of the agent whose Actor and Critic would be updated in this function.

        Returns: None

        Process:
            Parse Data from Sample to Observation, Action, Reward, Next Observation and Done Tag.
            Set up Current Agent
            Update Current Critic (Value Network) with TD Algorithm:
                1. Initialize the Gradient to Zero.
                2. Build a Tensor Contains All Target Actions Using Target Actor Networks and Next Observations.
                3. Calculate Target Critic Value:
                    - Combine Next Observation and Target Action in One To One Correspondence.
                    - Calculate Target Critic Value (TD Target).
                4. Calculate Critic Value:
                    - Combine Observation and Action in One To One Correspondence.
                    - Calculate Critic Value (TD Target).
                5. Calculate Critic's Loss Using MSELoss Function
                6. Backward Propagation.
                7. Update Parameters with Gradient Descent.
            Update Current Actor (Policy Network) with Deterministic Policy Gradient:
                1. Initialize the Gradient to Zero.
                2. Get Current Actors' Action in the Shape of One-Hot Vector.
                    - Get Current Actor Network Output with Current Observation.
                    - Transform the Output into a One-Hot Action Vector.
                3. Build the Input of Current Actor's Value Function.
                    - Build a tensor that contains all the actions.
                    - Combine Observations and Actions in One To One Correspondence.
                4. Calculate Actor's Loss Using Critic Network (Value Function) Output.
                5. Backward Propagation.
                6. Update Parameters with Gradient Descent.

        TODO:
        1. Why there is a term, "(1 - dones[agent_idx].view(-1, 1))", during calculation of TD target?
            To take termination in to consideration (To be verified).
        2. Why there is a term, "(current_actor_action_value ** 2).mean() * 1e-3", during calculation of Actor Loss?
            To make the output of the trained Actor more stable and smooth with this regular term (To be verified).
        """
        states, actions, rewards, next_states, dones = sample
        current_agent = self.agents[agent_idx]
        # Update Current Critic (Value Network) with TD Algorithm
        current_agent.critic_optimizer.zero_grad()
        # Here is the step to choose the actions from the actor network and we have two strategies.
        # Option 1: Use the target network strategy.
        target_action = [
            trans2onehot(_target_policy(_next_obs))
            for _target_policy, _next_obs in zip(self.target_policies, next_states)
        ]
        # Option 2: Use the double DQN strategy.
        # target_action = [
        #     # Choose actions from the original network!!!
        #     trans2onehot(policy(_next_obs))
        #     for policy, _next_obs in zip(self.policies, next_states)
        # ]
        target_critic_input = torch.cat((*next_states, *target_action), dim=1)
        target_critic_value = (rewards[agent_idx].view(-1, 1) +
                               self.gamma * current_agent.target_critic(target_critic_input) *
                               (1 - dones[agent_idx].view(-1, 1)))
        critic_input = torch.cat((*states, *actions), dim=1)
        critic_value = current_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        current_agent.critic_optimizer.step()
        # Update Current Actor (Policy Network) with Deep Deterministic Policy Gradient
        current_agent.actor_optimizer.zero_grad()
        current_actor_action_value = current_agent.actor(states[agent_idx])
        current_actor_action_onehot = gumbel_softmax(current_actor_action_value)
        all_actor_actions = [
            current_actor_action_onehot if i == agent_idx else trans2onehot(_policy(_state))
            for i, (_policy, _state) in enumerate(zip(self.policies, states))
        ]
        current_critic_input = torch.cat((*states, *all_actor_actions), dim=1)
        actor_loss = (-current_agent.critic(current_critic_input).mean() +
                      (current_actor_action_value ** 2).mean() * 1e-3)
        actor_loss.backward()
        current_agent.actor_optimizer.step()

    def update_all_targets_params(self):
        """Update all Target network's parameters using soft update method."""
        for agent in self.agents:
            agent.soft_update(agent.actor, agent.target_actor, self.tau)
            agent.soft_update(agent.critic, agent.target_critic, self.tau)