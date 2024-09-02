import gymnasium as gym
import tianshou as ts
# import envpool
import torch, numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
# This is a simple environment with a discrete action space,
env = gym.make('CartPole-v1')
train_envs = gym.make('CartPole-v1')
test_envs = gym.make('CartPole-v1')

class Net(nn.Module):
    def __init__(self,state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )
    def forward(self, obs, state = None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        # the raw output of prediction/classification model is called logits
        logits = self.model(obs.view(batch, -1))
        return logits, state

if __name__ == '__main__':
    state_shape = env.observation_space.shape or env.observation_space.n #简化赋值：如果左侧有定义且非False、非None、非空等，则取左侧值；否则取右侧值。
    action_shape = env.action_space.shape or env.action_space.n  #左侧连续空间 、 右侧为离散空间
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=0.9,
        estimation_step=3,
        target_update_freq=320
    )
    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    result = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=10, step_per_epoch=10000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold
    ).run()
    print(f'Finished training! Use {result["duration"]}')

    writer = SummaryWriter('log/dqn')
    logger = TensorboardLogger(writer)

    torch.save(policy.state_dict(), 'dqn.pth')
    policy.load_state_dict(torch.load('dqn.pth'))