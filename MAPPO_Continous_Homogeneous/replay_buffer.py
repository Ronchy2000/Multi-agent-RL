import numpy as np
import torch


class ReplayBuffer:
    """
    MAPPO ReplayBuffer for **continuous actions**.
    Each agent's action is a vector of length `act_dim`.
    """
    def __init__(self, args):
        self.N = args.N                          # number of agents
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.act_dim = args.act_dim              # <— NEW
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size

        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _empty(self, *shape):
        """Convenience: allocate float32 Numpy array."""
        return np.empty(shape, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # core API
    # ------------------------------------------------------------------ #
    def reset_buffer(self):
        """Allocate a fresh buffer for a new batch of episodes."""
        self.buffer = {
            # observations: (B, T, N, obs_dim)
            'obs_n':       self._empty(self.batch_size, self.episode_limit, self.N, self.obs_dim),

            # global state: (B, T, state_dim)
            's':           self._empty(self.batch_size, self.episode_limit, self.state_dim),

            # value estimates: (B, T + 1, N)
            'v_n':         self._empty(self.batch_size, self.episode_limit + 1, self.N),

            # actions: (B, T, N, act_dim)   ← continuous vector
            'a_n':         self._empty(self.batch_size, self.episode_limit, self.N, self.act_dim),

            # old log‑probs: (B, T, N)
            'a_logprob_n': self._empty(self.batch_size, self.episode_limit, self.N),

            # rewards: (B, T, N)
            'r_n':         self._empty(self.batch_size, self.episode_limit, self.N),

            # done flags: (B, T, N)
            'done_n':      self._empty(self.batch_size, self.episode_limit, self.N),
        }
        self.episode_num = 0

    def store_transition(
        self,
        episode_step,
        obs_n, s, v_n,
        a_n, a_logprob_n,
        r_n, done_n
    ):
        """
        Store one timestep for every agent in the current episode (#episode_num).
        Shapes expected:
            obs_n        (N, obs_dim)
            s            (state_dim,)
            v_n          (N,)
            a_n          (N, act_dim)     ← continuous
            a_logprob_n  (N,)
            r_n          (N,)
            done_n       (N,)
        """
        idx = self.episode_num
        self.buffer['obs_n'][idx, episode_step]       = obs_n
        self.buffer['s'][idx, episode_step]           = s
        self.buffer['v_n'][idx, episode_step]         = v_n
        self.buffer['a_n'][idx, episode_step]         = a_n
        self.buffer['a_logprob_n'][idx, episode_step] = a_logprob_n
        self.buffer['r_n'][idx, episode_step]         = r_n
        self.buffer['done_n'][idx, episode_step]      = done_n

    def store_last_value(self, episode_step, v_n):
        """
        After an episode ends, store the bootstrap value for V(s_{T}).
        `episode_step` should be T (i.e. episode length).
        """
        self.buffer['v_n'][self.episode_num, episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        """
        Convert the whole buffer to torch tensors.
        For continuous actions we keep everything in float32.
        """
        batch = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in self.buffer.items()
        }
        return batch

# ------------------------------------------------------------------ #
# 使用说明
# ------------------------------------------------------------------ #
# # 假设 args 额外包含 act_dim
# buffer = ReplayBuffer(args)
#
# # 在环境循环里：
# buffer.store_transition(
#     t,
#     obs_n=obs_array,                    # shape (N, obs_dim)
#     s=global_state,                    # shape (state_dim,)
#     v_n=values,                        # shape (N,)
#     a_n=actions,                       # shape (N, act_dim)  ← continuous
#     a_logprob_n=log_probs,             # shape (N,)
#     r_n=rewards,                       # shape (N,)
#     done_n=dones                       # shape (N,)
# )
#
# # episode 结束后
# buffer.store_last_value(t+1, bootstrap_values)