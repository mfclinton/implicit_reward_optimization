import torch
import numpy as np
# TODO
class TrajectoryBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, config):
        max_horizon = config.env.max_steps

        self.s = torch.zeros((buffer_size, max_horizon, state_dim), requires_grad=False)
        self.a = torch.zeros((buffer_size, max_horizon, action_dim), requires_grad=False)
        self.p = torch.zeros((buffer_size, max_horizon), requires_grad=False)
        self.r = torch.zeros((buffer_size, max_horizon), requires_grad=False)
        self.mask = torch.zeros((buffer_size, max_horizon), requires_grad=False)
        self.ids = torch.zeros(buffer_size, requires_grad=False)

        self.buffer_size = buffer_size
        self.episode_ctr = -1
        self.timestep_ctr = 0
        self.buffer_pos = -1
        self.valid_len = 0

    @property
    def size(self):
        return self.valid_len

    def reset(self):
        self.episode_ctr = -1
        self.timestep_ctr = 0
        self.buffer_pos = -1
        self.valid_len = 0

    def next(self):
        self.episode_ctr += 1
        self.buffer_pos += 1

        # Cycle around to the start of buffer (FIFO)
        if self.buffer_pos >= self.buffer_size:
            self.buffer_pos = 0

        if self.valid_len < self.buffer_size:
            self.valid_len += 1

        self.timestep_ctr = 0
        self.ids[self.buffer_pos] = self.episode_ctr

        self.r[self.buffer_pos].fill_(0)
        self.mask[self.buffer_pos].fill_(0)

    def add(self, s1, a1, p1, r1):
        pos = self.buffer_pos
        step = self.timestep_ctr
        print(f"Step {step} and Pos {pos}")

        self.s[pos][step] = torch.tensor(s1)
        self.a[pos][step] = torch.tensor(a1)
        self.p[pos][step] = torch.tensor(p1)
        self.r[pos][step] = torch.tensor(r1)
        self.mask[pos][step] = torch.tensor(1.0)

        self.timestep_ctr += 1

    def _get(self, idx):
        # ids represent the episode number
        # idx represents the buffer index
        # Both are not the same due to use of wrap around buffer
        ids = self.ids[idx]

        if self.valid_len >= self.buffer_size:
            # Subtract off the minimum value idx (as the idx has wrapped around in buffer)
            if self.buffer_pos + 1 == self.buffer_size:
                ids -= self.ids[0]
            else:
                ids -= self.ids[self.buffer_pos + 1]

        return ids, self.s[idx], self.a[idx], self.beta[idx], self.r[idx], self.mask[idx]

    def sample(self, batch_size):
        count = min(batch_size, self.valid_len)
        return self._get(np.random.choice(self.valid_len, count))