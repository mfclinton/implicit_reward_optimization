
# TODO
class TrajectoryBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, config):
        max_horizon = config.env.max_steps
        self.s = torch.zeros((buffer_size, max_horizon, state_dim), dtype=stype, requires_grad=False)
        self.a = torch.zeros((buffer_size, max_horizon, action_dim), dtype=atype, requires_grad=False)
        
        
        self.episode_ctr = -1
        self.timestep_ctr = 0
        self.buffer_pos = -1
        self.valid_len = 0

    def reset(self):
        self.episode_ctr = -1
        self.timestep_ctr = 0
        self.buffer_pos = -1
        self.valid_len = 0

    def next(self):
        pass

    def add(self, s1, a1):
        pos = self.buffer_pos
        step = self.timestep_ctr

        self.s[pos][step] = torch.tensor(s1, dtype=self.stype)
        self.a[pos][step] = torch.tensor(a1, dtype=self.atype)

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