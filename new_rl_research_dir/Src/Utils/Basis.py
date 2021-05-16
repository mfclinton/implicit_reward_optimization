import numpy as np
import torch
import torch.nn as nn

def get_Basis(config, env):
    if config.one_hot:
        return OneHot_Basis(config, env)
    else:
        raise Exception("Invalid Basis Function")

class Basis(nn.Module):
    def __init__(self):
        super(Basis, self).__init__()
        self.initialized = False

    def init(self, config):
        self.initialized = True
        # self.state_low = tensor(env.observation_space.low, dtype=float32, requires_grad=False)
        # self.state_high = tensor(env.observation_space.high, dtype=float32, requires_grad=False)

# Designed for Discrete Grids
class OneHot_Basis(Basis):
    def __init__(self):
        super(OneHot_Basis, self).__init__()

    def init(self, config):
        env = config.env
        super(OneHot_Basis, self).init(config)
        assert np.issubdtype(env.observation_space.dtype, np.integer)

        self.feature_dim = env.n_observations
        self.width = env.width

    def forward(self, state):
        state = state[0]
        idx = state[0] + self.width * state[1]
        output = torch.zeros(self.feature_dim)
        output[idx] = 1
        return output