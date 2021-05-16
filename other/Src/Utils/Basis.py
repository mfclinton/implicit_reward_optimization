import numpy as np

def get_Basis(config, env):
    if config.one_hot:
        return OneHot_Basis(config, env)
    else:
        raise Exception("Invalid Basis Function")

class Basis(nn.Module):
    def __init__(self):
        super(Basis, self).__init__()
        self.initialized = False

    def __init__(self, config, env):
        self.initialized = True
        self.state_low = tensor(env.observation_space.low, dtype=float32, requires_grad=False, device=config.device)
        self.state_high = tensor(env.observation_space.high, dtype=float32, requires_grad=False, device=config.device)

# Designed for Discrete Grids
class OneHot_Basis(Basis):
    def __init__(self, config, env):
        super().__init__(config)
        assert np.issubdtype(env.observation_space.dtype, np.integer)

        self.feature_dim = env.n_observations
        self.width = env.width

    def forward(self, state):
        idx = state[0] + self.width * state[1]
        output = torch.zeros(self.feature_dim)
        output[idx] = 1
        return output