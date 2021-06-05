import numpy as np
import torch
import torch.nn as nn
import itertools

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
class Raw_Basis(Basis):
    def __init__(self):
        super(Raw_Basis, self).__init__()

    def init(self, config):
        super(Raw_Basis, self).init(config)
        env = config.env

        self.feature_dim = env.n_observations

    def forward(self, state):

        return state.float()

# Designed for Discrete Grids
class OneHot_Basis(Basis):
    def __init__(self):
        super(OneHot_Basis, self).__init__()

    def init(self, config):
        env = config.env
        super(OneHot_Basis, self).init(config)
        assert np.issubdtype(env.observation_space.dtype, np.integer)

        self.feature_dim = env.n_observations
        self.width = env.width #TODO: Specific to Gridworld

    def forward(self, state):
        N, _ = state.size()
        idx = (state[:,:1] + self.width * state[:,1:]).long() #TODO do better, Assumes a width parameter

        output = torch.zeros((N, self.feature_dim)) #TODO: PRE ALLOCATE?

        output[torch.arange(N), idx.squeeze()] = 1

        return output

class Fourier_Basis(Basis):
    def __init__(self, fourier_order, fourier_coupled):
        super(Fourier_Basis, self).__init__()
        self.fourier_order = fourier_order
        self.fourier_coupled = fourier_coupled

    def init(self, config):
        env = config.env
        super(Fourier_Basis, self).init(config)
        dim = env.observation_space.shape[0]
        order = self.fourier_order
        # print("-------------------------------------------")

        if self.fourier_coupled:
            if (order+1)**dim > 1000:
                raise ValueError("Reduce Fourier order please... ")

            coeff = np.arange(0, order+1)
            weights = torch.from_numpy(np.array(list(itertools.product(coeff, repeat=dim))).T)  # size = n**d
            self.get_basis = self.coupled
            self.feature_dim = weights.shape[-1]
        else:
            weights = torch.from_numpy(np.arange(1, order + 1))
            self.get_basis = self.uncoupled
            self.feature_dim = weights.shape[-1] * dim

        self.basis_weights = weights.type(torch.FloatTensor).requires_grad_(False)
        self.dummy_param = torch.nn.Parameter(torch.rand(1).type(torch.FloatTensor))
        # self.init()
        # print("CREATED FOURIER")

    def coupled(self, x):
        # Creates a cosine only basis having order^(dim) terms
        basis = torch.matmul(x, self.basis_weights)
        basis = torch.cos(basis * np.pi)
        return basis

    def uncoupled(self, x):
        x = x.unsqueeze(2)  # convert shape from r*c to r*c*1
        basis = x * self.basis_weights  # Broadcast multiplication r*c*1 x 1*d => r*c*d
        basis = torch.cos(basis * np.pi)
        return basis.view(x.shape[0], -1)  # convert shape from r*c*d => r*(c*d)

    def preprocess(self, state):
        x = state
        if state.dtype is not torch.float32:
            x = state.float()
        return x

    def forward(self, state):
        # print(state, state.size())
        # TODO: from OG code
        x = self.preprocess(state)
        # print(x, self.get_basis(x))
        # 1/0
        return self.get_basis(x)
