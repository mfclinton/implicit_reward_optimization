import numpy as np
import torch
from Src.Algorithms.Algorithm import Algorithm
import torch.nn as nn

class GammaFunc(Algorithm):
    def __init__(self, optim=torch.optim.Adam, lr=.01):
        super(GammaFunc, self).__init__()
        self.optim = optim
        self.lr = lr

    def init(self, config):
        self.state_dim = config.basis.feature_dim
        self.action_dim = config.env.action_space.n

        self.fc1 = nn.Linear(self.state_dim, self.action_dim, bias=False)
        self.init_optimizer()

    # TODO: Only categorical
    def forward(self, state, action):
        action_indexes = torch.nonzero(action)
        x = self.fc1(state)
        x = torch.clip(x, 0, 1)
        in_g = torch.zeros(state.size()[:-1])
        in_g[action_indexes[:,0]] = x[action_indexes[:,0], action_indexes[:,1]]
        return in_g