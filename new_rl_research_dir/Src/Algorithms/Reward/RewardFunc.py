import numpy as np
import torch
from Src.Algorithms.Algorithm import Algorithm
import torch.nn as nn

# TODO: ONLY CATEGORICAL
class RewardFunc(Algorithm):
    def __init__(self, optim=torch.optim.Adam, lr=.01):
        super(RewardFunc, self).__init__()
        self.optim_func = optim
        self.lr = lr

    def init(self, config):
        self.config = config
        self.state_dim = config.basis.feature_dim
        self.action_dim = config.env.action_space.n

        self.fc1 = nn.Linear(self.state_dim, self.action_dim, bias=False)
        self.fc1.weight.data.fill_(0.0)
        print("INITIALIZE REWARD FUNCTION")
        self.init_optimizer()

    # TODO: Get Auxillary Code From Other Env

    # TODO: Only categorical
    def forward(self, state, action=None, min=-20, max=20):
        x = self.fc1(state)
        x = torch.clip(x, min, max)
        if action is None:
            return x

        action_indexes = torch.nonzero(action)
        in_r = torch.zeros(state.size()[:-1])
        in_r[action_indexes[:,0]] = x[action_indexes[:,0], action_indexes[:,1]]
        return in_r

    

        