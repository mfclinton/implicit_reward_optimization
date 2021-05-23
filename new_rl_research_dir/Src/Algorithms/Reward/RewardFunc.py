import numpy as np
import torch
from Src.Algorithms.Algorithm import Algorithm
import torch.nn as nn

class RewardFunc(Algorithm):
    def __init__(self, state_dim, config, optim=torch.optim.Adam, lr=.01):
        super(RewardFunc, self).__init__()

        self.fc1 = nn.Linear(self.state_dim, 1)
        self.init_optimizer(optim, lr)


    def forward(self, state):
        x = self.fc1(state)
        return x

    

        