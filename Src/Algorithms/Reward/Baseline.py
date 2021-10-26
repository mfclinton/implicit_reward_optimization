import torch
from torch import nn
from Src.Algorithms.Algorithm import Algorithm

class Baseline(Algorithm):
    def __init__(self, optim=torch.optim.Adam, lr=.01):
        super(Baseline, self).__init__()
        self.optim_func = optim
        self.lr = lr

    def init(self, config):
        self.config = config
        self.state_dim = config.basis.feature_dim
        self.value_fc1 = torch.nn.Linear(self.state_dim, 1)
        self.init_optimizer()

        # self.value_fc2 = torch.nn.Linear(inner_dim, 1)

    
    def forward(self, states):
        # print(self.state_dim, states.size())

        x = self.value_fc1(states)
        # x = self.value_fc2(x)
        return x

    # def step(self, loss, normalize_grad = True):
    #     self.optim.zero_grad()
    #     loss.backward()
    #     if normalize_grad:
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
    #     self.optim.step()

    
