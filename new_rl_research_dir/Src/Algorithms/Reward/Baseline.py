import torch
from torch import nn
from Src.Algorithms.Algorithm import Algorithm

class Baseline(Algorithm):
    def __init__(self, state_dim, optim=torch.optim.Adam, inner_dim=32, lr=.01):
        super(Baseline, self).__init__()
        self.state_dim = state_dim
        self.value_fc1 = torch.nn.Linear(self.state_dim, 1)
        self.init_optimizer(optim, lr)

        # self.value_fc2 = torch.nn.Linear(inner_dim, 1)

        # self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    
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

    
