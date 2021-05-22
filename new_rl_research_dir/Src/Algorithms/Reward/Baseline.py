import torch
from torch import nn

class Baseline(nn.Module):
    def __init__(self, state_dim, inner_dim=32):
        super(Baseline, self).__init__()
        self.state_dim = state_dim
        self.value_fc1 = torch.nn.Linear(self.state_dim, inner_dim)
        self.value_fc2 = torch.nn.Linear(inner_dim, 1)

    
    def forward(self, states):
        # print(self.state_dim, states.size())

        x = self.value_fc1(states)
        x = self.value_fc2(x)
        return x

    
