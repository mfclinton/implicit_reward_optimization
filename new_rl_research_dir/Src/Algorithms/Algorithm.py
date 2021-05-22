import torch
from torch import nn

class Algorithm(nn.Module):
    def __init__(self):
        super(Algorithm, self).__init__()

    def init_optimizer(self, optim, lr):
        print("LOL HAHAHAHA")
        self.optim = optim(self.parameters(), lr=lr)

    def step(self, loss, normalize_grad = True):
        self.optim.zero_grad()
        loss.backward()
        if normalize_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optim.step()

    
