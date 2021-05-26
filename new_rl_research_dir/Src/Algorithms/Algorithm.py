import torch
from torch import nn

class Algorithm(nn.Module):
    def __init__(self):
        super(Algorithm, self).__init__()

    def init_optimizer(self):
        self.optim = self.optim_func(self.parameters(), lr=self.lr, weight_decay=self.config.weight_decay)

    def step(self, loss = None, normalize_grad = False):

        if loss != None:
            self.optim.zero_grad()
            loss.backward()

        if normalize_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 30)
        self.optim.step()

    
