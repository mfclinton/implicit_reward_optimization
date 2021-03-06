import torch
from torch import nn

class Algorithm(nn.Module):
    def __init__(self):
        super(Algorithm, self).__init__()
        self.weight_decay = 0

    def init_optimizer(self):
        if hasattr(self, "amsgrad"):
            self.optim = self.optim_func(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad = self.amsgrad)
        else:
            self.optim = self.optim_func(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def step(self, loss = None, normalize_grad = False):

        if loss != None:
            self.optim.zero_grad()
            loss.backward()

        if normalize_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 30)
        self.optim.step()

    
