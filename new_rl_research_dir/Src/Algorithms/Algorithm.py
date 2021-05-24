import torch
from torch import nn

class Algorithm(nn.Module):
    def __init__(self):
        super(Algorithm, self).__init__()

    def init_optimizer(self):
        # print("LOL HAHAHAHA")
        self.optim = self.optim(self.parameters(), lr=self.lr)

    def step(self, loss = None, normalize_grad = True):
        # print(self.optim)
        self.optim.zero_grad()

        if loss != None:
            loss.backward()

        if normalize_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optim.step()

    
