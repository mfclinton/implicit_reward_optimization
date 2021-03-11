import torch
import torch.nn.functional as F

print(F.one_hot(torch.tensor(3),num_classes=25))