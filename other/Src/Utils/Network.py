import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, num_inputs, num_outputs, config):
        super(Network, self).__init__()
        self.initialized = False

    def init(self, num_inputs, num_outputs, config):
        self.initialized = True

        self.config = config

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        #  TODO define model and optim