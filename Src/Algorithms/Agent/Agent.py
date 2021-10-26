import numpy as np
import torch

class Agent:
    def __init__(self):
        self.initialized = False

    def init(self, config):
        self.initialized = True
        self.state_dim = config.env.observation_space.shape[0]
        self.action_dim = config.env.action_space.n
        self.config = config
    
    def reset(self):
        pass