import numpy as np
import torch

class Agent:
    def __init__(self):
        self.initialized = False

    def init(self, config):
        self.initialized = True
    
    def reset(self):
        pass