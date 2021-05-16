import numpy as np
import torch

class GammaFunc():
    def __init__(self):
        self.initialized = False

    def init(self, config):
        self.initialized = True