import numpy as np
import torch
from Src.Algorithms.Agent.Agent import Agent

class Reinforce(Agent):
    def __init__(self):
        super(Reinforce, self).__init__()
    
    def init(self, config):
        self.initialized = True

    def reset(self):
        super(Reinforce, self).reset()
    
    # gets the action for the state
    def get_action(self, state):
        pass

    # Updates Batch Episode History
    def update(self, s1, a1, prob, done):
        self.memory.add(s1, a1, prob)

        if done:
            self.optimize()

    # Optimize Agent
    def optimize(self):
        pass