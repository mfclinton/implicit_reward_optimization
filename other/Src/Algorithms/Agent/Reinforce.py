import numpy as np
import torch
from Src.Algorithms.Agent.Agent import Agent
from Src.Utils.Policy import Categorical
from Src.Utils.utils import TrajectoryBuffer

class Reinforce(Agent):
    def __init__(self, basis):
        super(Reinforce, self).__init__()
        self.state_features = basis
    
    def init(self, config):
        print('~~~~~~~~~~~~~~~~')
        self.state_features.init(config)
        print("TROLL")
        print(self.state_features.feature_dim)
        self.policy = Categorical(self.state_features.feature_dim, config, action_dim=None)
        # self.memory = TrajectoryBuffer()
        self.counter = 0

        self.initialized = True
        # TODO

    def reset(self):
        super(Reinforce, self).reset()
        self.counter += 1
        # TODO
    
    # gets the action for the state
    def get_action(self, state):
        state = torch.tensor(state, requires_grad=False)
        state = self.state_features.forward(state.view(1, -1))
        action, prob, dist = self.policy.get_action_w_prob_dist(state)

        return action, prob, dist

    # Updates Batch Episode History
    def update(self, s1, a1, prob, done):
        # TODO
        self.memory.add(s1, a1, prob)

        if done:
            self.optimize()

    # Optimize Agent
    def optimize(self):
        # TODO
        pass