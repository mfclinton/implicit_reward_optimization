import numpy as np
import torch
from Src.Algorithms.Agent.Agent import Agent
from Src.Utils.Policy import Categorical
from Src.Utils.utils import TrajectoryBuffer

class Reinforce(Agent):
    def __init__(self, basis, buffer_size=100, batch_size=20):
        super(Reinforce, self).__init__()
        self.state_features = basis

        self.buffer_size = buffer_size
        self.batch_size = batch_size
    
    def init(self, config):
        super(Reinforce, self).init(config)
        self.state_features.init(config)
        self.policy = Categorical(self.state_features.feature_dim, config, action_dim=None)
        self.memory = TrajectoryBuffer(self.buffer_size, self.state_dim, self.action_dim, config)
        self.counter = 0

        self.initialized = True
        # TODO

    def reset(self):
        super(Reinforce, self).reset()
        self.counter += 1
        self.memory.next()
        # TODO
    
    # gets the action for the state
    def get_action(self, state):
        state = torch.tensor(state, requires_grad=False)
        state = self.state_features.forward(state.view(1, -1))
        action, prob, dist = self.policy.get_action_w_prob_dist(state)

        return action, prob, dist

    # Updates Batch Episode History
    def update(self, s1, a1, prob, r1, s2, valid, done):
        self.memory.add(s1, a1, prob, r1)

        if done:
            self.optimize()

    # Optimize Agent
    def optimize(self):
        batch_size = self.memory.size if self.memory.size < self.batch_size else self.batch_size
        