import numpy as np
import torch
from Src.Algorithms.Agent.Agent import Agent

class Reinforce(Agent):
    def __init__(self, basis):
        super(Reinforce, self).__init__()
        self.state_features = basis
    
    def init(self, config):
        self.state_features.init()

        self.initialized = True
        # TODO

    def reset(self):
        super(Reinforce, self).reset()
        # TODO
    
    # gets the action for the state
    def get_action(self, state):
        state = torch.Tensor(state, requires_grad=False)
        state = self.state_features.forward(state.view(1, -1))
        # TODO
        # action, prob, dist = self.actor.get_action_w_prob_dist(state)



        # if self.config.debug:
        #     self.track_entropy(dist, action)

        return action, prob, dist

    # Updates Batch Episode History
    def update(self, s1, a1, prob, done):
        self.memory.add(s1, a1, prob)

        if done:
            self.optimize()

    # Optimize Agent
    def optimize(self):
        # TODO
        pass