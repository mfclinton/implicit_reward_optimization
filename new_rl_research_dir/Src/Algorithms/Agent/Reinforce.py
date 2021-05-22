import numpy as np
import torch
from Src.Algorithms.Agent.Agent import Agent
from Src.Utils.Policy import Categorical
from Src.Utils.utils import TrajectoryBuffer
from Src.Algorithms.Reward.Baseline import Baseline

class Reinforce(Agent):
    def __init__(self, basis, buffer_size=100, batch_size=20, lr = 0.1):
        super(Reinforce, self).__init__()
        self.state_features = basis

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr = lr
    
    def init(self, config):
        super(Reinforce, self).init(config)
        self.state_features.init(config)

        self.policy = Categorical(self.state_features.feature_dim, config, action_dim=None) #TODO, Dynamic
        self.baseline = Baseline(self.state_features.feature_dim)

        self.memory = TrajectoryBuffer(self.buffer_size, self.state_dim, self.action_dim, config)
        self.counter = 0

        self.initialized = True
        # TODO
        # TODOOOO

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

        if done and self.batch_size <= self.memory.valid_len:
            self.optimize()

    # Optimize Agent
    def optimize(self):
        batch_size = self.memory.size if self.memory.size < self.batch_size else self.batch_size
        # s, a, prob, r = self.memory.sample(batch_size)
        ids, s, a, prob, r, mask = self.memory.sample(batch_size)

        B, H, D = s.shape
        _, _, A = a.shape
        print("MAX : ", s[0].max(dim=0))
        print("MIN : ", s[0].min(dim=0))

        # print(a)

        s_feature = self.state_features.forward(s.view(B * H, D))

        # State Values for Baseline
        state_values = self.baseline(s_feature).view(B, H)

        # Masking
        state_values *= mask #TODO: Mask anything else?

        log_pi, dist_all = self.policy.get_logprob_dist(s_feature, a.view(B * H, -1))     # (BxH)xd, (BxH)xA
        log_pi = log_pi.view(B, H)                                                       # (BxH)x1 -> BxH
        pi_a = torch.exp(log_pi)

        # TODO: Make sure it doesnt modify
        returns = r
        # print(state_values, r)

        # gamma = 0.9 # TODO: TEMP gamma
        gamma = 1.0
        for i in range(H-2, -1, -1):
            returns[:, i] += returns[:, i+1] * gamma


        loss = 0
        log_pi_return = torch.sum(log_pi * (returns - state_values.detach()), dim=-1, keepdim=True)
        loss += - 1.0 * torch.sum(log_pi_return)
        
        sv_loss = torch.nn.functional.mse_loss(state_values, returns, reduction="mean")

        
        # TODO: Insert Lambda ?
        self.policy.step(loss)
        self.baseline.step(sv_loss)

        # TODO: REMOVE THIS
        self.memory.reset()

