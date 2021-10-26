import numpy as np
import torch
from Src.Algorithms.Agent.Agent import Agent
from Src.Utils.Policy import Categorical
from Src.Utils.utils import TrajectoryBuffer
from Src.Algorithms.Reward.Baseline import Baseline

class Reinforce(Agent):
    def __init__(self, policy, baseline):
        super(Reinforce, self).__init__()

        self.policy = policy
        self.baseline = baseline
    
    def init(self, config):
        super(Reinforce, self).init(config)
        self.policy.init(config)
        self.baseline.init(config)

        # TODO: These are hardcodes - also not sure if I need the Baseline anymore
        # self.policy = Categorical(config) #TODO, Dynamic
        # self.baseline = Baseline(config.basis.feature_dim)

        self.counter = 0

        self.initialized = True
        # TODO
        # TODOOOO

    def reset(self):
        super(Reinforce, self).reset()
        self.counter += 1
        # TODO

    # Optimize Agent
    # TODO: TEMP GAMMA
    def optimize(self, s_features, a, r, gamma=1.0):
        if not torch.is_tensor(gamma):
            gamma = torch.full_like(r, gamma)
        
        r = r.detach()
        gamma = gamma.detach()

        B, H, A = a.shape

        # State Values for Baseline
        state_values = self.baseline(s_features).view(B, H)

        log_pi, dist_all = self.policy.get_logprob_dist(s_features, a.view(B * H, -1))     # (BxH)xd, (BxH)xA
        log_pi = log_pi.view(B, H)                                                       # (BxH)x1 -> BxH
        pi_a = torch.exp(log_pi)

        # TODO: Make sure it doesnt modify
        returns = r
        for i in range(H-2, -1, -1):
            returns[:, i] += returns[:, i+1] * gamma[:, i]

        loss = 0
        log_pi_return = torch.sum(log_pi * (returns - state_values.detach()), dim=-1, keepdim=True)
        # log_pi_return = torch.sum(log_pi * returns, dim=-1, keepdim=True)
        loss += - 1.0 * torch.sum(log_pi_return)
        
        sv_loss = torch.nn.functional.mse_loss(state_values, returns, reduction="mean")

        
        # TODO: Insert Lambda ?
        self.policy.step(loss)
        self.baseline.step(sv_loss)

        # TODO: CHECK THIS
        

