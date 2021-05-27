import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Src.Algorithms.Algorithm import Algorithm

class Policy(Algorithm):
    def __init__(self):
        super(Policy, self).__init__()

    def init(self, config, amsgrad=True):
        self.config = config
        self.state_dim = config.basis.feature_dim
        self.action_dim = config.env.action_space.n
        self.amsgrad=amsgrad

class Categorical(Policy):
    def __init__(self, optim=torch.optim.Adam, lr=.01, weight_decay = 0):
        super(Categorical, self).__init__()
        self.optim_func = optim
        self.lr = lr
        self.weight_decay = weight_decay

    def init(self, config, action_dim=None):
        super(Categorical, self).init(config)
        # overrides the action dim variable defined by super-class

        if action_dim is not None:
            self.action_dim = action_dim
            
        self.fc1 = nn.Linear(self.state_dim, self.action_dim, bias=False)
        self.init_optimizer()

    def forward(self, state):
        x = self.fc1(state)
        return x

    def get_action_w_prob_dist(self, state, explore=0):
        x = self.forward(state)
        dist = F.softmax(x, -1)

        probs = dist.cpu().view(-1).data.numpy()

        action = torch.zeros(self.action_dim)
        action_idx = np.random.choice(self.action_dim, p=probs)
        action[action_idx] = 1.0

        return action, probs[action_idx], probs

    # def get_prob(self, state, action):
    #     x = self.forward(state)
    #     dist = F.softmax(x, -1)
    #     return dist.gather(1, action), dist

    def get_logprob_dist(self, state, action):
        action = action.long() #TODO: do better

        x = self.forward(state)                                                              # BxA
        log_dist = F.log_softmax(x, -1)
        
        action_indexes = torch.nonzero(action) #TODO: Assumes Categorical
        log_pi = torch.zeros(x.size()[:-1])
        log_pi[action_indexes[:,0]] = log_dist[action_indexes[:,0], action_indexes[:,1]]

        # TODO: Make assumption about discreteness
        return log_pi, log_dist                                          # BxAx(Bx1) -> B