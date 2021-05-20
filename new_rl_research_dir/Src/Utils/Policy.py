import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    def __init__(self, state_dim, config, action_dim=None):
        super(Policy, self).__init__()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = config.env.action_space.n

class Categorical(Policy):
    def __init__(self, state_dim, config, action_dim=None):
        super(Categorical, self).__init__(state_dim, config)
        # overrides the action dim variable defined by super-class
        if action_dim is not None:
            self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, self.action_dim)

    def re_init_optim(self):
        # TODO
        self.optim = self.config.optim(self.parameters(), lr=self.config.actor_lr)

    def forward(self, state):
        x = self.fc1(state)
        return x

    def get_action_w_prob_dist(self, state, explore=0):
        x = self.forward(state)
        dist = F.softmax(x, -1)

        probs = dist.cpu().view(-1).data.numpy()
        action = np.random.choice(self.action_dim, p=probs)

        return action, probs[action], probs

    def get_prob(self, state, action):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        return dist.gather(1, action), dist

    def get_logprob_dist(self, state, action):
        action = action.long() #TODO: do better

        x = self.forward(state)                                                              # BxA
        log_dist = F.log_softmax(x, -1)
        # TODO: Make assumption about discreteness
        return log_dist.sum(dim=1), log_dist                                          # BxAx(Bx1) -> B