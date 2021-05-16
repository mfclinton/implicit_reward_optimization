import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, num_inputs, num_outputs, config):
        super(Network, self).__init__()
        self.initialized = False

    def init(self, num_inputs, num_outputs, config):
        self.initialized = True

        self.config = config

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        #  TODO define model and optim

class Categorical(Network):
    def __init__(self, state_dim, config, action_dim=None):
        super(Categorical, self).__init__(state_dim, config)

        # overrides the action dim variable defined by super-class
        if action_dim is not None:
            self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, self.action_dim)
        self.init()

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
        x = self.forward(state)                                                              # BxA
        log_dist = F.log_softmax(x, -1)                                                      # BxA
        return log_dist.gather(1, action), log_dist                                          # BxAx(Bx1) -> B