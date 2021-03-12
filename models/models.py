import sys
import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb
import torch.nn.utils as utils
from torch.distributions import Categorical

# 
# POLICY MODELS
# 
class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.n

        self.linear1 = nn.Linear(num_inputs, hidden_size).double()
        self.linear2 = nn.Linear(hidden_size, num_outputs).double()
        self.sm = nn.Softmax(dim=0)

    def forward(self, inputs):
        x = inputs
        # print(self.linear1.weight.dtype)
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return self.sm(action_scores)

class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def select_action(self, state):
        probs = self.model(Variable(state).cuda())
        m = Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)

    def update_parameters(self, rewards, log_probs, gamma):
        eps = 1e-5
        R = 0
        policy_loss = 0
        returns = []
        for i in reversed(range(len(rewards))):
            r = rewards[i]
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        self.optimizer.zero_grad()
        for log_prob, R in zip(log_probs, returns):
            policy_loss += -log_prob * R

        # print("Loss : ", policy_loss)

        policy_loss.backward()
        # utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()


# 
# REWARD MODELS
# 
class Reward(nn.Module):
    def __init__(self, hidden_size, num_inputs):
        super(Reward, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size).double()
        self.linear2 = nn.Linear(hidden_size, 1).double()

    def forward(self, inputs):
        x = inputs
        # print(self.linear1.weight.dtype)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        r = torch.sigmoid(x)
        return r

class INTRINSIC_REWARD:
    def __init__(self, hidden_size, num_inputs):
        self.model = Reward(hidden_size, num_inputs)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def get_reward(self, state_action):
        r = self.model(Variable(state_action).cuda())
        return r

# 
# GAMMA MODELS
# 
class Gamma(nn.Module):
    def __init__(self, hidden_size, num_inputs):
        super(Gamma, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size).double()
        self.linear2 = nn.Linear(hidden_size, 1).double()

    def forward(self, inputs):
        x = inputs
        # print(self.linear1.weight.dtype)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        gamma = torch.sigmoid(x)
        return gamma

class INTRINSIC_GAMMA:
    def __init__(self, hidden_size, num_inputs):
        self.model = Gamma(hidden_size, num_inputs)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def get_gamma(self, state):
        gamma = self.model(Variable(state).cuda())
        return gamma
