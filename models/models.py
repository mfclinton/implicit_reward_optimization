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
from utils.helper import *

# 
# POLICY MODELS
# 
# INITIALIZATION TO ALL THE SAME CONSTANT COULD CAUSE ISSUES FOR INVERTIBILTY
class Policy(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.n

        self.linear1 = nn.Linear(num_inputs, num_outputs, bias=False).double()
        self.linear1.weight.data.fill_(0.) 
        self.sm = nn.Softmax(dim=0)

    def forward(self, inputs):
        x = inputs
        action_scores = self.linear1(x)
        return self.sm(action_scores)

class REINFORCE:
    def __init__(self, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(num_inputs, action_space)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-1)
        self.model.train()

    def select_action(self, state):
        probs = self.model(state)
        m = Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)

    def update_parameters(self, rewards, log_probs, gamma):
        # R = 0
        policy_loss = 0
        # returns = []
        # for i in reversed(range(len(rewards))):
        #     r = rewards[i]
        #     R = r + gamma * R
        #     returns.insert(0, R)

        # returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + eps)

        returns = get_returns_t(rewards, gamma, normalize=True)
        # print(returns)

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
    def __init__(self, num_inputs):
        super(Reward, self).__init__()

        self.linear1 = nn.Linear(num_inputs, 1, bias=False).double()
        self.linear1.weight.data.fill_(0.0)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = torch.tanh(x)

        return x

class INTRINSIC_REWARD:
    def __init__(self, num_inputs):
        self.model = Reward(num_inputs)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=1e-5)
        self.model.train()

    def get_reward(self, state_action):
        r = self.model(state_action)
        return r

# 
# GAMMA MODELS
# 
class Gamma(nn.Module):
    def __init__(self, num_inputs):
        super(Gamma, self).__init__()

        self.linear1 = nn.Linear(num_inputs, 1, bias=False).double()
        self.linear1.weight.data.fill_(0.) 
        

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        gamma = torch.sigmoid(x)
        return gamma

class INTRINSIC_GAMMA:
    def __init__(self, num_inputs):
        self.model = Gamma(num_inputs)
        self.model = self.model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-1, amsgrad=True)
        self.model.train()

    def get_gamma(self, state):
        gamma = self.model(state)
        gamma = torch.full_like(gamma,0.95) # TODO: REMOVE THISSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSs
        return gamma
