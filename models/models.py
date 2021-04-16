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
from utils.new_helper import *
import time

# 
# POLICY MODELS
# 
# INITIALIZATION TO ALL THE SAME CONSTANT COULD CAUSE ISSUES FOR INVERTIBILTY
class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, num_outputs, bias=False).double()
        self.linear1.weight.data.fill_(0.)

        self.sm = nn.Softmax(dim=0)

    def forward(self, inputs):
        x = inputs
        action_scores = self.linear1(x)
        return self.sm(action_scores)

class REINFORCE:
    def __init__(self, num_inputs, action_space):
        self.num_inputs = num_inputs
        self.action_space = action_space
        self.reset()

    def reset(self):
        # TODO: is hardcoded, fix for more complex networks
        self.model = Policy(self.num_inputs, self.action_space.n)
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)


    def select_action(self, state):
        probs = self.model(state)
        m = Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)

    def update_parameters(self, rewards, log_probs, cumu_gamma):
        # returns = get_returns_t(rewards, gamma, normalize=False)
        returns = Get_Discounted_Returns(rewards, cumu_gamma, normalize=False)

        self.optimizer.zero_grad()

        policy_loss = 0
        for log_prob, R in zip(log_probs, returns):
            policy_loss += -log_prob * R

        policy_loss.backward(retain_graph=True)
        utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()

class Chris_Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Chris_Policy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, num_outputs, bias=False).double()
        self.linear1.weight.data.fill_(0.)

    def forward(self, inputs):
        x = inputs
        action_scores = self.linear1(x)
        return action_scores

class CHRIS_REINFORCE:
    def __init__(self):
        self.num_inputs = 7
        self.num_actions = 7
        self.reset()

    def reset(self):
        # TODO: is hardcoded, fix for more complex networks
        self.model = Chris_Policy(self.num_inputs, self.num_actions)
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1)


    def select_action(self, state):
        probs = self.model(state)
        
        theta = probs[torch.nonzero(state)[0]]

        p_0 = 1 / (1 + torch.exp(- theta))
        p_1 = 1 - p_0
        probs = torch.stack([p_0, p_1], dim=1)[0]

        m = Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)

    def update_parameters(self, rewards, log_probs, cumu_gamma):
        # returns = get_returns_t(rewards, gamma, normalize=False)
        returns = Get_Discounted_Returns(rewards, cumu_gamma, normalize=False)

        self.optimizer.zero_grad()

        policy_loss = 0
        for log_prob, R in zip(log_probs, returns):
            policy_loss += -log_prob * R

        policy_loss.backward(retain_graph=True)
        # utils.clip_grad_norm_(self.model.parameters(), 40)
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
        x = torch.clip(x, -10, 10)
        # x = torch.tanh(x)
        # x = torch.sigmoid(x)

        return x

class INTRINSIC_REWARD:
    def __init__(self, num_inputs):
        self.model = Reward(num_inputs)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-1)
        self.model.train()

    def get_reward(self, state_action):
        r = self.model(state_action)

        # TEMP TEST
        # non_zero_idx = state_action.nonzero()[:,1]
        # states = non_zero_idx % 3
        # actions = non_zero_idx // 3

        # temp_reward = torch.tensor([[0,0,0],[0,1,0]])
        # r = temp_reward[actions,states]


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
        # TODO: (o___o)
        gamma = torch.full_like(gamma,0.4) # TODO: REMOVE THISSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSs
        return gamma
