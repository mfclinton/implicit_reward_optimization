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

class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.n

        self.linear1 = nn.Linear(num_inputs, hidden_size).double()
        self.linear2 = nn.Linear(hidden_size, num_outputs).double()

    def forward(self, inputs):
        x = inputs
        # print(self.linear1.weight.dtype)
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores)


class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def select_action(self, state):
        probs = self.model(Variable(state).cuda()) 
        action = probs.multinomial(1).data
        # print(probs)
        prob = probs[action]
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()

        return action[0], log_prob[0], entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R)).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
        loss = loss / len(rewards)
        print("Loss : " + str(loss.item()))
        # print(list(self.model.parameters())[0])
		
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()
