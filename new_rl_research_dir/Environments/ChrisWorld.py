import gym
from gym import error, spaces, utils
from gym.spaces import Discrete
from gym.utils import seeding
import logging
import torch
import numpy as np
import math
import matplotlib.pyplot  as plt
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.ticker import NullLocator


class ChrisWorld(object):
    def __init__(self):
        # customizable parameters
        self.n_observations = 1

        self.n_actions = 2
        self.action_space = Discrete(self.n_actions)

        self._start_state = 0
        # self._end_states = [3, 6]
        self._end_states = [3, 5]

        # self.state_transition_matrix = [[1,4],[2,2],[3,3],[3,3],[5,5],[6,6],[6,6]]
        # self.reward_mapping = [[1,0],[0,0],[0,0],[0,0],[0,0],[2,2],[0,0]]
        self.state_transition_matrix = [[1, 1], [2, 4], [3, 3], [4, 4], [5, 5], [6, 6]]
        self.reward_mapping = [[1, -1], [-2, 2], [100, 100], [0, 0], [-100, -100], [0, 0]]

        self.action_space = Discrete(2)
        self.observation_space = torch.ones((self.n_observations,))
        self.max_steps = 7 #USELESS


        # set environment state
        self.reset()

    def step(self, action: int):
        # print(self._state)
        action = action.nonzero()[0]
        cur_state = self._state.numpy()[0]
        next_state = self.state_transition_matrix[cur_state][action]
        reward = self.reward_mapping[cur_state][action]

        self._reward = reward
        self._done = next_state in self._end_states
        self._state = torch.tensor([next_state], requires_grad=False)
        self._timestep += 1

        return self._state, self._reward, torch.ones(self.action_space.n), self._done, {'No INFO implemented yet'}

    def reset(self):
        self._reward = 0
        self._action = None
        self._timestep = 0
        self._done = False
        self._state = torch.tensor([self._start_state])
        return self._state, torch.ones(self.action_space.n)

    def render(self):
        pass
