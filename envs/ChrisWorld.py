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

class ChrisWorld(gym.Env):
    def __init__(self):
        # customizable parameters
        self._start_state = 0
        self._end_states = [3, 6]

        self.state_transition_matrix = [[1,4],[2,2],[3,3],[3,3],[5,5],[6,6],[6,6]]
        self.reward_mapping = [[1,0],[0,0],[0,0],[0,0],[0,0],[2,2],[0,0]]

        # set environment state
        self.reset()

    def step(self, action: int):
        # print(self._state)
        cur_state = self._state['observation'].numpy()[0]
        next_state = self.state_transition_matrix[cur_state][action]
        reward = self.reward_mapping[cur_state][action]

        self._reward = reward
        self._done = next_state in self._end_states
        self._timestep += 1

        state_obj = {
            'observation': torch.tensor([next_state], requires_grad=False),  # pylint: disable=not-callable
            'reward': reward,
            'done': self._done
        }
        self._state = state_obj

        return state_obj

    def reset(self):
        self._state = {
            'observation': torch.tensor([self._start_state]),  # pylint: disable=not-callable
            'reward': 0,
            'done': False
        }
        self._reward = 0
        self._action = None
        self._timestep = 0
        self._done = False

    @property
    def state(self):
        return self._state

    @property
    def state_space(self):
        return Discrete(7)

    @property
    def observation_space(self):
        return Discrete(7)

    @property
    def action_space(self):
        return Discrete(7)

    def render(self):
        pass