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
    def __init__(self, do_heatmap=True):
        # customizable parameters
        self._start_state = 0
        self._end_states = [2]

        self.state_transition_matrix = torch.tensor([[1,2],[2,2]])
        self.reward_mapping = torch.tensor([[0,0],[0,1]])

        # Debug
        self.do_heatmap = do_heatmap

        # set environment state
        self.reset()

    def step(self, action: int):
        next_state = state_transition_matrix[self._state]
        reward = reward_mapping[next_state]

        self._reward = reward
        self._done = next_state in self._end_states
        self._timestep += 1

        state_obj = {
            'observation': torch.tensor([next_state], requires_grad=False),  # pylint: disable=not-callable
            'reward': reward,
            'done': self._done
        }
        self._state = state_obj

        if(self.do_heatmap):
            self.update_heatmap(state_obj)

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

        if(self.do_heatmap):
            self.heatmap = np.zeros(self._grid_dims)

    @property
    def state(self):
        return self._state

    @property
    def state_space(self):
        return Discrete(3)

    @property
    def observation_space(self):
        return Discrete(3)

    @property
    def action_space(self):
        return Discrete(2)

    def update_heatmap(self, state):
        x, y = self.get_x_y(state)
        if(x == 2 and y == 3):
            print(state)
        self.heatmap[y,x] += 1

    def render_heatmap(self):
        print("----- HEAT MAP -----")
        # normalized_heatmap = (self.heatmap - np.min(self.heatmap)) / (np.max(self.heatmap) - np.min(self.heatmap))
        print(self.heatmap / self.heatmap.sum())
        print("---------")
        # plt.imsave('heatmap.png', normalized_heatmap)
        # self.ax = sns.heatmap(self.heatmap, linewidth=0.5)
        # plt.show(block=False)
        # print("render")

    def render(self):
        pass