import gym
from gym import error, spaces, utils
from gym.spaces import Discrete
from gym.utils import seeding
import logging
import torch
import numpy as np
import math

class GridWorld(gym.Env):
    def __init__(self, grid_dims = (5, 5), start_state = 0, end_states = [24], obstacle_states = [12, 17], water_states = [22]):
        # customizable parameters
        self._grid_dims = grid_dims
        self._start_state = start_state
        self._end_states = end_states
        self._obstacle_states = obstacle_states
        self._water_states = water_states

        # stochasticity
        self._prStay = 0.1
        self._prRotate = 0.05

        # dicts mapping actions to the appropriate rotations
        self._rotateLeft = {0: 2, 1: 3, 2: 1, 3: 0}
        self._rotateRight = {0: 3, 1: 2, 2: 0, 3: 1}

        # set environment state
        self.reset()

    def step(self, action: int):
        next_state = self._calc_next_state(self._state['observation'].numpy()[0], action)
        reward = self._calc_reward(next_state)

        self._reward = reward
        self._done = next_state in self._end_states
        self._timestep += 1

        state_obj = {
            'observation': torch.tensor([next_state]),  # pylint: disable=not-callable
            'reward': reward,
            'done': self._done
        }
        self._state = state_obj

        return state_obj

    def generate_trajectory(self):
        pass #TODO


    def _calc_next_state(self, state: float, action: int) -> float:
        if state in self._end_states:
            return state

        noise = np.random.uniform()
        if torch.is_tensor(action):
            action = action.item()
        if noise < self._prStay:  # do nothing
            return state
        elif noise < (self._prStay + self._prRotate):
            action = self._rotateLeft[action]
        elif noise < (self._prStay + 2 * self._prRotate):
            action = self._rotateRight[action]

        # simulate taking a step in the environment
        nextState = state
        if action == 0:  # move up
            nextState = state - self._grid_dims[1]
        elif action == 1:  # move down
            nextState = state + self._grid_dims[1]
        elif action == 2 and (nextState % self._grid_dims[1] != 0):  # move left
            nextState = state - 1
        elif action == 3 and ((nextState + 1) % self._grid_dims[1] != 0):  # move right
            nextState = state + 1

        # check if the next state is valid and not an obstacle
        size = self._grid_dims[0] * self._grid_dims[1]
        if nextState >= 0 and nextState < size and nextState not in self._obstacle_states:
            return nextState
        else:
            return state

    def _calc_reward(self, state: int):
        # TODO: Gamma?
        if state in self._water_states:
            return -10.0
        elif state in self._end_states:
            return 10.0
        else:
            return 0.0

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
        return Discrete(self._grid_dims[0] * self._grid_dims[1])

    @property
    def observation_space(self):
        return Discrete(self._grid_dims[0] * self._grid_dims[1])

    @property
    def action_space(self):
        return Discrete(4)