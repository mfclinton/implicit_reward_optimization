import gym
import numpy as np
from gym import error, spaces, utils
from gym.spaces import Discrete


class Gridworld_687(object):
    def __init__(self, max_steps=200, action_prob=0.8, debug=True):
        self.debug = debug
        self.action_prob = action_prob
        self.n_actions = 4
        self.n_observations = 25
        self.action_space = Discrete(self.n_actions)
        self.observation_space = Discrete(25)

        self.max_steps = max_steps
        self.step_unit = 1
        self.repeat = 1
        
        # Gridworld Specific
        self.width = np.sqrt(self.n_observations)

        if debug:
            self.heatmap = np.zeros((width, width))

        self.reset()

    def seed(self, seed):
        self.seed = seed

    def get_embeddings(self):
        pass # TODO

    def render(self):
        pass

    def set_rewards(self):
        pass

    def reset(self):
        self.set_rewards()
        self.steps_taken = 0
        pass

    def get_valid_actions(self):
        pass # can remove

    def step(self, action):
        assert self.valid_actions[action]

        self.steps_taken += 1
        reward = 0

        term = self.is_terminal()
        if term:
            return self.curr_state, 0, self.valid_actions, term, {'No INFO implemented yet'}

        reward += self.step_reward



    # reward values associated with states
    def set_rewards(self):
        # All rewards
        self.G1_reward = -10 #100
        self.G2_reward = +10 #- 5

    def make_state(self):
        x, y = self.curr_pos
        state = [x, y]

        # TODO

        return state

    def get_goal_rewards(self, pos):
        for key, val in self.reward_states.items():
            region, reward = val
            if self.debug: print("Got reward {} in {} steps!! ".format(reward, self.steps_taken))
            return reward
        return 0

    # States where reward exist
    def get_reward_states(self):
        self.G1 = (2,4,2,4)
        self.G2 = (4,4,4,4)
        return {"G1": (self.G1, self.G1_reward),
                "G2": (self.G2, self.G2_reward)}

    # returns the list of obstacles
    def get_static_obstacles(self):
        self.O1 = (2,2,2,2)
        self.O2 = (2,3,2,3)
        obstacles = [self.O1, self.O2]
        
        return obstacles

    # Returns whether a position is valid
    def valid_pos(self, pos):
        flag = True

        # Check position within boundaries
        if not self.in_region(pos, [0,0,self.width - 1,self.width - 1]):
            flag = False

        # Checks if position is in obstacle
        for region in self.static_obstacles:
            if self.in_region(pos, region):
                flag = False
                break

        return flag

    # Pos: (x,y)
    # Region ((x1,y1), (x2,y2))
    def in_region(self, pos, region):
        x0, y0 = pos
        x1, y1, x2, y2 = region
        if x0 >= x1 and x0 <= x2 and y0 >= y1 and y0 <= y2:
            return True
        else:
            return False

    def is_terminal(self):
        if self.in_region(self.curr_pos, self.G1):
            return True
        elif self.steps_taken >= self.max_steps:
            return True
        else:
            return False