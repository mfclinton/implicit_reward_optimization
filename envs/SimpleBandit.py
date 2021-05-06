import torch
import gym
from gym.spaces import Discrete

# Taken From https://github.com/ahujaavi13/implicitly-learned-rewards/blob/master/envs/SimpleBandit.py

class SimpleBandit(gym.Env):
    def __init__(self, num_states=4, num_actions=4, max_trajectory_len=500):
        self.num_states = num_states
        self.num_actions = num_actions
        self.start_state = 0
        self.max_trajectory_len = max_trajectory_len
        self.end_states = (self.num_states - 1,)
        self.state_transition = torch.vstack(self.num_actions * [torch.arange(start=1, end=self.num_states)]).T
        self.rewards = torch.full((self.num_states, self.num_actions), fill_value=0.)

        self.env_state = {
            'observation': torch.tensor([self.start_state]),
            'reward': 0.,
            'done': False
        }

        self.reward = 0.
        self.action = None
        self.timestep = 0
        self.done = False

    def step(self, action):
        # print(self.state_transition)
        # 1/0
        cur_state = self.state['observation'].numpy()[0]
        next_state = self.state_transition[cur_state, action]
        # print(next_state)
        reward = 1 if action == 1 else self.rewards[cur_state, action]

        self.timestep += 1
        self.reward = reward if self.timestep < self.max_trajectory_len else -100
        self.done = next_state in self.end_states

        self.env_state = {
            'observation': torch.tensor([next_state]),
            'reward': self.reward,
            'done': self.done
        }

        return self.env_state

    def reset(self):
        self.env_state = {
            'observation': torch.tensor([self.start_state]),
            'reward': 0.,
            'done': False
        }
        self.reward = 0.
        self.action = None
        self.timestep = 0
        self.done = False

        return None

    def get_env_state(self):
        return self.env_state.values()

    @property
    def state(self):
        return self.env_state

    @property
    def state_space(self):
        return Discrete(self.num_states)

    @property
    def observation_space(self):
        return Discrete(self.num_states)

    @property
    def action_space(self):
        return Discrete(self.num_actions)

    def render(self, mode='human'):
        pass
