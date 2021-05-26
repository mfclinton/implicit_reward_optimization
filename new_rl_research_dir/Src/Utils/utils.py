import torch
import numpy as np
from scipy import stats
import sys
import os
from os import path, mkdir, listdir, fsync
from time import time
import matplotlib.pyplot as plt
import pandas as pd


# TODO
class TrajectoryBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, config):
        max_horizon = config.env.max_steps

        self.s = torch.zeros((buffer_size, max_horizon, state_dim), requires_grad=False)
        self.a = torch.zeros((buffer_size, max_horizon, action_dim), requires_grad=False)
        self.p = torch.zeros((buffer_size, max_horizon), requires_grad=False)
        self.r = torch.zeros((buffer_size, max_horizon), requires_grad=False)
        self.mask = torch.zeros((buffer_size, max_horizon), requires_grad=False)
        self.ids = torch.zeros(buffer_size, requires_grad=False)

        self.buffer_size = buffer_size
        self.episode_ctr = -1
        self.timestep_ctr = 0
        self.buffer_pos = -1
        self.valid_len = 0

    @property
    def size(self):
        return self.valid_len

    def reset(self):
        self.episode_ctr = -1
        self.timestep_ctr = 0
        self.buffer_pos = -1
        self.valid_len = 0
        # print("MEMORY RESET")

    def next(self):
        self.episode_ctr += 1
        self.buffer_pos += 1

        # Cycle around to the start of buffer (FIFO)
        if self.buffer_pos >= self.buffer_size:
            self.buffer_pos = 0

        if self.valid_len < self.buffer_size:
            self.valid_len += 1

        self.timestep_ctr = 0
        self.ids[self.buffer_pos] = self.episode_ctr

        self.s[self.buffer_pos].fill_(0)
        self.a[self.buffer_pos].fill_(0)
        self.p[self.buffer_pos].fill_(0)
        self.r[self.buffer_pos].fill_(0)
        self.mask[self.buffer_pos].fill_(0)

    def add(self, s1, a1, p1, r1):
        pos = self.buffer_pos
        step = self.timestep_ctr
        # print(f"Step {step} and Pos {pos}")

        self.s[pos][step] = torch.tensor(s1)
        self.a[pos][step] = torch.tensor(a1) #TODO: TEMP SOLUTION, might want to .copy()
        self.p[pos][step] = torch.tensor(p1)
        self.r[pos][step] = torch.tensor(r1)
        self.mask[pos][step] = torch.tensor(1.0)

        # print(self.a[pos][step])

        self.timestep_ctr += 1

    def _get(self, idx):
        # ids represent the episode number
        # idx represents the buffer index
        # Both are not the same due to use of wrap around buffer
        ids = self.ids[idx]

        if self.valid_len >= self.buffer_size:
            # Subtract off the minimum value idx (as the idx has wrapped around in buffer)
            if self.buffer_pos + 1 == self.buffer_size:
                ids -= self.ids[0]
            else:
                ids -= self.ids[self.buffer_pos + 1]

        # return self.s[idx], self.a[idx], self.p[idx], self.r[idx] #TODO
        return ids, self.s[idx], self.a[idx], self.p[idx], self.r[idx], self.mask[idx]

    def sample(self, batch_size, replace=True):
        count = min(batch_size, self.valid_len)
        return self._get(np.random.choice(self.valid_len, count, replace=replace))

class DataManager:
    def __init__(self):
        self.result_path = os.getcwd()
        # print(self.result_path)
        # if not os.path.exists(self.result_path):
        #     print(f"Creating Results Path at {self.result_path}")
        #     os.makedirs(self.result_path)

        self.reset()

    def reset(self):
        self.rewards = []
        self.internal_rewards = []
        self.returns = []
        self.internal_returns = []

    def update_rewards(self, reward):
        self.rewards.append(reward)

    def update_internal_rewards(self, internal_reward):
        self.internal_rewards.append(internal_reward)

    def update_returns(self):
        self.returns.append(self.rewards)
        self.rewards = []

    def update_internal_returns(self):
        self.internal_returns.append(self.internal_rewards)
        self.internal_rewards = []

    def process_returns(self, returns):
        np_returns = np.array(returns)

        m = np.mean(np_returns, axis=0)
        se = stats.sem(np_returns, axis=0)

        return m, se

    def save(self):
        m, se = self.process_returns(self.returns)
        self.save_csv(m, se, "r")
        self.save_plot(m, se, "r")
        self.save_rolling_plot(m, se, "r")

        i_m, i_se = self.process_returns(self.internal_returns)
        self.save_csv(i_m, i_se, "in_r")
        self.save_plot(i_m, i_se, "in_r")
        self.save_rolling_plot(i_m, i_se, "in_r")

    def save_csv(self, m, se, name=""):
        df = pd.DataFrame(np.stack((m, se), axis=1), columns=["Mean", "Standard Error"])
        df.to_csv(f"{self.result_path}/{name}_data.csv", index=False)

    def save_plot(self, m, se, name=""):
        fig = plt.figure()
        x = np.arange(m.shape[0])
        plt.errorbar(x, m, se, linestyle='None', marker='^')
        plt.ylabel("Total Reward")
        plt.savefig(f"{self.result_path}/{name}_graph.png")
        # plt.close(fig)


    def save_rolling_plot(self, m, se, name=""):
        fig = plt.figure()
        x = np.arange(m.shape[0])
        m = pd.DataFrame(data=m)
        se = pd.DataFrame(data=se)
        rolling_se = se.rolling(100, min_periods=1).mean().to_numpy().squeeze()
        rolling_m = m.rolling(100, min_periods=1).mean().to_numpy().squeeze()
        print(rolling_m.shape, rolling_se.shape)
        plt.errorbar(x, rolling_m, rolling_se, linestyle='None', marker='^')
        plt.ylabel("Total Reward")
        print(self.result_path)
        plt.savefig(f"{self.result_path}/{name}_rolling_graph.png")
        # plt.close(fig)


    def save_3d_reward_plot(self, m):
        pass



