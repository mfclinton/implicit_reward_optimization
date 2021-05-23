import torch
import numpy as np
import sys
from os import path, mkdir, listdir, fsync
from time import time


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

    def sample(self, batch_size):
        count = min(batch_size, self.valid_len)
        return self._get(np.random.choice(self.valid_len, count))


# From https://github.com/yashchandak/OptFuture_NSMDP/blob/master/Src/Utils/utils.py
class Logger(object):
    fwrite_frequency = 1800  # 30 min * 60 sec
    temp = 0

    def __init__(self, log_path, restore, method):
        self.terminal = sys.stdout
        self.file = 'file' in method
        self.term = 'term' in method

        if self.file:
            if restore:
                self.log = open(path.join(log_path, "logfile.log"), "a")
            else:
                self.log = open(path.join(log_path, "logfile.log"), "w")


    def write(self, message):
        if self.term:
            self.terminal.write(message)

        if self.file:
            self.log.write(message)

            # Save the file frequently
            if (time() - self.temp) > self.fwrite_frequency:
                self.flush()
                self.temp = time()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.

        # Save the contents of the file without closing
        # https://stackoverflow.com/questions/19756329/can-i-save-a-text-file-in-python-without-closing-it
        # WARNING: Time consuming process, Makes the code slow if too many writes
        if self.file:
            self.log.flush()
            fsync(self.log.fileno())

class DataManager:
    def __init__(self):
        pass    

