#!~miniconda3/envs/rl/bin python
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from time import time
import torch
import numpy as np

class Config:
    def __init__(self, 
    env, 
    agent, 
    reward_func, 
    gamma_func, 
    offpolicy=False, 
    max_episodes=100):
        self.env = env
        self.agent = agent
        self.reward_func = reward_func
        self.gamma_func = gamma_func
        self.offpolicy = offpolicy
        self.max_episodes = max_episodes


class Solver:
    def __init__(self, nonloaded_config):
        
        # Initializes Everything By Passing Config If Possible
        self.config = instantiate(nonloaded_config.config)
        self.init()

        # self.state_dim = np.shape(self.config.env.reset()[0])[0]
        # if len(self.config.env.action_space.shape) > 0:
        #     self.action_dim = self.config.env.action_space.shape[0]
        # else:
        #     self.action_dim = self.config.env.action_space.n
        
        # print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))
    
    # Initializes all classes in config
    def init(self):
        for attr, obj in vars(self.config).items():
            try:
                print(f"Initializing {attr}")
                obj.init(self.config)
            except: pass

    # Resets everything and return state and valid actions
    def reset(self):
        for attr, obj in vars(self.config).items():
            try:
                result = obj.reset()
                if attr == "env":
                    state, valid_actions = result
            except: pass
        return state, valid_actions
    
    def train(self):
        returns = []
        rewards = []

        start_ep = 0

        steps = 0
        t0 = time()
        for episode in range(start_ep, self.config.max_episodes):
            state, valid_actions = self.reset()

            step = 0
            done = False
            while not done:
                # get action
                # take step
                # update agent
                # update state

                # intra episode stats
                # add r
                step += 1
                # break if exceed max steps, check env
            
            steps += step

            if episode == self.config.max_episodes - 1:
                # TODO
                # append returns
                # save model and plots

                t0 = time()
                steps = 0
            
            
        

@hydra.main(config_path=".", config_name="config")
def main(nonloaded_config : DictConfig) -> None:
    t = time()
    solver = Solver(nonloaded_config)
    solver.train()
    print("Total time taken: {}".format(time()-t))


if __name__ == "__main__":
    main()