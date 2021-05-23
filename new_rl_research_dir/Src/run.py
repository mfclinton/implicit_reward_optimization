#!~miniconda3/envs/rl/bin python
from logging import config
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from time import time
import torch
import numpy as np
from Src.Utils.utils import Logger
import sys

class Config:
    def __init__(self, 
    env, 
    agent, 
    reward_func, 
    gamma_func, 
    offpolicy=False, 
    max_episodes=100,
    log_path=r"/media/mfclinton/647875097874DAEE/Users/mfcli/Documents/School/S21/Research/rl_research/new_rl_research_dir/logs",
    restore=False,
    method="file",
    seed=0):
        self.env = env
        self.agent = agent
        self.reward_func = reward_func
        self.gamma_func = gamma_func
        self.offpolicy = offpolicy
        self.max_episodes = max_episodes
        sys.stdout = Logger(log_path, restore, method)
        self.seed = seed


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
                # print(f"Initializing {attr}")
                obj.init(self.config)
            except Exception as err:
                # print(err)
                pass

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
        agent = self.config.agent
        env = self.config.env
        torch.manual_seed(self.config.seed) # TODO: Fix
        np.random.seed(self.config.seed)


        returns = []
        rewards = []

        start_ep = 0

        steps = 0
        t0 = time()
        for episode in range(start_ep, self.config.max_episodes):
            # if episode % 100 == 0:
            #     print("Episode : ", episode)

            state, valid_actions = self.reset()
            # print(state, "sassadsad")

            step, total_r = 0, 0
            done = False
            while not done:
                # get action
                action, extra_info, dist = agent.get_action(state)
                
                # take step
                new_state, reward, valid_actions, done, info = env.step(action=action)
                
                # update agent
                agent.update(state, action, extra_info, reward, new_state, valid_actions, done)
                

                # TODO
                # update state
                state = new_state

                # intra episode stats
                # add r
                total_r += reward
                step += 1
                # env.render()

            steps += step

            if episode == self.config.max_episodes - 1:
                # TODO
                # append returns
                # save model and plots

                t0 = time()
                steps = 0
            
            # print("Avg Reward ", total_r / step)
            print(f"Episode: {episode} | Total Reward: {total_r} | Length: {step} | Avg Reward: {total_r / step}")
            
            
        

@hydra.main(config_path=".", config_name="config")
# @hydra.main(config_path=".", config_name="config_GW")
def main(nonloaded_config : DictConfig) -> None:
    t = time()
    solver = Solver(nonloaded_config)
    solver.train()
    print("Total time taken: {}".format(time()-t))


if __name__ == "__main__":
    main()