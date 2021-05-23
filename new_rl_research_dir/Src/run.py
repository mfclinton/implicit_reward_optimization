#!~miniconda3/envs/rl/bin python
from logging import config
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from time import time
import torch
import numpy as np
from Src.Utils.utils import TrajectoryBuffer, DataManager
import sys
import random
import logging

class Config:
    def __init__(self,
    env, 
    basis,
    agent, 
    reward_func, 
    gamma_func,
    seed,
    num_runs,
    name="default",
    offpolicy=False, 
    max_episodes=100,
    log_path=r"/media/mfclinton/647875097874DAEE/Users/mfcli/Documents/School/S21/Research/rl_research/new_rl_research_dir/logs",
    restore=False,
    method="file",
    buffer_size=10000,
    batch_size=10):
        self.env = env
        # print(env)
        self.basis = basis
        self.agent = agent
        self.reward_func = reward_func
        self.gamma_func = gamma_func
        self.seed = seed
        self.name = name
        self.offpolicy = offpolicy
        self.max_episodes = max_episodes
        self.buffer_size = buffer_size
        self.batch_size = batch_size


class Solver:
    def __init__(self, nonloaded_config):
        
        # Initializes Everything By Passing Config If Possible
        self.config = instantiate(nonloaded_config.config)
        self.init()
    
    # Initializes all classes in config
    def init(self):
        # Dynamic Initialization
        for attr, obj in vars(self.config).items():
            try:
                print(f"Initializing {attr}")
                obj.init(self.config)
            except Exception as err:
                print(err)
                pass
        
        # Constant Initialization
        env = self.config.env
        state_dim = self.config.env.observation_space.shape[0]
        action_dim = self.config.env.action_space.n

        self.memory = TrajectoryBuffer(self.config.buffer_size, state_dim, action_dim, self.config)


    # Resets everything and return state and valid actions
    def reset(self):
        # Dynamic Resetting
        for attr, obj in vars(self.config).items():
            try:
                result = obj.reset()
                if attr == "env":
                    state, valid_actions = result
            except: pass

        # Constant Resetting
        self.memory.next() #TODO: check if this is resetting correctly

        return state, valid_actions
    
    def train(self, data_mngr):
        agent = self.config.agent
        env = self.config.env
        basis = self.config.basis

        start_ep = 0

        steps = 0
        t0 = time()
        for episode in range(start_ep, self.config.max_episodes):

            state, valid_actions = self.reset() #TODO: FIX RESETTING

            step, total_r = 0, 0
            done = False
            while not done:
                state_feature = torch.tensor(state, requires_grad=False)
                action, prob, dist = agent.policy.get_action_w_prob_dist(basis.forward(state_feature.view(1, -1)))
                
                new_state, reward, valid_actions, done, info = env.step(action=action)
                
                self.memory.add(state, action, prob, reward)      
                state = new_state
                total_r += reward

                step += 1

                # env.render()

            # Optimize Agent
            # batch_size = self.memory.size if self.memory.size < self.config.batch_size else self.config.batch_size

            if self.config.batch_size <= self.memory.episode_ctr:
                ids, s, a, prob, r, mask = self.memory.sample(self.config.batch_size)
                # print(s.size())
                B, H, D = s.shape
                _, _, A = a.shape

                s_features = basis.forward(s.view(B * H, D))
                # print(s_features.size(), mask.size())
                s_features *= mask.view(B*H, 1) #TODO: Check this

                agent.optimize(s_features, a, r)


                if not self.config.offpolicy:
                    self.memory.reset()

            data_mngr.update_rewards(total_r)
            steps += step


            if episode == self.config.max_episodes - 1:
                # TODO
                # append returns
                data_mngr.update_returns() #TODO: check if this is what I want
                # save model and plots

                t0 = time()
                # steps = 0
            
            # print("Avg Reward ", total_r / step)
            log.info(f"Episode: {episode} | Total Reward: {total_r} | Length: {step} | Avg Reward: {total_r / step}")
        # self.data_mngr.save()
            
log = logging.getLogger(__name__)
    
# @hydra.main(config_path=".", config_name="config")
@hydra.main(config_path=".", config_name="config_GW")
def main(nonloaded_config : DictConfig) -> None:

    # Set Seed
    # TODO: Propagate seeding in init functions, so can seed in config
    torch.manual_seed(nonloaded_config.config.seed)
    np.random.seed(nonloaded_config.config.seed)
    random.seed(nonloaded_config.config.seed)


    t = time()
    data_mngr = DataManager()

    for i in range(nonloaded_config.config.num_runs):
        solver = Solver(nonloaded_config)
        solver.train(data_mngr)

    data_mngr.save()
    with open("config_params", "w") as f:
        f.write(str(nonloaded_config))
    log.info("Total time taken: {}".format(time()-t))


if __name__ == "__main__":
    main()