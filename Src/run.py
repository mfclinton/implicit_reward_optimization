#!~miniconda3/envs/rl/bin python
from logging import config
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from time import time
import torch
import numpy as np
from Src.Utils.utils import TrajectoryBuffer, DataManager
from Src.Algorithms.Alg_Utils import *
import sys
import random
import logging
from hydra.experimental import compose, initialize
# from multiprocessing import Pool as ThreadPool

class Config:
    def __init__(self,
    env, 
    basis,
    agent, 
    reward_func, 
    gamma_func,
    gamma=0.99,
    name="default",
    offpolicy=False,
    T1=10,
    T2=100,
    T3=20,
    log_path=r"/media/mfclinton/647875097874DAEE/Users/mfcli/Documents/School/S21/Research/rl_research/new_rl_research_dir/logs",
    restore=False,
    method="file",
    buffer_size=10000,
    batch_size=10, #Not Used Parameter
    weight_decay=0.0,
    dropped_gamma=False,
    alpha=0.9,
    num_repeat_action=1):
        self.env = env
        # print(env)
        self.basis = basis
        self.agent = agent
        self.reward_func = reward_func
        self.gamma_func = gamma_func
        self.gamma = gamma
        self.name = name
        self.offpolicy = offpolicy
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.weight_decay = weight_decay #TODO: Check weight decay
        self.dropped_gamma = dropped_gamma
        self.alpha = alpha
        self.num_repeat_action = num_repeat_action

class Solver:
    def __init__(self, nonloaded_config):
        
        # Initializes Everything By Passing Config If Possible
        self.config = instantiate(nonloaded_config)
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
        # print(self.config.env)
        # print(self.config.env.observation_space)
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

    def generate_episode(self):
        env = self.config.env
        basis = self.config.basis
        agent = self.config.agent

        # state, valid_actions = self.reset() #TODO: FIX RESETTING
        self.memory.next() #TODO: check if correct
        state, valid_actions = env.reset() #TODO: FIX RESETTING

        step, total_r = 0, 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, requires_grad=False)
            # print(basis.forward(state_tensor.view(1, -1)))
            action, prob, dist = agent.policy.get_action_w_prob_dist(basis.forward(state_tensor.view(1, -1)))
            # print("WOW")
            # TODO: Fix valid actions
            for i in range(self.config.num_repeat_action):
                new_state, reward, valid_actions, done, info = env.step(action=action)
            # new_state, reward, valid_actions, done, info = env.step(action=action)
            
            self.memory.add(state, action, prob, reward)      
            state = new_state
            total_r += reward

            step += 1
            # env.render()

        return total_r, step
    
    def train(self, data_mngr, log):
        env = self.config.env
        basis = self.config.basis

        agent = self.config.agent
        reward_func = self.config.reward_func
        gamma_func = self.config.gamma_func

        alpha = self.config.alpha

        # Do Running Average
        H_value = 0
        B_value = 0
        A_value = 0

        # Ensures update in t1 loop
        # assert self.config.offpolicy or self.config.batch_size <= self.config.T2
        
        for t1 in range(self.config.T1):

            # Checkpoints model
            # save_interval = (self.config.T1 - 1) // 3
            # if save_interval == 0 or t1 % save_interval == 0:
                # data_mngr.save_reward_and_gamma_func(reward_func, gamma_func, t1)

            # agent.init(self.config)
            if not self.config.offpolicy:
                self.memory.reset()

            for t2 in range(self.config.T2):
                # Get Trajectory
                # TODO: Check it's okay to generate an initial episode
                if not self.config.offpolicy or self.memory.episode_ctr < self.config.T3:
                    total_r, step = self.generate_episode()
                    sample = self.memory._get([self.memory.buffer_pos])
                    # data_mngr.update_rewards(total_r)
                else:
                    sample = self.memory.sample(1)

                # Optimize Agent
                # batch_size = self.memory.size if self.memory.size < self.config.batch_size else self.config.batch_size
                ids, s, a, _, r, mask = sample
                B, H, D = s.shape
                _, _, A = a.shape

                s_features, _, in_r, in_g = Process_Sample(sample, basis, agent, reward_func, gamma_func)

                agent.optimize(s_features, a, in_r.detach() + r, in_g.detach()) # TODO: Add back r

                total_r = r.sum()
                total_in_r = in_r.sum().detach()
                # log.info(f"T1: {t1} | T2: {t2} | Total Reward: {total_r} | Avg Reward: {total_r / step}")
                # log.info(f"T1: {t1} | T2: {t2} | Total Internal Reward: {total_in_r} | Length: {step} | Avg Internal Reward: {total_in_r / step}")
                # data_mngr.update_internal_rewards(total_in_r)

            # TODO: Make different batch sizes
            sample = self.memory.sample(self.config.batch_size, replace=False)
            _, s, a, _, r, mask = sample
            
            B, H, D = s.shape
            _, _, A = a.shape

            s_features, log_pi, in_r, in_g = Process_Sample(sample, basis, agent, reward_func, gamma_func)
            
            new_H_value = 0
            new_B_value = 0
            new_A_value = 0

            for b in range(B):
                cumu_in_g = Get_Cumulative_Gamma(in_g[b]).detach() * mask[b]
                disc_in_r = Get_Discounted_Returns(in_r[b], cumu_in_g, normalize=False).detach()

                # TODO: CHECK GRADIENTS
                phi = calc_grads(agent.policy, log_pi[b], True).detach()
                d_in_r = calc_grads(reward_func, in_r[b], True).detach()
                d_in_g = calc_grads(gamma_func, in_g[b], True).detach()

                new_H_value += Approximate_H(phi, disc_in_r, self.config.weight_decay).detach()
                new_B_value += Calculate_B(phi, d_in_g, in_r[b]).detach()
                new_A_value += Approximate_A(phi, cumu_in_g, d_in_r).detach()
            
            # Average
            new_H_value /= B
            new_B_value /= B
            new_A_value /= B

            # Update of H B A
            H_value = alpha * H_value + (1-alpha) * new_H_value
            B_value = alpha * B_value + (1-alpha) * new_B_value
            A_value = alpha * A_value + (1-alpha) * new_A_value

            # env.heatmap = np.zeros((env.width, env.width)) # TODO REMOVE THIS
            c_value = 0
            for t3 in range(self.config.T3):
                total_r, step = self.generate_episode()
                sample = self.memory._get([self.memory.buffer_pos])
                _, s, a, _, r, mask = sample

                B, H, D = s.shape #Note: B = 1
                _, _, A = a.shape

                s_features, log_pi, in_r, in_g = Process_Sample(sample, basis, agent, reward_func, gamma_func)

                log_pi = log_pi.squeeze()
                in_r = in_r.squeeze() #Not Used
                in_g = in_g.squeeze() #Not Used
                mask = mask.squeeze()

                gamma = torch.full((H,), self.config.gamma).detach()
                gamma *= mask
                cumu_gamma = Get_Cumulative_Gamma(gamma).detach() * mask

                phi = calc_grads(agent.policy, log_pi, True).detach()

                disc_r = Get_Discounted_Returns(r, cumu_gamma, normalize=False).detach()
                c_value += Calculate_C(phi, cumu_gamma, disc_r).detach()

                data_mngr.update_rewards(total_r)
                # print(B_value, H_value, A_value)
                total_in_r = in_r.sum().detach()
                log.info(f"T1: {t1} | T3: {t3} | Total Reward: {total_r} | Length: {step} | Avg Reward: {total_r / step}")
                log.info(f"T1: {t1} | T3: {t3} | Total Internal Reward: {total_in_r} | Length: {step} | Avg Internal Reward: {total_in_r / step}")

            # Average Results Together
            c_value /= self.config.T3

            nonzero_idx = (H_value == 0).nonzero()
            H_value[nonzero_idx] += 1

            d_reward_func = - c_value * (A_value.squeeze() / H_value.squeeze())
            d_gamma_func = - c_value * (B_value.squeeze() / H_value.squeeze())

            reward_func.optim.zero_grad()
            gamma_func.optim.zero_grad()
            

            # TODO: Make sure right shape
            reward_func.fc1.weight.grad = d_reward_func.view(reward_func.fc1.weight.shape).detach()
            gamma_func.fc1.weight.grad = d_gamma_func.view(gamma_func.fc1.weight.shape).detach()

            reward_func.step(normalize_grad=True)
            gamma_func.step()

            # TODO: REMOVE
            # env.debug_rewards(gamma_func, basis, print_r_map=True, is_gamma=True)

            if t1 == self.config.T1 - 1:
                data_mngr.update_returns()
                data_mngr.save_model(reward_func, "Reward_Model_1")
                data_mngr.save_model(gamma_func, "Gamma_Model_1")
            elif (t1 == 2 * self.config.T1 // 3) or (t1 == self.config.T1 // 3) or (t1 == 0):
                data_mngr.save_model(reward_func, f"Reward_Model_{t1 / self.config.T1}")
                data_mngr.save_model(gamma_func, f"Gamma_Model_{t1 / self.config.T1}")

                # data_mngr.update_internal_returns()

            
def run_thread(nonloaded_config, seed):
    log = logging.getLogger(__name__)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    t = time()
    data_mngr = DataManager(save_just_data = True)
    solver = Solver(nonloaded_config.config)
    solver.train(data_mngr, log)
    # data_mngr.save()

    with open("config_params", "w") as f:
        f.write(str(nonloaded_config))

    log.info("Total time taken: {}".format(time()-t))
    return data_mngr.total_r

@hydra.main(config_path=".", config_name="config")
# @hydra.main(config_path=".", config_name="config_GW")
def main(nonloaded_config : DictConfig) -> None:

    # Set Seed
    seed = nonloaded_config.seed
    total_score = 0
    for i in range(nonloaded_config.num_runs):
        total_score += run_thread(nonloaded_config, seed + 10000 * i) #TODO: use number of runs param
    return total_score

if __name__ == "__main__":
    main()