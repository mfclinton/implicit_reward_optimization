import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.Gridworld import GridWorld
from envs.ChrisWorld import ChrisWorld
from envs.SimpleBandit import SimpleBandit

from models.models import REINFORCE, INTRINSIC_REWARD, INTRINSIC_GAMMA, CHRIS_REINFORCE
import torch.nn.functional as F
import torch
import torch.nn.utils as utils
from torch.distributions import Categorical
from utils.helper import *
from utils.new_helper import *
import pandas as pd
# Memory Leak
import gc
import random
from pathlib import Path

import matplotlib.pyplot as plt
from new_learned_reward import Get_Prior_Reward

# Updates Our Trajectory Information Given A One-Step Data Sample
def update_and_get_action(env, agent, data, states, actions, rewards, log_probs):
    # print(data["observation"], env.state_space.n)
    # print("LOL")
    state = onehot_state(data["observation"], env.state_space.n)
    action, log_prob = agent.select_action(state)

    # Updates our trajectory lists
    states.append(state) # (S,)
    actions.append(action) # (1)
    rewards.append(data["reward"]) # (1)
    log_probs.append(log_prob) # () TODO

    return action

def Get_Trajectory(env, agent):
    #Resets The Environment
    env.reset()

    # Trajectory Data
    states = [] # (S,)
    actions = [] # (1)
    rewards = [] # (1)
    log_probs = [] # () TODO

    #Gets the starting state
    data = env.state 
    action = update_and_get_action(env, agent, data, states, actions, rewards, log_probs) #Updates trajectory with initial state, gets first action

    while not data["done"]:
        data = env.step(action) #take action
        action = update_and_get_action(env, agent, data, states, actions, rewards, log_probs) #updates trajectory with new info
    
    return states, torch.tensor(actions), torch.tensor(rewards), log_probs


def Run_Gridworld_Implicit(T1): 
    env = GridWorld(obstacle_states = []) # Creates Environment
    # env = SimpleBandit()
    agent = REINFORCE(env.state_space.n, env.action_space) #Create Policy Function, (S) --> (25) --> (A)
    in_gamma = INTRINSIC_GAMMA(env.state_space.n * env.action_space.n) #Creates Intrinsic Gamma (S) --> (25) --> (1)

    prior_id = 3
    aux = Get_Prior_Reward(env, prior_id)

    # DEBUG
    agent.reset() ### Might not be necessary, can slow down learning

    actual_reward_over_time = []
    for t1 in range(T1):
        print("episode ", t1)
        
        states, actions, real_rewards, log_probs = Get_Trajectory(env, agent)
        actual_reward_over_time.append(np.array(real_rewards).sum())
        
        # Creates a matrix of size (time_steps, S)
        states_matrix = torch.stack(states, dim=0)
        if prior_id != -1:
            s_idx = torch.nonzero(states_matrix)[:,1]
            aux_rewards = aux[actions * (env._grid_dims[0] * env._grid_dims[1]) + s_idx]
            real_rewards += aux_rewards
        # print(real_rewards)

        # Creates a State Action matrix of size (time_steps, S * A)
        
        state_actions = onehot_states_to_state_action(states_matrix, actions, env.action_space.n)
        gammas = in_gamma.get_gamma(state_actions).squeeze() #(time_steps, S) --> (time_steps, 1)
        gammas = torch.full_like(gammas, 0.9)
        # print(gammas)

        # cumu_gammas = get_cumulative_multiply_front(in_gammas)
        cumu_gammas = Get_Cumulative_Gamma(gammas)
        
        agent.update_parameters(real_rewards, log_probs, cumu_gammas)
        


    s = pd.Series(actual_reward_over_time)
    plt.plot(s.rolling(5).mean())
    plt.ylabel("avg reward")

    plt.show()

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    np.random.seed(0)
    Run_Gridworld_Implicit(15 * 50)