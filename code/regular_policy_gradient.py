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


def Run_Gridworld_Implicit(T1, T2, T3, approximate, reuse_trajectories):
    Use_Chris_World = False
    Save_Data = False
    prior_id = 1
    
    env = GridWorld() # Creates Environment
    # env = SimpleBandit()
    agent = REINFORCE(env.state_space.n, env.action_space) #Create Policy Function, (S) --> (25) --> (A)

    if Use_Chris_World:
        env = ChrisWorld() #TODO: remove
        agent = CHRIS_REINFORCE() #TODO: remove

    # agent = REINFORCE(env.state_space.n, env.action_space) #Create Policy Function, (S) --> (25) --> (A)
    # in_reward = INTRINSIC_REWARD(env.state_space.n * env.action_space.n, None) #Create Intrinsic Reward Function, (S * A) --> (25) --> (1)
    in_reward = INTRINSIC_REWARD(env.state_space.n * env.action_space.n, Get_Prior_Reward(env, prior_id)) #Create Intrinsic Reward Function, (S * A) --> (25) --> (1)
    in_gamma = INTRINSIC_GAMMA(env.state_space.n * env.action_space.n) #Creates Intrinsic Gamma (S) --> (25) --> (1)
    
    # DEBUG
    actual_reward_over_time = []
    trajectories = []
    for t1 in range(T1):
        # TODO: Can we keep the same agent across iterations?
        # agent = REINFORCE(env.state_space.n, env.action_space)
        agent.reset() ### Might not be necessary, can slow down learning
        for t2 in range(T2):
            # Get Trajectory returns lists of elements with the following dims
            # States, (S,)
            # Actions, (1,)
            # Real_Rewards, (1,)
            # Log_Probs, TODO
            # states, actions, real_rewards, log_probs = Get_Trajectory(env, agent) #Samples a trajectory 
            if not reuse_trajectories:
                states, actions, real_rewards, log_probs = Get_Trajectory(env, agent)
            else:
                # Skip T2 Loop if no Trajectories
                if(len(trajectories) == 0):
                    break ### How to manage first few trajectories , can cause issues with graph representation

                states, actions, real_rewards, log_probs = random.choice(trajectories)
                # states, actions, real_rewards = random.choice(trajectories)

            # Creates a matrix of size (time_steps, S)
            states_matrix = torch.stack(states, dim=0)

            # Gets Log Probs, TODO find out which log probs we are sampled from
            if reuse_trajectories:
                probs = agent.model(states_matrix)
                sampler = Categorical(probs)
                log_probs = sampler.log_prob(actions) ### Store Log Probs or Recalculate

            # Creates a State Action matrix of size (time_steps, S * A)
            state_actions = onehot_states_to_state_action(states_matrix, actions, env.action_space.n)

            # Evaluates intrinsic reward and gamma
            in_rewards = in_reward.get_reward(state_actions).squeeze() #(time_steps, S*A) --> (time_steps, 1)
            in_gammas = in_gamma.get_gamma(state_actions).squeeze() #(time_steps, S) --> (time_steps, 1)
            
            # cumu_gammas = get_cumulative_multiply_front(in_gammas)
            cumu_gammas = Get_Cumulative_Gamma(in_gammas)

            # TODO: Check about ^t for each gamma in the traj
            # only care about from start
            # discounted_in_returns = Get_Discounted_Returns(in_rewards, cumu_gammas, normalize=False)
            
            agent.update_parameters(in_rewards, log_probs, cumu_gammas)
            # agent.update_parameters(torch.tensor(real_rewards), log_probs, cumu_gammas)

        c /= T3
        H /= T3
        A /= T3

        # print(c)
        # print(H)
        # print(A)

        # TODO: Check, Sign of Gradient in Paper Is For Update
        d_in_reward_params = None
        d_in_gamma_params = None
        if(not approximate):
            d_in_reward_params = torch.matmul(c, torch.matmul(torch.inverse(H), A))
        else:
            # print(c.size(), A.size(), H.size())
            # print(c, A, H)
            d_in_reward_params = - c * (A.squeeze() / H.squeeze()) #TODO make negative
            d_in_gamma_params = - c * (B.squeeze() / H.squeeze())
        # print(c.shape, H.shape, A.shape, d_in_reward_params.shape, in_reward.model.linear1.weight.size())

        # Hack to maintain memory, check later
        agent.optimizer.zero_grad()
        in_reward.optimizer.zero_grad()
        in_gamma.optimizer.zero_grad()
        # TODO: Temporary hack for my NN

        reward_map = get_full_state_reward(env, in_reward)

        in_reward.model.linear1.weight.grad = d_in_reward_params.unsqueeze(-1).T.detach()
        in_gamma.model.linear1.weight.grad = d_in_gamma_params.unsqueeze(-1).T.detach()
        utils.clip_grad_norm_(in_reward.model.parameters(), 40)
        in_reward.optimizer.step()


    print("Actual Reward Over Time") # still need to rescale graph
    s = pd.Series(actual_reward_over_time)
    plt.plot(s.rolling(5).mean())
    plt.ylabel("avg reward")

    plt.show()

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    np.random.seed(0)
    Run_Gridworld_Implicit(15, 50, 5, True, True)