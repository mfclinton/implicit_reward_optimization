import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.Gridworld import GridWorld
from envs.ChrisWorld import ChrisWorld
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

# Make this better
def Get_Prior_Reward(env, prior_id):
    reformat_prior = False

    # Run to right
    if prior_id == 0:
        reward_1 = torch.full((env.action_space.n * env.state_space.n,), -1.)
        reward_1[env.state_space.n * 3:] = 1.0
        print(reward_1)
    elif prior_id == 1:
        # MANHATTEN
        h, w = env._grid_dims
        reward_1 = torch.zeros((env.action_space.n * env.state_space.n,))
        
        for y in range(h):
            for x in range(w):
                state = y * h + x
                for action in range(env.action_space.n):
                    new_x = x
                    new_y = y
                    if action == 0:  # move up
                        new_y = y - 1
                    elif action == 1:  # move down
                        new_y = y + 1
                    elif action == 2:  # move left
                        new_x = x - 1
                    elif action == 3:  # move right
                        new_x = x + 1
                    
                    reward_1[state * 4 + action] = - (np.abs(new_x - 4) + np.abs(new_y - 4)) / 8 + (0.125 * 2)
        
        # reward_1 = reward_1.view(25,4)
        # bottom_left_indexes = torch.tensor([10,11,12,15,16,17,20,21,22])
        # reward_1[bottom_left_indexes, :] -= 2
        # reward_1 = reward_1.view(-1)
        
        reformat_prior = True
        print(reward_1.view(25,4))
    elif prior_id == 2:
        # Avoid bottom left
        reward_1 = torch.zeros((env.action_space.n * env.state_space.n,)).view(25,4)
        bottom_left_indexes = torch.tensor([10,11,12,15,16,17,20,21,22])
        reward_1[bottom_left_indexes, :] -= 2
        reward_1 = reward_1.view(-1)
        reformat_prior = True
        print(reward_1.view(25,4))
    else:
        return None

    if(reformat_prior):
        print("Reformated Prior")
        reward_1 = reward_1.view(25,4)
        reward_1 = reward_1.T.reshape(-1)

    return reward_1



def Run_Gridworld_Implicit(T1, T2, T3, approximate, reuse_trajectories):
    Use_Chris_World = False
    Save_Data = True
    prior_id = -1
    
    env = GridWorld() # Creates Environment
    agent = REINFORCE(env.state_space.n, env.action_space) #Create Policy Function, (S) --> (25) --> (A)

    if Use_Chris_World:
        env = ChrisWorld() #TODO: remove
        agent = CHRIS_REINFORCE() #TODO: remove

    # agent = REINFORCE(env.state_space.n, env.action_space) #Create Policy Function, (S) --> (25) --> (A)
    # in_reward = INTRINSIC_REWARD(env.state_space.n * env.action_space.n, None) #Create Intrinsic Reward Function, (S * A) --> (25) --> (1)
    in_reward = INTRINSIC_REWARD(env.state_space.n * env.action_space.n, Get_Prior_Reward(env, prior_id)) #Create Intrinsic Reward Function, (S * A) --> (25) --> (1)
    in_gamma = INTRINSIC_GAMMA(env.state_space.n) #Creates Intrinsic Gamma (S) --> (25) --> (1)
    
    # DEBUG
    actual_reward_over_time = []
    trajectories = []
    for t1 in range(T1):
        # TODO: Can we keep the same agent across iterations?
        # agent = REINFORCE(env.state_space.n, env.action_space)
        agent.reset()
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
                    break

                states, actions, real_rewards, log_probs = random.choice(trajectories)
                # states, actions, real_rewards = random.choice(trajectories)

            # Creates a matrix of size (time_steps, S)
            states_matrix = torch.stack(states, dim=0)

            # Gets Log Probs, TODO find out which log probs we are sampled from
            if reuse_trajectories:
                probs = agent.model(states_matrix)
                sampler = Categorical(probs)
                log_probs = sampler.log_prob(actions)

            # Creates a State Action matrix of size (time_steps, S * A)
            state_actions = onehot_states_to_state_action(states_matrix, actions, env.action_space.n)

            # Evaluates intrinsic reward and gamma
            in_rewards = in_reward.get_reward(state_actions).squeeze() #(time_steps, S*A) --> (time_steps, 1)
            in_gammas = in_gamma.get_gamma(states_matrix).squeeze() #(time_steps, S) --> (time_steps, 1)
            
            # cumu_gammas = get_cumulative_multiply_front(in_gammas)
            cumu_gammas = Get_Cumulative_Gamma(in_gammas)

            # TODO: Check about ^t for each gamma in the traj
            # only care about from start
            # discounted_in_returns = Get_Discounted_Returns(in_rewards, cumu_gammas, normalize=False)
            
            agent.update_parameters(in_rewards, log_probs, cumu_gammas)
            # agent.update_parameters(torch.tensor(real_rewards), log_probs, cumu_gammas)
        
            # Debug Stuff
            if(not reuse_trajectories):
                actual_return = real_rewards.sum()
                actual_reward_over_time.append(actual_return)

        c = 0
        H = 0
        A = 0

        # Debugging
        total_steps = 0
        total_average_actual_reward = 0
        total_average_intrinsic_reward = 0
        visited_states = torch.zeros((env.state_space.n))

        for t3 in range(T3):
            #Same as before
            states, actions, real_rewards, log_probs = Get_Trajectory(env, agent) #Samples a trajectory
            if reuse_trajectories:
                trajectories.append((states, actions, real_rewards, log_probs))
                # trajectories.append((states, actions, real_rewards))

            # (T)
            real_rewards = torch.tensor(real_rewards, requires_grad=False)

            # (T, S)
            states_matrix = torch.stack(states, dim=0)
            # (T, S * A)
            state_actions = onehot_states_to_state_action(states_matrix, actions, env.action_space.n)
            
            # (T)
            in_rewards = in_reward.get_reward(state_actions).squeeze() #(time_steps, S*A) --> (time_steps, 1)
            # (T)
            in_gammas = in_gamma.get_gamma(states_matrix).squeeze() #(time_steps, S) --> (time_steps, 1)
            
            # (T)
            # cumu_gammas = get_cumulative_multiply_front(in_gammas)
            cumu_gammas = Get_Cumulative_Gamma(in_gammas)
            
            # (T) Debug
            discounted_in_rewards = in_rewards * cumu_gammas[0]

            # BOTH (T, num_params(agent))
            if(not approximate):
                phi, d_phi = get_param_gradient(agent, log_probs, get_sec_grads = True)
            else:
                phi = get_param_gradient(agent, log_probs, get_sec_grads = False)


            # (T, num_params(agent))
            # cumu_phi = get_cumulative_sum_front(phi)
            cumu_phi = Get_Cumulative_Phi(phi)

            cumu_d_phi = torch.zeros(cumu_phi.size()[0])
            if(not approximate):
                # (T, num_params(agent))
                cumu_d_phi = get_cumulative_sum_front(d_phi)

            # (T, num_params(in_reward))
            d_in_reward = get_param_gradient(in_reward, in_rewards, get_sec_grads = False)
            # TODO: GAMMA
            # d_in_gamma = get_param_gradient(in_gamma, in_gammas, get_sec_grads = False)

            # (num_params(agent))
            # c += (cumu_phi.T * real_rewards).T.sum(axis=0)
            c += Calculate_C(cumu_phi, real_rewards)
            # b += cumu_phi * in_gammas * d_in_gamma * ( - log_probs)
            # b += Calculate_B()

            if (approximate):
                H += Approximate_H(cumu_phi, discounted_in_rewards)
                A += Approximate_A(phi, cumu_gammas, d_in_reward)
            else:
                pass # TODO

            T = len(states)
            # Debugging
            total_steps += T
            actual_return = real_rewards.sum()
            actual_reward_over_time.append(actual_return)
            total_average_actual_reward += actual_return
            total_average_intrinsic_reward += discounted_in_rewards.sum()
            visited_states += states_matrix.sum(axis=0)

        if(not approximate):
            H += torch.diag(torch.full((H.size()[0],),1e-10))
        else:
            H += 1e-10
        
        c /= T3
        H /= T3
        A /= T3

        # TODO: Check, Sign of Gradient in Paper Is For Update
        d_in_reward_params = None
        if(not approximate):
            d_in_reward_params = torch.matmul(c, torch.matmul(torch.inverse(H), A))
        else:
            # print(c.size(), A.size(), H.size())
            d_in_reward_params = - c * (A.squeeze() / H.squeeze()) #TODO make negative
        print(c.shape, H.shape, A.shape, d_in_reward_params.shape, in_reward.model.linear1.weight.size())

        # Hack to maintain memory, check later
        agent.optimizer.zero_grad()
        in_reward.optimizer.zero_grad()
        in_gamma.optimizer.zero_grad()
        # TODO: Temporary hack for my NN

        reward_map = get_full_state_reward(env, in_reward)

        in_reward.model.linear1.weight.grad = d_in_reward_params.unsqueeze(-1).T.detach()
        utils.clip_grad_norm_(in_reward.model.parameters(), 40)
        in_reward.optimizer.step()

        # DEBUGGING
        print("--- Reward Map---")
        print(reward_map)
        if not Use_Chris_World:
            print("--- Top Moves ---")
            print(reward_map.argmax(axis=1).view(5,5))
            print("--- Total Visited States ---")
            print(visited_states.view(5,5))
        else:
            print("--- Top Moves ---")
            print(reward_map.argmax(axis=1))
            print("--- Total Visited States ---")
            print(visited_states)

        print("--- Other ---")
        print("Average Steps: ", total_steps / T3)
        print("Average Actual Reward: ", total_average_actual_reward / T3)
        print("Average Intrinsic Reward: ", total_average_intrinsic_reward / T3)
        # print(in_rewards)
        print("Iteration ", t1)
    
    if not Use_Chris_World:
        print("FINAL REWARD MAP")
        reward_map = get_full_state_reward(env, in_reward)
        print(reward_map)
        print(reward_map.argmax(axis=1).view(5,5))
        if in_reward.prior_reward != None:
            print("VISUALIZE LEARNED REWARD FUNCTION W/O PRIOR")
            in_reward.prior_reward *= -1
            reward_map = get_full_state_reward(env, in_reward)
            print(reward_map)
            print(reward_map.argmax(axis=1).view(5,5))
            in_reward.prior_reward *= -1



    c_word_str = "c_word" if Use_Chris_World == True else ""
    using_prior_str = "with_prior" if in_reward.prior_reward != None else ""
    result_path =  "saved\\reward_{0}_{1}_{2}_({3},{4},{5})_{6}_{7}{8}\\".format(actual_reward_over_time[-1].item(), approximate, reuse_trajectories, T1, T2, T3, c_word_str, using_prior_str, prior_id)

    print("Actual Reward Over Time") # still need to rescale graph
    print(actual_reward_over_time)
    s = pd.Series(actual_reward_over_time)
    # print(s.rolling(50).mean())
    plt.plot(s.rolling(50).mean())
    plt.ylabel("avg reward")

    if Save_Data:
        try:
            os.mkdir(result_path)
        except:
            print("directory already exists")
            result_path += str(random.random()) + "\\"
            os.mkdir(result_path)
        
        torch.save(in_reward.model.state_dict(), result_path + "reward_model")
        torch.save(in_gamma.model.state_dict(), result_path + "gamma_model")
        plt.savefig(result_path + "graph.png")

    plt.show()
    # TODO: Need to elongate graph to inclue the inner updates

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    Run_Gridworld_Implicit(50, 500, 100, True, True)