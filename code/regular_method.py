import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.Gridworld import GridWorld
from models.models import REINFORCE, INTRINSIC_REWARD, INTRINSIC_GAMMA
import torch.nn.functional as F
import torch
from utils.helper import *
# Memory Leak
import gc

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

def Get_Trajectory(env, agent, max_steps):
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

    steps = 0
    while not data["done"]:
        data = env.step(action) #take action
        action = update_and_get_action(env, agent, data, states, actions, rewards, log_probs) #updates trajectory with new info
        
        steps += 1
        if steps >= max_steps: #we've taken max_steps steps at this point
            break
    
    return states, actions, rewards, log_probs


def Run_Gridworld_Implicit(T1, T2, T3):
    env = GridWorld() # Creates Environment
    agent = REINFORCE(env.state_space.n, env.action_space) #Create Policy Function, (S) --> (25) --> (A)
    in_reward = INTRINSIC_REWARD(env.state_space.n * env.action_space.n) #Create Intrinsic Reward Function, (S * A) --> (25) --> (1)
    in_gamma = INTRINSIC_GAMMA(env.state_space.n) #Creates Intrinsic Gamma (S) --> (25) --> (1)
    max_timesteps = 1000 # Max number of steps in an episode
    for t1 in range(T1):
        for t2 in range(T2):
            # Get Trajectory returns lists of elements with the following dims
            # States, (S,)
            # Actions, (1,)
            # Real_Rewards, (1,)
            # Log_Probs, TODO
            states, actions, real_rewards, log_probs = Get_Trajectory(env, agent, max_timesteps) #Samples a trajectory
            
            # Creates a matrix of size (time_steps, S)
            states_matrix = torch.stack(states, dim=0)
            # Creates a State Action matrix of size (time_steps, S * A)
            state_actions = onehot_states_to_state_action(states_matrix, actions, env.action_space.n)

            # Evaluates intrinsic reward and gamma
            in_rewards = in_reward.get_reward(state_actions).squeeze() #(time_steps, S*A) --> (time_steps, 1)
            in_gammas = in_gamma.get_gamma(states_matrix).squeeze() #(time_steps, S) --> (time_steps, 1)
            cumu_gammas = get_cumulative_multiply_front(in_gammas)

            # TODO: Check about ^t for each gamma in the traj
            discounted_in_rewards = in_rewards * cumu_gammas
            
            # Hack: Set gamma to 1, pass in discounted rewards
            agent.update_parameters(discounted_in_rewards, log_probs, 1)
        
        c = 0
        H = 0
        A = 0

        # Debugging
        total_steps = 0
        for t3 in range(T3):
            #Same as before
            states, actions, real_rewards, log_probs = Get_Trajectory(env, agent, max_timesteps) #Samples a trajectory
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
            cumu_gammas = get_cumulative_multiply_front(in_gammas)
            # (T)
            discounted_in_rewards = in_rewards * cumu_gammas

            # BOTH (T, num_params(agent))
            phi = 0
            d_phi = 0
            evaluate_d_phi = False

            if(evaluate_d_phi):
                phi, d_phi = get_param_gradient(agent, log_probs, get_sec_grads = True)
            else:
                phi = get_param_gradient(agent, log_probs, get_sec_grads = False)


            # (T, num_params(agent))
            cumu_phi = get_cumulative_sum_front(phi)

            cumu_d_phi = torch.zeros(cumu_phi.size()[0])
            if(evaluate_d_phi):
                # (T, num_params(agent))
                cumu_d_phi = get_cumulative_sum_front(d_phi)

            # (T, num_params(in_reward))
            d_in_reward = get_param_gradient(in_reward, in_rewards, get_sec_grads = False)

            # (num_params(agent))
            c += (cumu_phi.T * real_rewards).T.sum(axis=0)

            T = len(states)

            # Debugging
            total_steps += T
            for t in range(T):
                # === Computing H ===
                phi_s_a = cumu_phi[t].unsqueeze(-1) #(num_params(agent), 1)
                # (num_params(agent), num_params(agent))
                cumu_phi_cum_phi_T = torch.matmul(phi_s_a, phi_s_a.T)
                # TODO: make sure adding 2nd deriv right
                H += (cumu_phi_cum_phi_T + cumu_d_phi[t]) * real_rewards[t]
                
                # === Computing A ===
                # (T - t)
                k_gammas = get_cumulative_multiply_front(in_gammas[t:])
                # (1, num_parameters(in_reward))
                gamma_d_r = (d_in_reward[t:].T * k_gammas).sum(axis=1).unsqueeze(-1).T
                # (num_parameters(agent), num_parameters(in_reward))
                A += torch.matmul(phi_s_a, gamma_d_r)

        c /= T3
        H /= T3
        H += torch.diag(torch.full((H.size()[0],),1e-6))
        A /= T3

        # TODO: Check
        d_in_reward_params = - torch.matmul(c, torch.matmul(torch.inverse(H), A))
        # print(c.shape, H.shape, A.shape, d_in_reward_params.shape, in_reward.model.linear1.weight.size())

        # Hack to maintain memory, check later
        agent.optimizer.zero_grad()
        in_reward.optimizer.zero_grad()
        in_gamma.optimizer.zero_grad()
        # TODO: Temporary hack for my NN

        in_reward.model.linear1.weight.grad = d_in_reward_params.unsqueeze(-1).T.detach()
        in_reward.optimizer.step()

        # DEBUGGING
        print("--- Reward Map---")
        reward_map = get_full_state_reward(env, in_reward)
        print(reward_map)
        print("Average Steps: ", total_steps / T3)

        # meme = 0
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             # print(type(obj), obj.size())
        #             meme += 1
        #     except:
        #         pass
        # print("MEME: ", meme)

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    Run_Gridworld_Implicit(100, 100, 0)