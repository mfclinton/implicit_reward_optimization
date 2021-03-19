import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.Gridworld import GridWorld
from models.models import REINFORCE, INTRINSIC_REWARD, INTRINSIC_GAMMA
import torch.nn.functional as F
import torch
from utils.helper import *

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
            print(cumu_gammas)
            1/0

            # TODO: Check about ^t for each gamma in the traj
            discounted_in_rewards = in_rewards * cumu_gammas
            # Hack: Set gamma to 1, pass in discounted rewards
            agent.update_parameters(discounted_in_rewards, log_probs, 1)
        
        c = 0
        h = 0
        a = 0
        A = 0
        H = 0
        avg_real_rewards = 0
        avg_fake_rewards = 0
        for t3 in range(T3):
            states, actions, real_rewards, log_probs = Get_Trajectory(env, agent, max_timesteps)

            #temp solution to get phi
            phi = []
            d_phi = []
            for lp in log_probs:
                agent.optimizer.zero_grad()
                lp.backward(retain_graph = True)
                log_pi = [torch.flatten(p.grad) for p in agent.model.parameters()]
                log_pi = torch.cat(log_pi, dim=0)
                phi.append(log_pi)
                
                lp.backward(retain_graph = True)
                d_log_pi = [torch.flatten(p.grad) for p in agent.model.parameters()]
                d_log_pi = torch.cat(d_log_pi, dim=0)
                d_phi.append(d_log_pi)

            phi = torch.stack(phi, dim=0)
            d_phi = torch.stack(d_phi, dim=0)

            real_rewards = torch.tensor(real_rewards)
            for t in range(phi.size()[0]):
                unsqueezed = phi[t].unsqueeze(-1)
                term = torch.matmul(unsqueezed, unsqueezed.T)
                term += d_phi[t]
                term *= real_rewards[t]
                H += term
                # print(left_term.size())
                # 1/0
            # print(H.size())

            # print(phi.shape, d_phi.shape)



            avg_real_rewards += real_rewards.sum()
            cumu_log_prob = get_cumulative_sum_front(phi)
            # returns = get_returns_t(real_rewards, 1, normalize=False)


            states_matrix = torch.stack(states, dim=0)
            state_actions = onehot_states_to_state_action(states_matrix, actions, env.action_space.n)

            in_rewards = in_reward.get_reward(state_actions).squeeze()
            in_gammas = in_gamma.get_gamma(states_matrix).squeeze()
            in_gammas = torch.ones_like(in_gammas) * .95 #TODO: REMOVE THIS

            # TODO: Check about ^t for each gamma in the traj
            # discounted_in_rewards = in_rewards * torch.pow(in_gammas, torch.arange(in_gammas.size()[0]))
            # avg_fake_rewards += discounted_in_rewards.sum()


            c += (cumu_log_prob.T * real_rewards).T.sum(axis=0)
            # print(c.shape)
            h += (cumu_log_prob * (cumu_log_prob.T * real_rewards).T).sum(axis=0)
            # print(h.shape)


            d_in_rewards = []
            # print(1/0)
            for ir in in_rewards:
                in_reward.optimizer.zero_grad()
                # print(ir)
                ir.backward(retain_graph=True)
                # print("-")
                d_in_r = [torch.flatten(p.grad) for p in in_reward.model.parameters()]
                d_in_r = torch.cat(d_in_r, dim=0)
                d_in_rewards.append(d_in_r)

            d_in_rewards = torch.stack(d_in_rewards, dim=0)
            # print(d_in_rewards.shape, cumu_log_prob.shape)
            # 1/0
            for t in range(phi.size()[0]):
                d_rwds = (d_in_rewards[t:,:].T * torch.pow(in_gammas[t:], torch.arange(in_gammas.size()[0] - t))).T
                # print(torch.sum(d_rwds, axis=0).size())
                # print(d_rwds.size())
                term = torch.matmul(phi[t].unsqueeze(-1), torch.sum(d_rwds, axis=0).unsqueeze(-1).T)
                # print(term.size())
                A += term

        print(print(H))
        print(torch.inverse(H).size(), A.size())
        1/0

            # a += (cumu_log_prob * discounted_in_rewards).sum()
        c /= T3
        h /= T3
        a /= T3
        avg_real_rewards /= T3
        avg_fake_rewards /= T3

        if(h != 0):
            reward_loss = -c * (a / h)
            in_reward.optimizer.zero_grad()
            reward_loss.backward()
            in_reward.optimizer.step()


            print("Reward Avg : " + str(avg_real_rewards))
            print("Fake Reward Total : " + str(avg_fake_rewards))
            # print(in_reward.model.linear2.weight.view(5,5))


            print("--- Reward Map---")
            # reward_map = get_average_state_reward(env, in_reward)
            reward_map = get_full_state_reward(env, in_reward)
            print(reward_map)
            # print(reward_map.view(5,5))
            print("------")
        else:
            print("skipped")


            # in_gamma.optimizer.zero_grad()
            # gamma_loss.backward()
            # in_gamma.optimizer.step()
        env.render()
        print("T1 : " + str(t1))


    


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    Run_Gridworld_Implicit(100, 25, 200)