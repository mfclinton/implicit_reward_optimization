import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.Gridworld import GridWorld
from models.models import REINFORCE, INTRINSIC_REWARD, INTRINSIC_GAMMA
import torch.nn.functional as F
import torch
from utils.helper import *


def Get_Trajectory(env, agent, max_steps):
    env.reset()

    # Trajectory Data
    states = []
    actions = []
    rewards = []
    log_probs = []

    def update_and_get_action(data):
        state = onehot_state(data["observation"], env.state_space.n)
        action, log_prob = agent.select_action(state)

        states.append(state)
        actions.append(action)
        rewards.append(data["reward"])
        log_probs.append(log_prob)

        return action

    data = env.state
    action = update_and_get_action(data)

    steps = 0
    while not data["done"]:
        data = env.step(action)
        action = update_and_get_action(data)
        
        steps += 1
        if steps >= max_steps:
            break
    
    # print("here")
    # env.render()
    return states, actions, rewards, log_probs



def Run_Gridworld_Implicit(T1, T2):
    env = GridWorld()
    agent = REINFORCE(25, env.state_space.n, env.action_space)
    in_reward = INTRINSIC_REWARD(25, env.state_space.n * env.action_space.n)
    in_gamma = INTRINSIC_GAMMA(25, env.state_space.n)

    for t1 in range(T1):
        real_rewards = None
        log_probs = None
        discounted_in_rewards = None
        in_gammas = None
        for t2 in range(T2):
            states, actions, real_rewards, log_probs = Get_Trajectory(env, agent, 1000)
            # print(log_probs)
            states_matrix = torch.stack(states, dim=0)
            state_actions = onehot_states_to_state_action(states_matrix, actions, env.action_space.n)

            in_rewards = in_reward.get_reward(state_actions).squeeze()
            in_gammas = in_gamma.get_gamma(states_matrix).squeeze()

            # TODO: Check about ^t for each gamma in the traj
            discounted_in_rewards = in_rewards * torch.pow(in_gammas, torch.arange(in_gammas.size()[0]).cuda())
            # Hack: Set gamma to 1, pass in discounted rewards
            agent.update_parameters(discounted_in_rewards, log_probs, 1)
        
        real_rewards = torch.tensor(real_rewards).cuda()

        cumu_log_prob = get_cumulative_sum_front(log_probs)
        # returns = get_returns_t(real_rewards, 1, normalize=False)
        c = (cumu_log_prob * real_rewards).sum()
        h = (cumu_log_prob * (cumu_log_prob * real_rewards)).sum()
        a = (cumu_log_prob * discounted_in_rewards).sum()

        if(h != 0):
            reward_loss = -c * (a / h)
            in_reward.optimizer.zero_grad()
            reward_loss.backward()
            in_reward.optimizer.step()


            print("Reward Total : " + str(real_rewards.sum()))
            print("Fake Reward Total : " + str(discounted_in_rewards.sum()))
            # print(in_reward.model.linear2.weight.view(5,5))
            temp = torch.zeros(25)
            for s in range(env.state_space.n):
                embed_s = onehot_state(s, env.state_space.n)
                a, _ = agent.select_action(embed_s)
                temp[s] = a

            # print(temp.view(5,5))
        else:
            print("skipped")


        # in_gamma.optimizer.zero_grad()
        # gamma_loss.backward()
        # in_gamma.optimizer.step()
        env.render()
        print("T1 : " + str(t1))


    


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    Run_Gridworld_Implicit(100, 100)