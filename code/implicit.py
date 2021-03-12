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
    
    return states, actions, rewards, log_probs



def Run_Gridworld_Implicit(T1, T2):
    env = GridWorld()
    agent = REINFORCE(25, env.state_space.n, env.action_space)
    in_reward = INTRINSIC_REWARD(25, env.state_space.n * env.action_space.n)
    in_gamma = INTRINSIC_GAMMA(25, env.state_space.n)

    for t1 in range(T1):
        for t2 in range(T2):
            states, actions, rewards, log_probs = Get_Trajectory(env, agent, 1000)
            
            states_matrix = torch.stack(states, dim=0)
            state_actions = onehot_states_to_state_action(states_matrix, actions, env.action_space.n)

            in_rewards = in_reward.get_reward(state_actions).squeeze()
            in_gammas = in_gamma.get_gamma(states_matrix).squeeze()

            # TODO: Check about ^t for each gamma in the traj
            discounted_rewards = in_rewards * torch.pow(in_gammas, torch.arange(in_gammas.size()[0]).cuda())

            # Hack: Set gamma to 1, pass in discounted rewards
            agent.update_parameters(discounted_rewards, log_probs, 1)
    
    # c = 


if __name__ == "__main__":
    Run_Gridworld_Implicit(10, 10)