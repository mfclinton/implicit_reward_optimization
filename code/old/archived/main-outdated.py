import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.Gridworld import GridWorld
from envs.ChrisWorld import ChrisWorld
from models.models import REINFORCE
import torch.nn.functional as F
import torch
from utils.helper import *
import math


def Get_Trajectory(env, agent, max_steps):
    env.reset()

    # Trajectory Data
    states = []
    actions = []
    rewards = []
    log_probs = []

    def update_and_get_action(data):
        state = onehot_state(data["observation"], env.state_space.n)
        param = agent.model.forward(state)
        prob = 1 / (1 + torch.exp(-param))
        action = 0
        if(np.random.rand() > prob):
            action = 1
            prob = 1 - prob
        
        log_prob = torch.log(prob)

        # action, log_prob = agent.select_action(state)

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



def Run_Gridworld(max_steps = 1000, episodes = 100):
    env = ChrisWorld()
    agent = REINFORCE(env.state_space.n, env.action_space)

    for episode in range(episodes):
        states, actions, rewards, log_probs = Get_Trajectory(env, agent, 1000)
        print(len(states))
        agent.update_parameters(rewards, log_probs, .95)
        print(rewards)
        # print(actions)

if __name__ == "__main__":
    Run_Gridworld(episodes=500)