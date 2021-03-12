import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.Gridworld import GridWorld
from models.models import REINFORCE
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



def Run_Gridworld(max_steps = 1000, episodes = 100):
    env = GridWorld()
    agent = REINFORCE(25, env.state_space.n, env.action_space)

    for episode in range(episodes):
        states, actions, rewards, log_probs = Get_Trajectory(env, agent, 1000)
        # print(len(states))
        agent.update_parameters(rewards, log_probs, .95)

if __name__ == "__main__":
    Run_Gridworld(episodes=1000)