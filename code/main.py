import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.Gridworld import GridWorld
from models.models import REINFORCE
import torch.nn.functional as F
import torch

def encode_state(state, num_states):
    encoded = torch.zeros(num_states, dtype=torch.double)
    encoded[state] = 1.0
    # print(encoded)
    return encoded


def Run_Gridworld(max_steps = 1000, episodes = 100):
    env = GridWorld()
    agent = REINFORCE(25, env.state_space.n, env.action_space)

    for episode in range(episodes):
        env.reset()
        state = env.state

        rewards = []
        log_probs = []
        entropies = []
        gamma = .99

        encoded_obs = encode_state(state["observation"], env.state_space.n)
        action, log_prob, entropy = agent.select_action(encoded_obs)

        rewards.append(state["reward"])
        log_probs.append(log_prob)
        entropies.append(entropy)
        
        # print("LOL")
        returns = 0
        steps = 0

        while not state["done"]:
            state = env.step(action)
            encoded_obs = encode_state(state["observation"], env.state_space.n)
            print(state, action)

            action, log_prob, entropy = agent.select_action(encoded_obs)

            rewards.append(state["reward"])
            log_probs.append(log_prob)
            entropies.append(entropy)

            returns += state["reward"]
            steps += 1
            if steps >= max_steps:
                break
        
        # print(len(rewards), len(log_probs), len(entropies))
        # print(rewards[0],log_probs[0], entropies[0])
        agent.update_parameters(rewards, log_probs, entropies, gamma)

if __name__ == "__main__":
    Run_Gridworld(episodes=150)