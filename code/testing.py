import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.models import *
from envs.Gridworld import GridWorld
from torch.distributions import Categorical

# Notes
# Gonna have Agent keep internal log needed for Reinforce
def Test_Reinforce():
    env = GridWorld()
    data = env.state
    print("Initial State : ", data)

    # (hidden_size, input_size, output_size)
    agent = REINFORCE(1, env.state_space.n, env.action_space)
    state = onehot_state(data["observation"], env.state_space.n) #(S)
    # probability of actions
    probs = agent.model(state)
    print("Probs : ", probs)

    m = Categorical(probs)
    # chooses action based on probs
    action = m.sample() # (1,)
    # log probability of taking action
    log_prob = m.log_prob(action) # (1,)
    print("Action: ", action)
    print("LogProb: ", m.log_prob(action))
    log_prob.backward() # Computes the gradient of Log(pi(a|s)) w/ respect to weights
    phi = [torch.flatten(p.grad) for p in agent.model.parameters()]
    phi = torch.cat(phi, dim=0)
    print("Size of Phi : ", len(phi))
    print(phi)

def Run_Tests():
    Test_Reinforce()

if __name__ == "__main__":
    Run_Tests()