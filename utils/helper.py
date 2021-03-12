import torch
import torch.nn.functional as F

def onehot_state(state, num_states):
    encoded = torch.zeros(num_states, dtype=torch.double)
    encoded[state] = 1.0
    return encoded

def onehot_state_action(state, num_states, action, num_actions):
    encoded = torch.zeros(num_states * num_actions, dtype=torch.double)
    encoded[state + num_states * action] = 1
    return encoded
