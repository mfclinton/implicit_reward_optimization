import torch
import torch.nn.functional as F

def onehot_state(state, num_states):
    encoded = torch.zeros(num_states, dtype=torch.double)
    encoded[state] = 1.0
    return encoded

def onehot_states_to_state_action(encoded_states, actions, num_actions):
    time_steps = len(actions)
    num_states = encoded_states.size()[1]

    idx = torch.nonzero(encoded_states)
    idx[:,1] += torch.tensor(actions) * num_states
    encoded = torch.zeros((time_steps, num_states * num_actions), dtype=torch.double)
    encoded[idx[:,0], idx[:,1]] = 1
    return encoded
