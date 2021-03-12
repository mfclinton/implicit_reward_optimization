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

def get_returns_t(rewards, gamma, normalize=False):
    eps = 1e-5
    R = 0
    returns = []
    for i in reversed(range(len(rewards))):
        r = rewards[i]
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns).cuda()
    returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns

def get_cumulative_sum_front(x):
    total = 0
    cumu_sums = []

    for i in range(len(x)):
        value = x[i]
        total = value + total
        cumu_sums.append(total)

    cumu_sums = torch.tensor(cumu_sums).cuda()
    return cumu_sums
