import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def onehot_state(state, num_states):
    encoded = torch.zeros(num_states, dtype=torch.double, requires_grad=False)
    encoded[state] = 1.0
    return encoded

def onehot_states_to_state_action(encoded_states, actions, num_actions):
    time_steps = len(actions)
    num_states = encoded_states.size()[1]
    # print(encoded_states)

    idx = torch.nonzero(encoded_states)
    idx[:,1] += torch.tensor(actions, requires_grad=False) * num_states
    encoded = torch.zeros((time_steps, num_states * num_actions), dtype=torch.double, requires_grad=False)
    # print(idx)
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

    returns = Variable(torch.tensor(returns), requires_grad=False)
    if(normalize):
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns

def get_cumulative_sum_front(x):
    total = 0
    cumu_sums = torch.zeros_like(x)

    for i in range(x.shape[0]):
        value = x[i]
        # print(value, x.shape)
        total = value + total
        cumu_sums[i] = total
    return cumu_sums

def get_cumulative_multiply_front(x):
    total = 1
    cumu_mul = torch.ones_like(x)

    for i in range(x.shape[0]):
        value = x[i]
        # print(value, x.shape)
        total = value * total
        cumu_mul[i] = total
    return cumu_mul

def get_param_gradient(model, values, get_sec_grads = False):
    # (T, num_parameters(model))
    grads = []
    # (T, num_parameters(model), num_parameters(model))
    sec_grads = []
    for v in values:
        # Gets the gradient of values w/ respect to model params
        # (num_parameters(model))
        param_grads = torch.autograd.grad(v, model.model.parameters(), create_graph=get_sec_grads, retain_graph=True)
        d_v = torch.cat([torch.flatten(grad) for grad in param_grads], dim=0)
        grads.append(d_v)

        # TODO: review
        if(get_sec_grads):
            dd_v = torch.zeros((d_v.size()[0], d_v.size()[0]))
            for row, d in enumerate(d_v):
                d_param_grads = torch.autograd.grad(d, model.model.parameters(), retain_graph=True)
                dd_v_row = torch.cat([torch.flatten(d_grad) for d_grad in d_param_grads], dim=0)
                dd_v[row] = dd_v_row
            sec_grads.append(dd_v)

    grads = torch.stack(grads, dim=0)
    

    if(get_sec_grads):
        # (T, num_parameters(model), num_parameters(model))
        sec_grads = torch.stack(sec_grads, dim=0)
        return grads, sec_grads

    return grads

# def get_cumulative_sum_front(x):
#     total = 0
#     cumu_sums = []

#     for i in range(len(x)):
#         value = x[i]
#         total = value + total
#         cumu_sums.append(total)

#     cumu_sums = torch.tensor(cumu_sums).cuda()
#     return cumu_sums

def get_average_state_reward(env, in_reward):
    reward_map = torch.zeros(env.state_space.n)
    for s in range(env.state_space.n):
        repeated_state = np.repeat(np.expand_dims(onehot_state(s, env.state_space.n), axis=0), env.action_space.n, axis=0)
        repeated_state = torch.tensor(repeated_state)

        state_action = onehot_states_to_state_action(repeated_state, torch.arange(env.action_space.n), env.action_space.n)
        reward_map[s] += in_reward.get_reward(state_action).mean()
    return reward_map

def get_full_state_reward(env, in_reward):
    reward_map = torch.zeros(env.state_space.n, env.action_space.n)
    for s in range(env.state_space.n):
        repeated_state = np.repeat(np.expand_dims(onehot_state(s, env.state_space.n), axis=0), env.action_space.n, axis=0)
        repeated_state = torch.tensor(repeated_state)

        state_action = onehot_states_to_state_action(repeated_state, torch.arange(env.action_space.n), env.action_space.n)

        reward_map[s,:] = torch.squeeze(in_reward.get_reward(state_action))
    return reward_map