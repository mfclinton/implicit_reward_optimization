from numpy import gradient
import torch

# PAPER EQUATIONS
def Approximate_H(phi, discounted_in_returns, weight_decay):
    # Debug Check: Make sure right shapes
    # TODO: Check weight decay
    return ((phi * phi) * torch.abs(discounted_in_returns.view(-1,1))).sum(axis = 0) - weight_decay 

def Approximate_A(phi, cumu_gamma, d_inner_reward):
    # TODO: Make sure A is correct
    return (phi * torch.matmul(cumu_gamma, d_inner_reward)).sum(dim = 0) 
    # (T, num_param_policy) ||||| (T, T) and (T, num_params_reward) -> T, num_params_reward

def Calculate_C(phi, cumu_gammas, discounted_returns):
    return (torch.mm(cumu_gammas, phi) * discounted_returns.view(-1,1)).sum(dim=0) 

def Calculate_B(phi, d_gammas, inner_reward):
    B = 0
    for j in range(phi.shape[0]):
        r_term = (torch.cumprod(d_gammas[j:,:],dim=0) * inner_reward[j:].view(-1,1)).sum(dim=0)
        B += phi[j,:] * r_term
    return B

# HELPERS
def Get_Cumulative_Gamma(inner_gamma):
    # inner_gamma[:] = .95
    T = inner_gamma.size()[0]
    cumu_gamma = torch.zeros((T,T), dtype=torch.float32)
    for t in range(T):
        cumu_gamma[t,t+1:] = torch.cumprod(inner_gamma[t+1:], dim=0, dtype=torch.float32)
        cumu_gamma[t,t] = 1 #TODO Check if extra 1's cause issue?
    return cumu_gamma

def Get_Discounted_Returns(rewards, cumu_gamma, normalize=False, eps=1e-4):
    returns = (cumu_gamma * rewards).sum(dim=1)
    if(normalize):
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns

def calc_grads(model, values, retain_graph=True):
    grads = []
    for v in values:
        param_grads = torch.autograd.grad(v, model.parameters(), retain_graph=retain_graph) #TODO: FIX ALLOW UNUSED
        d_v = torch.cat([torch.flatten(grad) for grad in param_grads], dim=0)
        grads.append(d_v)

    # print(len(grads), grads[0].shape)
    grads = torch.stack(grads, dim=0)
    # print(grads.shape)
    # 1/0

    return grads
