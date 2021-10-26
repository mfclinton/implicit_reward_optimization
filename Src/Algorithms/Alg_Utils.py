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

def Process_Sample(sample, basis, agent, reward_func, gamma_func):
    ids, s, a, prob, r, mask = sample
    B, H, D = s.shape
    _, _, A = a.shape

    s_features = basis.forward(s.view(B * H, D))
    s_features *= mask.view(B*H, 1) #TODO: Check this

    log_pi, dist_all = agent.policy.get_logprob_dist(s_features, a.view(B * H, A))
    log_pi = log_pi.view(B, H) * mask
    # print(log_pi.view(-1).nonzero().size(), log_pi.size(), "loggg Mate")

    in_r = reward_func(s.view(B * H, D), s_features, a.view(B*H, A)).view(B,H)
    in_r *= mask
    # print(in_r.nonzero().size(), in_r.size(), "reward Mate")
    # print(in_r, r)

    in_g = gamma_func(s_features, a.view(B*H, A)).view(B,H)
    in_g *= mask
    # print(in_g.nonzero().size(), in_g.size(), "gamma Mate")

    return s_features, log_pi, in_r, in_g

def calc_grads(model, values, retain_graph=True):
    grads = []
    for v in values:
        param_grads = torch.autograd.grad(v, model.parameters(), retain_graph=retain_graph) #TODO: FIX ALLOW UNUSED
        d_v = torch.cat([torch.flatten(grad) for grad in param_grads], dim=0)
        grads.append(d_v)

    # print(len(grads), grads[0].shape)
    grads = torch.stack(grads, dim=0)

    return grads