import torch
# <desc> cumu_phi: cumulative phi values, shape (N, len(agent_params))
# <desc> inner_discounted_reward: instrinsic rewards from learned reward function with learned discounting, shape (N,)
# <return> returns H approximation
def Approximate_H(cumu_phi, inner_discounted_reward):
    # Debug Check: Make sure right shapes
    return ((cumu_phi * cumu_phi).T * inner_discounted_reward).T.sum(axis = 0)

# <desc> phi: phi values, shape (N, len(reward_agent_params))
# <desc> cumu_gamma: multiplied together intrinsic gamma values, shape (N,N) is triangular)
# <desc> d_inner_reward: the derivative of the intrinsic reward parameters, shape (N,len(reward_fn_params)))
# <return> returns A approximation
def Approximate_A(phi, cumu_gamma, d_inner_reward):
    # print(phi, cumu_gamma)
    return (phi * torch.matmul(cumu_gamma, d_inner_reward)).sum(dim = 0)

# <desc> cumu_phi: cumulative phi values, shape (N, len(agent_params))
# <desc> real_rewards: the real returns earned by the agent, shape (N,)
# <return> returns c
def Calculate_C(cumu_phi, real_rewards):
    return (cumu_phi.T * real_rewards).T.sum(dim=0)

# <desc> inner_gamma: intrinsic gamma values, shape (N,))
# <returns> cumu_gamma, cumulative sum of gamma starting from the row index, shape (N, N)
def Get_Cumulative_Gamma(inner_gamma):
    T = inner_gamma.size()[0]
    cumu_gamma = torch.zeros((T,T), dtype=torch.float64)
    for t in range(T):
        cumu_gamma[t,t:] = torch.cumprod(inner_gamma[t:], dim=0, dtype=torch.float64)
    return cumu_gamma
    # Debug Check: Make Sure Triangular

# <desc> phi: phi values, shape (N, len(reward_agent_params))
# <returns> cumu_phi, cumulative sum of phi, shape (N, len(reward_agent_params))
def Get_Cumulative_Phi(phi):
    return torch.cumsum(phi, dim=0) #TODO: check order of sum