import torch
# <desc> cumu_phi: cumulative phi values, shape (N, len(agent_params))
# <desc> inner_discounted_reward: instrinsic rewards from learned reward function with learned discounting, shape (N,)
# <return> returns H approximation
def Approximate_H(cumu_phi, inner_discounted_reward):
    # Debug Check: Make sure right shapes
    return ((cumu_phi * cumu_phi).T * inner_discounted_reward).T.sum(axis = 0)

def New_Approximate_H(phi, discounted_in_returns):
    # Debug Check: Make sure right shapes
    return ((phi * phi) * torch.abs(discounted_in_returns.view(-1,1))).sum(axis = 0)


# <desc> phi: phi values, shape (N, len(agent_params))
# <desc> cumu_gamma: multiplied together intrinsic gamma values, shape (N,N) is triangular)
# <desc> d_inner_reward: the derivative of the intrinsic reward parameters, shape (N,len(reward_fn_params)))
# <return> returns A approximation
def Approximate_A(phi, cumu_gamma, d_inner_reward):
    # print(phi, cumu_gamma)
    # print(phi.size(), cumu_gamma.size(), d_inner_reward.size())
    return (phi * torch.matmul(cumu_gamma, d_inner_reward)).sum(dim = 0)

def New_Approximate_A(phi, cumu_gamma, d_inner_reward):
    # print(phi, cumu_gamma)
    # print(phi.size(), cumu_gamma.size(), d_inner_reward.size())
    # print(phi.size(), cumu_gamma.size(), d_inner_reward.size())

    # # DOUBLE CHECK THAT APPROXIMATION IS RIGHT
    return (phi * torch.matmul(cumu_gamma, d_inner_reward)).sum(dim = 0)

# <desc> cumu_phi: cumulative phi values, shape (N, len(agent_params))
# <desc> real_rewards: the real returns earned by the agent, shape (N,)
# <return> returns c
def Calculate_C(cumu_phi, real_rewards):
    return (cumu_phi.T * real_rewards).T.sum(dim=0)

def New_Calculate_C(phi, cumu_gammas, discounted_returns):
    # print(phi.size(), cumu_gammas.size(), discounted_returns.size())
    # print(phi, cumu_gammas[0,:])
    # print(phi * cumu_gammas[0,:].view(-1,1))
    ## Double check, should this just direct multiplication. Why [0,:]?
    return (torch.mm(cumu_gammas, phi) * discounted_returns.view(-1,1)).sum(dim=0)

    # return (phi * cumu_gammas[0,:].view(-1,1) * discounted_returns.view(-1,1)).sum(dim=0)
    
    # return (phi * discounted_returns.view(-1,1)).sum(dim=0)


# TODO: CHECK THIS
def Calculate_B(phi, d_gammas, inner_reward):
    B = 0
    for j in range(phi.shape[0]):
        # print(inner_reward[j:].view(-1,1))
        r_term = (torch.cumprod(d_gammas[j:,:],dim=0) * inner_reward[j:].view(-1,1)).sum(dim=0)
        # print(phi.shape, inner_reward.shape, d_gammas.shape)
        B += phi[j,:] * r_term
    return B

# <desc> inner_gamma: intrinsic gamma values, shape (N,))
# <returns> cumu_gamma, cumulative sum of gamma starting from the row index, shape (N, N)
def Get_Cumulative_Gamma(inner_gamma):
    # inner_gamma[:] = .95
    T = inner_gamma.size()[0]
    cumu_gamma = torch.zeros((T,T), dtype=torch.float64)
    for t in range(T):
        cumu_gamma[t,t:] = torch.cumprod(inner_gamma[t:], dim=0, dtype=torch.float64)
        cumu_gamma[t,t] = 1 #TODO remove this
    return cumu_gamma
    # Debug Check: Make Sure Triangular

# <desc> phi: phi values, shape (N, len(reward_agent_params))
# <returns> cumu_phi, cumulative sum of phi, shape (N, len(reward_agent_params))
def Get_Cumulative_Phi(phi):
    return torch.cumsum(phi, dim=0) #TODO: check order of sum

def Get_Discounted_Returns(rewards, cumu_gamma, normalize=False):
    returns = (cumu_gamma * rewards).sum(dim=1)
    if(normalize):
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns