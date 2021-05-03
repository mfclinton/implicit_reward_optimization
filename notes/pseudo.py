import torch

# Note: .detach to prevent gradients

# Number of Intervals For The Respective Loop
# ~25

# Use Swarm, explore
# Htop track memory usage, allocate for job

# Experiment for hyperparemeters on gypsum
T1 = 100
T2 = 100
T3 = 100

# Max Timesteps in Environment
num_timesteps = 1000

# 687 Gridworld
env = GridWorld(params)

# Our 3 Models for the policy function, reward function, and gamma function
# Optimizer Notes : AMS Grad, Look into Adam Params
# Might be able to experiment with higher step grad
# Experiment with weight / output initialization
agent = REINFORCE(25, env.state_space.n, env.action_space) # Experiment with tabular
in_reward = INTRINSIC_REWARD(25, env.state_space.n * env.action_space.n)
in_gamma = INTRINSIC_GAMMA(25, env.state_space.n)

# Note: tabular derivs to debug, 1


# Outer Loop
for i in range(T1):
    # Interior Loop
    # Overall, this loop solely is dedicated to optimizing our policy function to the current intrinsic reward / gamma function
    for j in range(T2):
        # States: the states visited in our trajectory at each timestep (S_1 ... S_n)       List<Tensor(size = 25)>, one-hot encoding
        # Actions: the actions taken at in timestep in the state S_t, (A_1 ... A_n)     List<int>
        # Real Rewards: the objective rewards gained at each timestep (R_1 ... R_n)     List<double>
        # Log Prob: The log probability of taking of taking the action taken at A_t, (L_prob_1, ..., L_prob_n)      List<Tensor(size = 1)>
        states, actions, real_rewards, log_probs = Get_Trajectory(env, agent, num_timesteps)

        # Gets a Tensor(size = (timesteps, states * actions)), one-hot encoding of state/action pair [(S_t = 0, A = 0), (S_t = 1, A = 0) ... (S_t = n - 1, A = 3), (S_t = n, A = 3)]
        state_actions = onehot_states_to_state_action(states, actions, env.action_space.n)

        # Evaluates our intrinsic reward based off our state_actions matrix
        in_rewards = in_reward.get_reward(state_actions).squeeze()

        # Gets the internal gamma values for each state visited in our trajectory
        in_gammas = in_gamma.get_gamma(states).squeeze()

        # Multiplies the intrinsic reward at each timestep by the internal gamma at each timestep, giving us the intrinsic discounted reward for each timestep
        discounted_in_rewards = in_rewards * torch.pow(in_gammas, torch.arange(in_gammas.size()[0])) #TODO: fix gammas, not independent of eachother

        # Updates our policy function (Agent) using REINFORCE, passing in our discounted intrinsic rewards, our log probabilitities, and gamma = 1 (Hack since we already applied gamma to discounted rewards)
        agent.update_parameters(discounted_in_rewards, log_probs, 1)

    # NOTE: Unsure how to handle the exterior optimization in terms of loops

    # Initialize to 0
    c, h, a, b = 0
    for k in range(T3):
        # Same deal as before
        states, actions, real_rewards, log_probs = Get_Trajectory(env, agent, num_timesteps)

        # Create a new tensor of the same shape, where cumu_log_prob[i] = log_probs[0:i+1].sum()
        cumu_log_prob = get_cumulative_sum_front(log_probs)

        # Same deal as before
        state_actions = onehot_states_to_state_action(states, actions, env.action_space.n)
        in_rewards = in_reward.get_reward(state_actions).squeeze()
        in_gammas = in_gamma.get_gamma(states).squeeze()
        discounted_in_rewards = in_rewards * torch.pow(in_gammas, torch.arange(in_gammas.size()[0]).cuda())

        # Calculates the respective values from the paper NOTE: MOST IMPORTANT TO CHECK THIS
        # Question about using AutoGradient correctly
        # cumu_log_prob should be length of agent parameterization * timesteps
        # flatten weight array
        # https://pytorch.org/docs/stable/autograd.html, grad w/ respect
        c += (cumu_log_prob * real_rewards).sum() #TODO : comment shapes 
        h += (cumu_log_prob * (cumu_log_prob * real_rewards)).sum() #Note: matrix * vector, sum over axis
        # TODO: what derivatives * derivatives ^ 
        # TODO : real_rewards derivative with respect to reward function
        a += (cumu_log_prob * discounted_in_rewards).sum() #TODO / CHECK: discounted_in_rewards should be a sum, want a vector remove sum
        # Discounting only starts at k, check
        b += # Not sure how to handle autogradient considering 2 gamma instances are in there
    c, h, a, b = average(c), average(h), average(a), average(b)

    reward_loss = -c * (a / h)


    # Updates our Reward Function with respect to the reward_loss0
    in_reward.optimizer.zero_grad()
    reward_loss.backward()
    # computes cumulative log prob derivs too, check
    in_reward.optimizer.step() 

    # in_gamma.optimizer.zero_grad()
    # gamma_loss.backward()
    # in_gamma.optimizer.step()

        

