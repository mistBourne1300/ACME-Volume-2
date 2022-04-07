# policy_iteration.py
"""Volume 2: Policy Function Iteration.
<Name>
<Class>
<Date>
"""

import gym
import numpy as np

# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3

P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]



# Problem 1
def value_iteration(P, nS ,nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    V_old = np.zeros(nS)
    for k in range(maxiter):
        V_new = V_old.copy()
        for s in range(nS):
            sa_vector = np.zeros(nA)
            for a in range(nA):
                for tuple_info in P[s][a]:
                    p,s_,u,_ = tuple_info

                    sa_vector[a] += (p * (u + beta * V_old[s_]))
            V_new[s] = np.max(sa_vector)
        if np.linalg.norm(V_new - V_old) < tol:
            return V_new, k
        V_old = V_new
    
    return V_new, k

# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    policy_vector = np.zeros(nS)
    for s in range(nS):
        sa_vector = np.zeros(nA)
        for a in range(nA):
            for tuple_info in P[s][a]:
                p,s_,u,_ = tuple_info
                sa_vector[a] += (p*(u+beta*v[s_]))
        policy_vector[s] = np.argmax(sa_vector)
    return policy_vector

# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8, maxiter = 3000):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    V_old = np.zeros(nS)
    for k in range(maxiter):
        V_new = V_old.copy()
        for s in range(nS):
            sa_value = 0
            a = policy[s]
            for tuple_info in P[s][a]:
                p,s_,u,_ = tuple_info
                sa_value += (p*(u+beta*V_old[s_]))
            V_new[s] = sa_value
        if np.linalg.norm(V_new - V_old) < tol:
            return V_new,k
        V_old = V_new

    return V_new,k

# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """
    policy = np.zeros(nS)
    for k in range(maxiter):
        value,n = compute_policy_v(P,nS,nA,policy,beta,tol)
        new_policy = extract_policy(P,nS,nA,value,beta)
        if np.linalg.norm(new_policy - policy) < tol:
            break
        policy = new_policy
    return value,policy,k

# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """
    if basic_case:
        num_states = 16
        env_name = 'FrozenLake-v1'
    else:
        num_states = 64
        env_name = 'FrozenLake8x8-v1'
    vi_mean_reward_potatoes = 0
    pi_mean_reward_potatoes = 0
    vi_reward_potatoes = []
    pi_reward_potatoes = []
    with gym.make(env_name) as env:
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        dict_P = env.P
        pi_value,pi_policy,pi_k = policy_iteration(dict_P,num_states,num_actions)
        vi_value,vi_k = value_iteration(dict_P,num_states,num_actions)
        vi_policy = extract_policy(dict_P,num_states,num_actions,vi_value)

        for i in range(M):
            print(f'{i}/{M} ({100*i/M:.1f}%)', end = "\r")
            vi_reward_potatoes.append(run_simulation(env,vi_policy,render))
            pi_reward_potatoes.append(run_simulation(env,pi_policy,render))
    
    vi_mean_reward_potatoes = np.mean(vi_reward_potatoes)
    pi_mean_reward_potatoes = np.mean(pi_reward_potatoes)
    return vi_policy,vi_mean_reward_potatoes, pi_value, pi_policy, pi_mean_reward_potatoes

# Problem 6
def run_simulation(env, policy, render=True, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    obs = env.reset()
    done = False
    total_reward = 0
    counter = 0
    while not done:
        env.render(mode='human')
        obs,reward,done,_ = env.step(int(policy[obs]))
        total_reward+= beta**counter*reward
        counter += 1
    return total_reward



if __name__ == "__main__":
    print("PROBLEM 1:\n")
    v,k = value_iteration(P,4,4)
    print(v)

    print("\n\nPROBLEM 2:\n")
    policy = extract_policy(P,4,4,v)
    print(policy)

    print("\n\nPROBLEM 3:\n")
    value = compute_policy_v(P,4,4,policy)
    print(value)

    print("\n\nPROBLEM 4:\n")
    value,policy,k = policy_iteration(P,4,4)
    print(f'value: {value}')
    print(f'policy: {policy}')

    # print("\n\nPROBLEM 5:\n")
    # print(frozen_lake())

    print("\n\nPROBLEM 6:\n")
    print("4x4:")
    vi_policy,vi_mean_reward, pi_value,pi_policy,pi_mean_reward = frozen_lake()
    print(f"vi_policy: {vi_policy}")
    print(f'vi_mean_reward: {vi_mean_reward}')
    print(f'pi_value: {pi_value}')
    print(f'pi_policy: {pi_policy}')
    print(f'pi_mean_reward: {pi_mean_reward}')
