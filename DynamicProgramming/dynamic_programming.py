# dynamic_programming.py
"""Volume 2: Dynamic Programming.
<Name>
<Class>
<Date>
"""

from email import policy
#from cvxpy import CvxAttr2Constr
import matplotlib.pyplot as plt
import numpy as np
import time


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    # initialize values with the 0 percent chance of choosing the final candidate correctly
    values = [0]
    t0 = None
    for i in range(N-1,0,-1):
        values.append(max(i*values[-1]/(i+1) + 1/N, values[-1]))
        if not t0 and values[-1] > i/N:
            t0 = i
    
    return np.max(values), t0


# Problem 2
def graph_stopping_times(M, plot = True):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    num_candidates = np.arange(3,M+1)
    opt_stop_perc = []
    probs = []
    for i in num_candidates:
        # print(f"iteration {i}", end = "\r")
        value, t0 = calc_stopping(i)
        opt_stop_perc.append(t0/i)
        probs.append(value)
    if plot:
        plt.plot(num_candidates, opt_stop_perc, label = "Optimal Stopping Percentage")
        plt.plot(num_candidates, probs, label = "Probabilities")
        plt.title(f"Optimal Stopping Percentage for Number of Candidates up to {M}")
        plt.xlabel("Number of Candidates")
        plt.ylabel("Optimal Stopping Percentage")
        plt.legend()
        plt.show()
    # print()

    return opt_stop_perc[-1]


# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    w = np.array([i/N for i in range(0,N+1)])
    C = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(i+1):
            C[i,j] = u(w[i] - w[j])
    return C


# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    A = np.zeros((N+1,T+1))
    P = np.zeros((N+1, T+1))
    w = np.array([i/N for i in range(0,N+1)])
    P[:,T] = w
    for i in range(N+1):
        A[i,T] = u(i/N)
    
    for t in range(T-1, -1, -1):
        continuously_variable_transmission = np.zeros((N+1, N+1))
        for i in range(N+1):
            for j in range(i+1):
                continuously_variable_transmission[i,j] = u(w[i] - w[j]) + B*A[j,t+1]
        A[:,t] = np.max(continuously_variable_transmission, axis=1)
        for i in range(N+1):
            P[i,t] = w[i] - w[np.argmax(continuously_variable_transmission[i,:])]
    return A,P


# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    A,P = eat_cake(T,N,B,u)
    w = np.array([i/N for i in range(0,N+1)])
    pieces_left = N
    policy_vect = []
    for t in range(len(P[0])):
        # print(f'time t={t}')
        policy_vect.append(P[pieces_left, t])
        # print(f"\tate {policy_vect[-1]} of the cake")
        num_pieces = np.argmin(np.abs(w-P[pieces_left,t]))
        # print(f"\tate {num_pieces} pieces of cake")
        pieces_left -= num_pieces
        
    
    return np.array(policy_vect)


if __name__ == "__main__":
    print("PROBLEM 1:")
    print(calc_stopping(67))

    print("\n\nPROBLEM 2:")
    start = time.time()
    print(graph_stopping_times(1000))
    print(f'this took {time.time() - start:.3f} seconds.')

    print("\nPROBLEM 3:")
    print(get_consumption(10, lambda x: x))

    T = 3
    N = 6
    B = .75
    u = lambda x: np.sqrt(x)
    betas = np.array([B**i for i in range(T+1)])

    if N<1000:
        print("\nPROBLEMS 4-6:")
        A,P = eat_cake(T,N,B,u)
        print(A)
        print(P)
        max_value = np.max(A)

    print("\nPROBLEM 7:")
    policy_vect = find_policy(T,N,B,u)
    print(policy_vect)
    found_max_value = betas.T@u(policy_vect)
    if N<1000:
        print(f'max value from value matrix:\t{max_value:.5f}')
    print(f'max value from policy:\t\t{found_max_value:.5f}')

