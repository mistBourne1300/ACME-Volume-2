# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
<Name>
<Class>
<Date>
"""

from re import search
from tracemalloc import start
import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    m,n = A.shape
    def get_F(x,lamb,mu):
        first_comp = A.T@lamb + mu - c
        second_comp = A@x-b
        third_comp = np.diag(mu)@x
        return np.concatenate((first_comp,second_comp,third_comp))

    def DF_top():
        zero11 = np.zeros((n,n))
        zero22 = np.zeros((m,m))
        zero23 = np.zeros((m,n))
        I = np.eye(n)
        col1 = np.vstack((zero11,A))
        col2 = np.vstack((A.T,zero22))
        col3 = np.vstack((I,zero23))
        return np.hstack((col1,col2,col3))
    
    derivative_of_F_top = DF_top()

    def DF_bottom(mu,x):
        M = np.diag(mu)
        X = np.diag(x)
        zero32 = np.zeros((n,m))
        return np.hstack((M,zero32,X))

    
    def search_duality(F,x,mu):
        nu = x.T@mu/n
        DF = np.vstack((derivative_of_F_top,DF_bottom(mu,x)))
        right_right_side = np.concatenate((np.zeros(n),np.zeros(m),.1*nu*np.ones(n)))
        right_side = -F + right_right_side
        delta = la.lu_solve(la.lu_factor(DF),right_side)
        return delta, nu
    
    def step_length(delta,x,mu):
        alpha_max = np.min(-mu/delta[-n:])
        if alpha_max<0: alpha_max = 1
        alpha_max = np.min([1,0.95*alpha_max])

        delta_max = np.min(-x/delta[:n])
        if delta_max < 0: delta_max = 1
        delta_max = np.min([1, 0.95*delta_max])
        print("alpha_max, delta_max")
        print(alpha_max, delta_max)
        return alpha_max, delta_max

    x,lamb,mu = starting_point(A,b,c)
    for i in range(niter):
        F = get_F(x,lamb,mu)
        delta,nu = search_duality(F,x,mu)
        alpha_max, delta_max = step_length(delta, x, mu)
        x += delta_max*delta[:n]
        lamb += alpha_max*delta[n:-n]
        mu += alpha_max*delta[-n:]
        if np.abs(nu)<tol:
            return x, c.T@x
        
    return x,c.T@x


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    x,y = [],[]
    with open(filename) as file:
        for line in file.readlines():
            dat = line.strip.split()
            y.append(dat[0])
            x.append(dat[1])
    
    


    raise NotImplementedError("Problem 5 Incomplete")


if __name__ == "__main__":
    j,k = 3,3
    A,b,c,x = randomLP(j,k)
    point, value = interiorPoint(A,b,c)
    print(f'optimal x: {x}\nreturned x: {point[:k]}')
    print(f'close?: {np.allclose(x,point[:k])}')
    print(f'optimal value: {c[:k].T@x}\nreturned value: {value}')

    leastAbsoluteDeviations()