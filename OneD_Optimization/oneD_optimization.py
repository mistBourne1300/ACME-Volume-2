# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy.optimize import golden, newton
from scipy.optimize.linesearch import scalar_search_armijo as ssa
from autograd import numpy as anp
from autograd import grad
import matplotlib.pyplot as plt

# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=15):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    x0 = (a+b)/2
    midas = (1+np.sqrt(5))/2
    for i in range(maxiter):
        c = (b-a)/midas
        _a = b-c
        _b = a+c
        if f(_a) <= f(_b):
            b = _b
        else:
            a = _a
        
        x1 = (a+b)/2
        if np.abs(x0 - x1) < tol:
            return x1, True, i+1
        x0 = x1
    
    return x1, False, maxiter



# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=15):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """

    for i in range(maxiter):
        x1 = x0 - df(x0)/d2f(x0)
        if np.abs(x0 - x1) < tol:
            return x1, True, i+1
        x0 = x1
    
    return x1, False, maxiter


# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=15):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    for i in range(maxiter):
        x2 = (x0*df(x1) - x1*df(x0))/(df(x1) - df(x0))
        if np.abs(x2 - x1) < tol:
            return x2, True, i+1
        x0 = x1
        x1 = x2
    
    return x2, False, maxiter


# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    dfp = Df(x).T@p
    fx = f(x)
    while (f(x+alpha*p) > fx + c*alpha*dfp):
        alpha = rho*alpha

    return alpha

if __name__ == "__main__":
    f = lambda x: np.exp(x) - 4*x
    print("\n\nPROBLEM 1:")
    print(f'mine: {golden_section(f, 0, 3, tol = .001, maxiter=50)}')
    print(f'scipy: {golden(f, brack=(0,3), tol = .001)}')

    df = lambda x: 2*x + 5*np.cos(5*x)
    d2f = lambda x: 2 - 25*np.sin(5*x)

    print("\n\nPROBLEM 2:")
    print(f'mine: {newton1d(df, d2f, 0, tol = 1e-10, maxiter=500)}')
    print(f'scipy: {newton(df, x0 = 0, fprime = d2f, tol = 1e-5, maxiter = 500)}')
    
    
    f = lambda x: x**2 + np.sin(x) + np.sin(10*x)
    df = lambda x: 2*x + np.cos(x) + 10*np.cos(10*x)
    print("\n\nPROBLEM 3:")
    print(f'mine: {secant1d(df, 0,-1,tol=1e-10,maxiter=500)}')
    print(f'scipy: {newton(df, x0 = 0, tol=1e-10, maxiter=500)}')
    # domain = np.linspace(-1.5, 1.5, 1000)
    # plt.plot(domain, f(domain))
    # plt.show()

    f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
    df = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])

    x = anp.array([150., .03, 40.])
    p = anp.array([-.5, -100., -4.5])
    phi = lambda alpha: f(x + alpha*p)
    dphi = grad(phi)    
    alpha, _ = ssa(phi, phi(0.), dphi(0.))

    print("\n\nPROBLEM 4:")
    print(f'mine: {backtracking(f, df, x, p)}')
    print(f'scipy: {alpha}')
