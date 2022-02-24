# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
<Name>
<Class>
<Date>
"""
import numpy as np
import scipy.optimize as opium
from autograd import elementwise_grad, grad
from scipy import linalg as la
import os
from autograd import numpy as anp
import matplotlib.pyplot as plt

# Problem 1
def steepest_descent(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # while we still have iterations to use
    for i in range(maxiter):
        df0 = df(x0)
        phi = lambda a: f(x0 - a*df0.T)
        # use scipy's line search
        alpha = opium.minimize_scalar(phi).x
        x1 = x0 - alpha*df0.T
        # if the point is close enough, return
        if la.norm(df0.T) < tol:
            return x1, True, i

        x0 = x1
        
    return x1, False, i


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    n = Q.shape[0]
    r0 = Q@x0 - b
    d0 = -r0
    k = 0
    while la.norm(r0) > tol and k < n:
        ak = (r0.T@r0)/(d0.T@Q@d0)
        x1 = x0 + ak*d0
        r1 = r0 + ak*Q@d0
        beta = (r1.T@r1)/(r0.T@r0)
        d1 = -r1 + beta*d0
        k += 1

        r0 = r1
        x0 = x1
        d0 = d1
    return x1, (k <= n), k


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    r0 = -df(x0).T
    d0 = r0
    fun = lambda a: f(x0 + a*d0)
    a0 = opium.minimize_scalar(fun).x
    x1 = x0 + a0*d0
    x0 = x1
    k = 1
    while la.norm(r0) >= tol and k < maxiter:
        r1 = -df(x0).T
        beta = (r1.T@r1)/(r0.T@r0)
        d1 = r1 + beta*d0
        fun = lambda a: f(x0 + a*d1)
        a0 = opium.minimize_scalar(fun).x
        x1 = x0 + a0*d1
        k += 1

        r0 = r1
        d0 = d1
        x0 = x1

    return x1, (k < maxiter), k


# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    dat = np.loadtxt("linregression.txt")
    y = dat[:,0]
    A = np.ones_like(dat)
    A[:,1:] = dat[:,1:]
    Q = A.T@A
    newbie = A.T@y
    return conjugate_gradient(Q,newbie,x0)[0]



# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        neg_log_like = lambda b: anp.sum([anp.log(1+anp.exp(-(b[0] + b[1]*x[i]))) + (1-y[i])*(b[0]+b[1]*x[i]) for i in range(len(x))])
        beta = nonlinear_conjugate_gradient(neg_log_like, grad(neg_log_like), guess)[0]
        self.b0 = beta[0]
        self.b1 = beta[1]

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        return 1/(1+np.exp(-(self.b0 + self.b1*x)))


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    dat = np.load("challenger.npy")
    temperature = dat[:,0]
    damage = dat[:,1]
    infinite_regret = LogisticRegression1D()
    infinite_regret.fit(temperature, damage, guess)
    domain = np.linspace(30,100)
    plt.plot(domain, infinite_regret.predict(domain), color = 'goldenrod')
    plt.plot(temperature, damage, 'bo')
    plt.plot(31., infinite_regret.predict(31.), 'ro')
    plt.legend(['P(Damage) at launch', 'Previous Damage', 'Damage at 31F'])
    plt.xlabel("Temperature")
    plt.ylabel("O-Ring Damage")
    plt.title("Probability of O-Ring Damage")

    plt.show()

    return infinite_regret.predict(31.)


if __name__ == "__main__":
    # # PROBLEM 1
    # f1 = lambda x: x[0]**4 + x[1]**4 + x[2]**4
    # rosy = lambda x: 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

    # x01 = np.array([1.,1.,1.])
    # x0rosy = np.array([-2.,2.])
    # print(steepest_descent(f1, grad(f1), x01))
    # print(steepest_descent(rosy, grad(rosy), x0rosy, maxiter = int(1e7)))

    # # PROBLEM 2
    # Q = np.array(   [[2.,0.],
    #                 [0.,4.]])
    # b = np.array([1,8])
    # x0 = np.array([1,1])
    # print(conjugate_gradient(Q,b,x0))
 
    # # PROBLEM 3
    # print(nonlinear_conjugate_gradient(rosy, grad(rosy), x0rosy, maxiter = 500))

    # PROBLEM 4
    os.chdir("/Users/chase/Desktop/Math321Volume2/byu_vol2/GradientMethods")
    print(prob4())

    # # PROBLEMS 5 and 6
    # print(prob6())
