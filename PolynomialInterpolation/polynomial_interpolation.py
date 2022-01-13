# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
<Name>
<Class>
<Date>
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator
from scipy.fftpack import fft
import os

def runge(domain):
    return 1/(1+25*domain**2)

# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    def compute_demon(x,j):
        product_array = [x-xint[k] for k in range(len(xint))]
        return np.product(np.delete(product_array, j))
    # compute the Lj's 
    demons = [compute_demon(xint[j], j) for j in range(len(xint))]
    demons = np.array(demons)

    mat = []
    for point in points:
        submat = [compute_demon(point, j) for j in range(len(xint))]
        submat = submat/demons
        mat.append(submat)
    
    mat = np.array(mat)


    return mat@yint


# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        self.xint = xint
        self.yint = yint
        def bary_weights(points):
            """Computes the barycentric weights for a given set of distinct points
            {x_0,...,x_n} (the elements of parameter x).
            """
            weights = []
            for j in range(len(points)):
                wj = 1
                for k in range(len(points)):
                    if j == k: continue
                    wj *= points[j] - points[k]
                weights.append(1/wj)
            return weights
        self.weights = np.array(bary_weights(xint))


    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        numerator = 0
        denominatorinator = 0
        for j in range(len(self.xint)):
            numerator += self.yint[j] * self.weights[j]/(points-self.xint[j])
            denominatorinator += self.weights[j]/(points-self.xint[j])
        
        return numerator / denominatorinator

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """

        # make them lists so I can pop them off
        xint = list(xint)
        yint = list(yint)

        # make them lists so I can append
        self.xint = list(self.xint)
        self.weights = list(self.weights)
        self.yint = list(self.yint)

        while xint:
            self.yint.append(yint.pop())
            i = xint.pop()
            self.weights = [self.weights[j] / (self.xint[j] - i) for j in range(len(self.weights))]
            new = 1/np.product([(i-k) for k in self.xint])
            self.weights.append(new)
            self.xint.append(i)
        

        # convert everything back to np arrays
        self.weights = np.array(self.weights)
        self.xint = np.array(self.xint)
        self.yint = np.array(self.yint)

        
        


# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    domain = np.linspace(-1,1, 400)
    rung = runge(domain)
    power = 2**np.array([2,3,4,5,6,7,8])
    equal_err = []
    cheby_err = []
    for n in power:
        points = np.linspace(-1,1,n)
        interpol = BarycentricInterpolator(points, runge(points))
        # print(np.linalg.norm(interpol(domain) - rung, ord=np.inf))
        equal_err.append(np.linalg.norm(interpol(domain) - rung, ord=np.inf))
        # plt.plot(domain, interpol(domain), 'b')

        cheby_points = np.array([np.cos(j*np.pi/n) for j in range(n)])
        interpol = BarycentricInterpolator(cheby_points, runge(cheby_points))
        cheby_err.append(np.linalg.norm(interpol(domain) - rung, ord = np.inf))
        # plt.plot(domain, rung, 'r')
        # plt.plot(domain, interpol(domain), 'k')
        # plt.show()

    plt.loglog(power, equal_err, base = 2)
    plt.loglog(power, cheby_err, base = 2)
    plt.xlabel("number of points")
    plt.ylabel("error")
    plt.show()


# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    cheby_points = [np.cos(j*np.pi/n) for j in range(n+1)]
    for p in (cheby_points[-2:0:-1]):
        cheby_points.append(p)
    cheby_points = np.array(cheby_points)
    ak = 1/(2*n) * fft(f(cheby_points)).real
    ak[1:n] *= 2
    return ak[:n+1]



# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    fx = lambda a,b,n: .5*(a+b + (a-b) * np.cos(np.arange(n+1) * np.pi / n))
    
    data = np.load("airdata.npy")
    print(len(data))
    a,b = 0,366-1/24
    domain = np.linspace(a,b, 8784)
    points = fx(a,b,n)
    temp = np.abs(points - domain.reshape(8784,1))
    temp2 = np.argmin(temp, axis = 0)

    poly = BarycentricInterpolator(domain[temp2], data[temp2])
    plt.plot(domain, data, 'r')
    plt.plot(domain, poly(domain), 'k')
    plt.show()
    raise NotImplementedError("Problem 7 Incomplete")


if __name__ == "__main__":
    os.chdir("/Users/chase/Desktop/Math321Volume2/byu_vol2/PolynomialInterpolation")
    # xint = np.linspace(-1,1,11)
    # print(xint)
    # yint = runge(xint)
    # points = np.linspace(-.99,.99,1000)

    # # plt.plot(points, lagrange(xint, yint, points))
    # plt.plot(points, runge(points))
    # # plt.show()

    # b = Barycentric(xint[:6], yint[:6])
    # plt.plot(points, b(points))
    # plt.show()



    # new_whys = runge(xint[6:])
    # b.add_weights(xint[6:], new_whys)
    # plt.plot(points, b(points), 'k')
    # plt.plot(points, runge(points))
    # plt.show()

    # prob5()

    # f = lambda x: -3 + 2*x**2 - x**3 + x**4
    # pcoeffs = [-3,0,2,-1,1]
    # ccoeffs = np.polynomial.chebyshev.poly2cheb(pcoeffs)
    # print(f'np: {ccoeffs}')

    # print(f'mine: {chebyshev_coeffs(f, 4)}')

    prob7(75 )