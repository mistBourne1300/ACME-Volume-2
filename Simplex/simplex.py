"""Volume 2: Simplex

<Name>
<Date>
<Class>
"""

from xmlrpc.server import SimpleXMLRPCDispatcher
import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        if np.min(b) < 0:
            raise ValueError("problem is not feasible at the origin")
        self.c = c
        self.A = A
        self.b = b
        self.m,self.n = A.shape
        self.D = self._generatedictionary(c,A,b)
        

    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        _A_ = np.hstack((A,np.identity(self.m)))
        _c_ = np.concatenate((c, np.zeros(self.m)))
        Dtop = np.hstack(([0], _c_.T))
        Dbottom = np.hstack((b.reshape(-1,1), -_A_))
        return np.vstack((Dtop, Dbottom))


    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        for i in range(1,len(self.D[0])):
            if self.D[0][i] < 0:
                return i
        return -1

    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        if np.min(self.D[1:,index]) >= 0:
            raise RuntimeError("problem is unbounded, no solution exists")
        
        mask = self.D[1:,index] >= 0 
        ratios = -self.D[1:,0] / self.D[1:,index]
        ratios[mask] = np.inf
        return np.argmin(ratios) +1

    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        j = self._pivot_col()
        i = self._pivot_row(j)
        self.D[i,:] /= -self.D[i,j]

        for k in range(len(self.D)):
            if k == i: continue
            self.D[k,:] += self.D[k,j]*self.D[i,:]


    # Problem 5
    def solve(self, v = False):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        while np.min(self.D[0,1:]) < 0:
            if v: print(self.D, end = "\n\n")
            self.pivot()
        if v: print(self.D)
        # min_val = self.D[0,0]
        independent = dict()
        dependent = dict()
        for j in range(1,len(self.D[0])):
            if self.D[0,j] == 0:
                # dependent
                index = self.D[:,j].tolist().index(-1)
                # print(index)
                dependent[j] = self.D[index,0]
            elif self.D[0,j] > 0:
                # independent
                independent[j] = 0
            else:
                raise RuntimeError("problem solving failed")
        return self.D[0,0], dependent, independent

# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    dat = np.load('productMix.npz')
    _A_ = dat['A']
    m,n = _A_.shape
    p = dat['p']
    m = dat['m']
    d = dat['d']

    A = np.vstack((_A_, np.identity(n)))
    b = np.concatenate((m,d))
    c = -p

    minval, dependent, independent = SimplexSolver(c,A,b).solve()
    prod_soln = np.zeros(len(d))
    max_index = len(d) - 1
    for i in dependent.keys():
        if i > max_index: continue
        prod_soln[i] = dependent[i]
    for i in independent.keys():
        if i > max_index: continue
        prod_soln[i] = 0
    return prod_soln

if __name__ == "__main__":
    A = np.array([[1,-1],[3,1],[4,3]])
    c = np.array([-3,-2])
    b = np.array([2,5,7])

    simplexsolverclassthing = SimplexSolver(c,A,b)
    # print(simplexsolverclassthing.D)
    # print(simplexsolverclassthing._pivot_col(), simplexsolverclassthing._pivot_row(1))
    print(simplexsolverclassthing.solve(v=True))
    # print(simplexsolverclassthing.D)
    # print("\n\n\n\n")
    print(prob6())
