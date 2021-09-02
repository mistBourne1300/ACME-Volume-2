# python_intro.py
"""Python Essentials: Introduction to Python.
<Name>
<Class>
<Date>
"""

import numpy as np

#import matplotlib as mat

#Problem 1
#will print the first three (a,b,c) separated by 5 spaces, then the last two (d,e) separated by one space
def isolate(a, b, c, d, e):
    print(a, b, c, sep = "     ", end = " ")
    print(d, e)

#Problem 2
def first_half(string):
    return string[:int(len(string)/2)]
    #raise NotImplementedError("Problem 2 Incomplete")


def backward(first_string):
    return first_string[::-1]

#Problem 3
#will initailize a list, then append eagle
def list_ops():
    ops_list = ["bear", 'ant', 'cat', 'dog']
    #print(ops_list)
    ops_list.append("eagle")
    #print(ops_list)
    ops_list[2] = "fox"
    #print(ops_list)
    ops_list.pop(1)
    #print(ops_list)
    ops_list.sort()
    #print(ops_list)
    ops_list.reverse()
    #print(ops_list)
    ops_list[ops_list.index("eagle")] = "hawk"
    #print(ops_list)
    ops_list.append("hunter")
    #print(ops_list)
    return ops_list

#Problem 4
def alt_harmonic(n):
    """Return the partial sum of the first n terms of the alternating
    harmonic series. Use this function to approximate ln(2).
    """
    return sum([((-1)**(i+1)/i) for i in range(1, n+1)])



def prob5(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.
    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    A_copy = np.copy(A)
    mask = A_copy < 0
    A_copy[mask] = 0
    return A_copy

def prob6():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.array([0,2,4,1,3,5]).reshape((2,3))
    B = np.array([3,0,0,3,3,0,3,3,3]).reshape((3,3))
    C = -2 * np.identity(3)
    I = np.identity(3)
    #print(A, B, C, sep = "\n\n", end = "\n\n\n")

    left = np.vstack((np.zeros((3,3)), A, B))
    #print("left", left, end = "\n\n")
    mid = np.vstack((A.T, np.zeros((2,2)), np.zeros((3,2))))
    #print("mid:", mid, end="\n\n")
    right = np.vstack((I, np.zeros((2,3)), C))
    #print(left, mid, right, sep = "\n\n\n", end = "\n\n\n")

    return np.hstack((left,mid,right))

def prob7(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    return A / A.sum(axis = 1).reshape((-1,1))


def prob8():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """

    grid = np.load("grid.npy")
    #print(grid, end="\n\n\n")
    #grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:]
    #finds all the horizontal products

    #grid[:-3,:] * grid[1:-2, :] * grid[2:-1, :] * grid[3:, :]
    #find all the vertical products

    #grid[:-3, :-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:, 3:]
    #find right diagonal products

    #print(grid[3:, :-3].size, grid[2:-1, 1:-2].size, grid[1:-2, 2:-1].size, grid[:-3, 3:].size, end="\n\n")
    #finds all left diagonal products

    maxes = []
    maxes.append(np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:]))
    maxes.append(np.max(grid[:-3,:] * grid[1:-2, :] * grid[2:-1, :] * grid[3:, :]))
    maxes.append(np.max(grid[:-3, :-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:, 3:]))
    maxes.append(np.max(grid[3:, :-3] * grid[2:-1, 1:-2] * grid[1:-2, 2:-1] * grid[:-3, 3:]))

    #print(maxes)
    return np.max(maxes)







    #print("\n\n\n\n\n")
    return None
    raise NotImplementedError("Problem 8 Incomplete")



'''isolate(1,2,3,4,5)
print(first_half("hello"))
print(backward("Hello!"))
print(list_ops())
print(alt_harmonic(500000))
print(prob5(np.array([1,-2,3,-4,5,-6,7,-8,9,-10])))
print(prob6())
print(prob7(np.array([[1,1,0],[0,1,0],[1,1,1]])))
print(prob8())'''