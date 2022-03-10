# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name>
<Class>
<Date>
"""
import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(3,nonneg = True)
    c = np.array([2,1,3])
    objective = cp.Minimize(c.T @ x)
    A = np.array([  [1,2,0],
                    [0,2,-4],
                    [-2,-10,-3]])
    b = np.array([3,1,-12])
    P = np.eye(3)
    constraints = [A@x <= b, P@x >= 0]
    problem = cp.Problem(objective, constraints)
    opium = problem.solve()
    return np.ravel(x.value), opium

# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    n = A.shape[1]
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x,1))
    constraints = [A@x == b]
    problem = cp.Problem(objective, constraints)
    opium = problem.solve()
    return np.ravel(x.value), opium

# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(6,nonneg = True)
    c = np.array([4,7,6,8,8,9])
    objective = cp.Minimize(c.T @ x)
    Supply = np.array([ [1,1,0,0,0,0],
                        [0,0,1,1,0,0],
                        [0,0,0,0,1,1]])
    sup_remum_ = np.array([7,2,4])

    Demand = np.array([ [1,0,1,0,1,0],
                        [0,1,0,1,0,1]])
    dem_apples = np.array([5,8])
    
    P = np.eye(6)
    constraints = [Supply@x == sup_remum_, Demand@x == dem_apples, P@x >= 0]
    problem = cp.Problem(objective, constraints)
    opium = problem.solve()
    return np.ravel(x.value), opium

# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(3)
    Q = np.array([  [3,2,1],
                    [2,4,2],
                    [1,2,3]])
    r = np.array([3,0,1])
    problem = cp.Problem(cp.Minimize(.5 * cp.quad_form(x,Q) + r.T@x))
    opium = problem.solve()
    return x.value, opium

# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    n = A.shape[1]
    x = cp.Variable(n, nonneg = True)
    I = np.eye(n)
    one = np.ones(n)
    objective = cp.Minimize(cp.norm(A@x - b, 2))
    constraints = [one@x == 1, I@x >= 0]
    problem = cp.Problem(objective, constraints)
    opium = problem.solve()
    return x.value, opium


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    dat = np.load('food.npy', allow_pickle=True)
    price = dat[:,0]
    servings_per_container = dat[:,1]
    calories_per_container = servings_per_container * dat[:,2]
    fat_per_container = servings_per_container * dat[:,3]
    sugar_per_container = servings_per_container * dat[:,4]
    calcium_per_container = servings_per_container * dat[:,5]
    fiber_per_container = servings_per_container * dat[:,6]
    protien_per_container = servings_per_container * dat[:,7]
    I = np.eye(len(price))
    x = cp.Variable(len(price), nonneg = True)
    objective = cp.Minimize(price.T@x)
    constraints = [calories_per_container.T@x <= 2000, fat_per_container.T@x <= 65, sugar_per_container.T@x <= 50, calcium_per_container.T@x >= 1000, fiber_per_container.T@x >= 25, protien_per_container.T@x >= 46, I@x >= 0]
    problem = cp.Problem(objective, constraints)
    opium = problem.solve()
    return x.value, opium





if __name__ == "__main__":
    print("Problem 1:")
    minimizer, value = prob1()
    print(f'minimizer: {minimizer}\nvalue: {value}\n')

    print("Problem 2:")
    A = np.array([  [1,2,1,1],
                    [0,3,-2,-1]])
    b = np.array([7,4])
    minimizer, value = l1Min(A,b)
    print(f'minimizer: {minimizer}\nvalue: {value}\n')

    print("Problem 3:")
    minimizer, value = prob3()
    print(f'minimizer: {minimizer}\nvalue: {value}\n')

    print("problem 4:")
    minimizer, value = prob4()
    print(f'minimizer: {minimizer}\nvalue: {value}\n')

    print("problem 5:")
    minimizer, value = prob5(A,b)
    print(f'minimizer: {minimizer}\nvalue: {value}\n')

    print("Problem 6:")
    minimizer, value = prob6()
    print(f'minimizer: {minimizer}\nvalue: {value}\n')
    foods = ["ramen", 'potatoes', 'milk', 'eggs', 'pasta', 'frozen pizza', 'potato chips', 'frozen broccoli', 'carrots', 'bananas', 'tortillas', 'cheese', 'yogurt', 'bread', 'chicken', 'rice', 'pasta sauce', 'lettuce']
    food_dict = {i:food for i, food in enumerate(foods)}
    print('Foods sorted from best to worst:')
    for i in np.argsort(minimizer)[::-1]:
        print("\t", food_dict[i])



