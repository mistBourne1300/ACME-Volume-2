# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<Name>
<Class>
<Date>
"""
import os
import numpy as np
from numpy.random.mtrand import negative_binomial
from scipy import linalg as la
from scipy.spatial import KDTree
from scipy import stats

# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    min_dist = min(la.norm(X-z, axis=1))
    element = np.argmin(la.norm(X-z, axis=1))
    # print(min_dist, element)
    return min_dist, X[element]


# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        if(type(x) != np.ndarray):
            raise TypeError("x is not an ndarray")

        self.value = x
        self.left = None
        self.right = None
        self.pivot = None

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None
    
    def __str__(self):
        """
            String representation: a hierarchical list of nodes and their axes.
        """

        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        new_node = KDTNode(data)
        if not self.root:
            new_node.pivot = 0
            self.root = new_node
            self.k = len(data)
            return
        if(len(data) != self.k):
            raise ValueError("length of data does not match k")
        current = self.root
        # while loop finds the node that is the parent of the data to be inserted
        while current:
            # if the data is a duplicate, raise value error
            if np.allclose(current.value, data): raise ValueError("data cannot be duplicate")
            # if the data at the pivot is greater, either insert it as the right child, 
            # or move curr_node to the right child
            if data[current.pivot] >= current.value[current.pivot]:
                if not current.right:
                    current.right = KDTNode(data)
                    current.right.pivot = (current.pivot+1)%self.k
                    return
                else:
                    current = current.right
            else:
                if not current.left:
                    current.left = KDTNode(data)
                    current.left.pivot = (current.pivot+1)%self.k
                    return
                else:
                    current = current.left
        # raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        def KDSearch(current, nearest, dist):
            if current is None:
                return nearest, dist
            x = current.value
            i = current.pivot
            if np.linalg.norm(x - z) < dist:
                nearest = current
                dist = np.linalg.norm(x - z)
            if z[i] < x[i]:
                nearest, dist = KDSearch(current.left, nearest, dist)
                if z[i] + dist >= x[i]:
                    nearest, dist = KDSearch(current.right, nearest, dist)
            else:
                nearest, dist = KDSearch(current.right, nearest, dist)
                if z[i] <= x[i]:
                    nearest, dist = KDSearch(current.left, nearest, dist)
            return nearest, dist
        node, dist = KDSearch(self.root, self.root, np.linalg.norm(self.root.value - z))
        return node.value, dist
        raise NotImplementedError("Problem 4 Incomplete")

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self,n_neighbors):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.labels = y
        self.aspen = KDTree(X)
    
    def predict(self, z):
        distances, indices = self.aspen.query(z, k=self.n_neighbors)
        return stats.mode(self.labels[indices])[0][0]

# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load(filename)
    X_train = data["X_train"].astype(float)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(float)
    y_test = data["y_test"]
    recognizer = KNeighborsClassifier(n_neighbors)
    recognizer.fit(X_train, y_train)
    predictions = [recognizer.predict(x) for x in X_test]
    return np.mean([predictions[i] == y_test[i] for i in range(len(y_test))])
    raise NotImplementedError("Problem 6 Incomplete")


if __name__ == "__main__":
    # X = np.array(np.random.random((4,3)))
    z = np.array(np.random.random(3))
    # # print(X)
    print(f'z: {z}')
    # # print(f'type of z: {type(z)} type of np.ndarray: {np.ndarray}')
    # # print(exhaustive_search(X, z))
    # node = KDTNode(z)
    # print(node.value)

    data = np.array([[1,1,1],[0,0,5],[1,2,3],[4,3,5],[3,5,2],[6,2,5],[6,2,4],[9,5,7],[25,72,3],[4,1,6],[72,6,3]])
    maple = KDT()
    for datum in data:
        #print(datum)
        maple.insert(datum)
    print(maple.find(np.array([72,6,3])).value)
    print(maple.query(z))

    # Problem 5

    y = np.array([1,1,1,1,1,0,0,0,0,1,0])
    prob5 = KNeighborsClassifier(2)
    prob5.fit(data, y)
    print(f'prob5.predict([0,0,0]) == {prob5.predict([4,0,0])}')

    # problem 6
    os.chdir("/Users/chase/Desktop/Math321Volume2/byu_vol2/NearestNeighbor")
    print(f'prob 6 accuracy: {prob6(4)}')
    pass
    
