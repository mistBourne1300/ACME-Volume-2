# binary_trees.py
"""Volume 2: Binary Trees.
<Name>
<Class>
<Date>
"""

# These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from numpy.core.numeric import tensordot
from numpy.lib import stride_tricks
from numpy.random.mtrand import rand, random


class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        # method for recursively finding a node
        def check_node_data(node):
            if node == None:
                raise ValueError("Data not found in list")
            elif node.value == data:
                return node
            return check_node_data(node.next)
        
        # call the above method, and return the node it found
        return check_node_data(self.head)
        raise NotImplementedError("Problem 1 Incomplete")


class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        n = BSTNode(data)
        if self.root == None:
            self.root = n
            return 
        
        current = self.root
        while current:
            if n.value > current.value:
                if not current.right:
                    current.right = n
                    n.prev = current
                    return
                else:
                    current = current.right
            elif n.value < current.value:
                if not current.left:
                    current.left = n
                    n.prev = current
                    return
                else:
                    current = current.left
            else: # the data is a duplicate. not allowed
                raise ValueError("cannot input duplicate data")
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        # get the node containing the data to be removed
        target = self.find(data)

        def find_immediate_predecessor(node):
            if not node.left:
                raise RuntimeError(f'node {node.value} has no left child')
            current = node.left
            while current.right:
                current = current.right
            return current



        
        # if the node is a leaf, simply remove it
        if target.left == None and target.right == None:
            print("removing leaf")
            
            # if the target is the root, set root to None
            if self.root is target:
                print("\t removing root")
                self.root = None
            #otherwise, we need to see what side of the previous node the target is on
            else:
                parent = target.prev
                if parent.right is target:
                    print(f"\tremoving parent.right")
                    parent.right = None
                elif parent.left is  target:
                    print('\tremoving from parent.left')
                    parent.left = None
                else:
                    raise RuntimeError("parent cannot find target, target has no children")
            

            return 
                
        
        # if the node has a single child, the parent adopts it

        # target has only a right child
        elif target.left == None and target.right:
            print('removing node with right child')
            child = target.right
            
            
            # if the target is the root, assign the child to be root
            if self.root is target:
                print('\tremoving root')
                self.root = child
                child.prev = None

            # the target is not the root, and has a parent
            else:
                parent = target.prev
                child.prev = parent
                if parent.left is target:
                    print('\tremoving parent.left')
                    parent.left = child
                elif parent.right is target:
                    print('\tremoving parent.right')
                    parent.right = child
                else:
                    raise RuntimeError("parent cannot find target, target has right child")
            
            return
        
        # target has only the left child
        elif target.right == None and target.left:
            print('removing node with left child')
            child = target.left

            # if the target is the root, assign child to be root
            if self.root is target:
                print('\tremoving root')
                self.root = child
                child.prev = None
            
            # the target is not the root, and thus has a parent
            else:
                parent = target.prev
                child.prev = parent
                if parent.left is target:
                    print('\tremoving parent.left')
                    parent.left = child
                    child.prev = parent
                elif parent.right is target:
                    print('\t removing parent.right')
                    parent.right = child
                    child.prev = parent
                else:
                    raise RuntimeError("parent cannot find target, target has left child")
            
            return
        
        # if the node has two children, swap it with the immediate predecessor, and then remove
        elif target.right and target.left:
            print('removing node with two children')
            predecessor = find_immediate_predecessor(target)
            print(f'Predecessor: {predecessor.value}')

            temp = predecessor.value
            self.remove(predecessor.value)
            target.value = temp
            

            return
        
        else: # something went horribly wrong
            raise RuntimeError("target somehow doesn't have any children, and has children")
        
        raise NotImplementedError("Problem 3 Incomplete")

    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    english_file = open("english.txt")
    # start_time = time.time()
    english_words = [line.strip() for line in english_file]
    # print(f'It took {time.time() - start_time} seconds to read all words into the vector')

    # this is the size of the sample to grab. 
    # Note: stop cannot exceed 16, or random will try to grab more words than are in the vector
    stop = 16
    sample_sizes = 2**np.arange(start = 3, stop = stop, step = 1)
    
    SLList_loading_times = []
    SLList_search_times = []

    bin_tree_loading_times = []
    bin_tree_search_times = []

    AVL_tree_loading_times = []
    AVL_tree_search_times = []

    # get the loading and search times for the three data types
    for size in sample_sizes:
        # print(f'size: {size}')
        random_sample = np.random.choice(english_words, size=size, replace = False)
        # print(random_sample)
        search_sample_size = 5
        search_sample = np.random.choice(random_sample, size = search_sample_size, replace=False)
        # print(f'\tfinding {search_sample} in data')

        SLList = SinglyLinkedList()
        binary_tree = BST()
        AVL_tree = AVL()

        # timing the initialization of the singly linked list
        # print('\t\tinitializing SLList')
        start_time = time.time()
        for s in random_sample:
            SLList.append(s)
        SLList_loading_times.append(time.time() - start_time)

        # timing the search function for the singly linked list
        start_time = time.time()
        for word in search_sample:
            # print(f'\t\t\tfinding {word} in SLList:')
            word_found = SLList.iterative_find(word)
            # print(f'\t\t\t found {word_found.value} in SLList')
        SLList_search_times.append(time.time() - start_time)

        # timing the initialization of the BST
        # print('\t\tinitializing BST')
        start_time = time.time()
        for s in random_sample:
            binary_tree.insert(s)
        bin_tree_loading_times.append(time.time() - start_time)

        # timing the search function for our BST
        start_time = time.time()
        for word in search_sample:
            # print(f'\t\t\tfinding {word} in BST:')
            word_found = binary_tree.find(word)
            # print(f'\t\t\t found {word_found.value} in BST')
        bin_tree_search_times.append(time.time() - start_time)

        # timing the initializer for the AVL tree
        # print('\t\tinitializing AVL')
        start_time = time.time()
        for s in random_sample:
            AVL_tree.insert(s)
        AVL_tree_loading_times.append(time.time() - start_time)

        # timing the search function for the AVL tree
        start_time = time.time()
        for word in search_sample:
            # print(f'\t\t\tfinding {word} in AVL:')
            word_found = AVL_tree.find(word)
            # print(f'\t\t\t found {word_found.value} in AVL tree')
        AVL_tree_search_times.append(time.time() - start_time)
    
    plt.subplot(121)
    plt.title("Loading Times")
    plt.loglog(sample_sizes, SLList_loading_times, 'k', base = 2)
    plt.loglog(sample_sizes, bin_tree_loading_times, 'r', base = 2)
    plt.loglog(sample_sizes, AVL_tree_loading_times, 'b', base = 2)
    plt.legend(["SLList", "Binary Tree", "AVL Tree"])
    
    plt.subplot(122)
    plt.title("Search Times")
    plt.loglog(sample_sizes, SLList_search_times, 'k', base = 2)
    plt.loglog(sample_sizes, bin_tree_search_times, 'r', base = 2)
    plt.loglog(sample_sizes, AVL_tree_search_times, 'b', base = 2)
    plt.legend(["SLList", "Binary Tree", "AVL Tree"])

    plt.savefig(f"Timing_Figs/load_and_search_times_stop={stop}.png")


if __name__ == "__main__":
    # ************** PROBLEM 1 *************

    # myList = SinglyLinkedList()
    # for i in range(10):
    #     myList.append(i)
    # for i in range(11):
    #     print(myList.recursive_find(i).value)

    # ************** PROBLEM 2 *************

    myBST = BST()
    for i in [4, 3, 6, 5, 7, 8, 1]:
        print(f'{myBST}\n\n')
        myBST.insert(i)

    print(f'{myBST}')
    data = int(input("enter the data to remove: "))
    myBST.remove(data)
    print(myBST)


    # ************** PROBLEM 4 *************
    os.chdir("/Users/chase/Desktop/Math321Volume2/byu_vol2/BinaryTrees")
    prob4()
