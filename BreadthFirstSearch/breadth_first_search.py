# breadth_first_search.py
"""Volume 2: Breadth-First Search.
<Name>
<Class>
<Date>
"""
from collections import deque
import random
import networkx as nx
import os
import numpy as np
from matplotlib import pyplot as plt

# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        if n not in self.d:
            self.d[n] = set({})

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        if u in self.d:
            self.d[u].add(v)
        else:
            self.d[u] = set({v})
        
        if v in self.d:
            self.d[v].add(u)
        else:
            self.d[v] = set({u})

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        # if the node is not in the graph, raise an error
        if n not in self.d:
            raise KeyError(f'n: {n} not in graph')
        # for each node in the graph, get rid of connections to n, if there are any
        for i in self.d:
            self.d[i].discard(n)
        self.d.pop(n) # remove node n

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        # raise an error if u or v is not in the graph
        if v not in self.d or u not in self.d:
            raise KeyError(f"u: {u} or v: {v} not in graph")
        
        # these will automatically raise the right error if the ndoe are not connected
        self.d[u].remove(v)
        self.d[v].remove(u)

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        # we can't find the traversal from a node that doesn't exist: raise error
        if source not in self.d:
            raise KeyError(f'source {source} not in graph')
        M = set({source})
        V = list()
        Q = deque([source])
        # while we stilll have nodes to search 
        while Q:
            # add the current node to the list of visited nodes
            curr_node = Q.popleft()
            V.append(curr_node)
            # add each neighbor to the list of nodes to visit
            for neigh in self.d[curr_node]:
                if neigh not in M:
                    Q.append(neigh)
                    M.add(neigh)
        return V


    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        # we can't find the shortest path between nodes that don't exist: raise error
        if source not in self.d or target not in self.d:
            raise KeyError(f'source: {source} or target: {target} not in graph')
        M = set({source})
        S = [[source]]

        # while we still have nodes to check
        while S:
            # add the current node to the set of nodes already visited
            path = S.pop(0)
            M.add(path[-1])
            # if we've reached the target, return
            if path[-1] == target:
                return path
            # add each neighbor to the queue
            for node in self.d[path[-1]] - M:
                S.append(path+[node])
        


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        file = open(filename)
        filelines = file.readlines()
        self.movies = set()
        self.actors = set()
        self.graph = nx.Graph()
        for line in filelines:
            stripped = line.strip() # need the /n at the end stripped
            splitline = stripped.split(sep = '/') # per the specs, we are guaranteed that / will only occur where we want it
            self.movies.add(splitline[0]) # add the movie (splitline[0]) to the set of movies
            for actor in splitline[1:]: # add each actor to the set of actors and connect them to the movie
                self.actors.add(actor)
                self.graph.add_edge(splitline[0], actor)
        file.close()

    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        # return the shortest path using the nx module
        return nx.shortest_path(self.graph, source, target), nx.shortest_path_length(self.graph, source, target)/2
        raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        path_lengths = []
        for actor in self.actors:
            # compute the length between the two actors
            if actor != target:
                path_lengths.append(nx.shortest_path_length(self.graph, actor, target)/2)
        
        # create a histogram for the path lengths
        plt.hist(path_lengths)
        plt.title(f"Path Lengths to {target}")
        plt.xlabel(f'Path length to {target}')
        plt.ylabel('Frequency')
        plt.show()
        return np.mean(path_lengths) # return the mean of the lengths

        raise NotImplementedError("Problem 6 Incomplete")


if __name__ == "__main__":
    g = Graph()
    for i in range(10):
        g.add_node(i)
    g.add_edge(9,8)
    g.add_node(8)
    g.add_edge(2,3)
    g.add_edge(9,10)
    for i in range(10,20):
        g.add_edge(i+1,i)
    print(f'complete:\n{g}\n\n')
    g.remove_node(10)
    print(f'removed node 10:\n{g}\n\n')
    g.remove_edge(8,9)
    print(f'removed (8,9):\n{g}\n\n')
    g = Graph()
    num_nodes = 100
    source = 10
    target = 50
    connectivity = 3
    for i in range(num_nodes):
        g.add_node(i)
        for j in range(random.choice(range(connectivity))):
            g.add_edge(i,random.choice(range(num_nodes)))
    
    print(f'random {num_nodes}-node graph:\n{g}\n\n')
    print(f'traverse from {source}: {g.traverse(source)}\n\n')

    print(f'bfs from {source} to {target}: {g.shortest_path(source, target)}\n\n')
    
    os.chdir("/Users/chase/Desktop/Math321Volume2/byu_vol2/BreadthFirstSearch")

    kevin = MovieGraph(filename='movie_data_small.txt')
    print(f'num movies: {len(kevin.movies)}')
    print(f'num actors {len(kevin.actors)}\n')
    print(kevin.path_to_actor("Liam Neeson", "Kevin Bacon"))
    print(f'average path length to kevin bacon: {kevin.average_number("Kevin Bacon")}')
    
    
    pass