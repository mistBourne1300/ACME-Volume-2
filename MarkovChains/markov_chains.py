# markov_chains.py
"""Volume 2: Markov Chains.
<Name>
<Class>
<Date>
"""

import numpy as np
import os
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        m,n = A.shape
        if m != n: raise ValueError("A is not square matrix")
        if not np.allclose(np.sum(A, axis = 0), np.ones(A.shape[0])): raise ValueError("Columns of A do not sum to 1")
        self.labels = states
        self.array = A
        self.label_dict = dict([])
        for i in range(len(states)):
            self.label_dict[states[i]] = i

    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        col = self.array[:, self.label_dict[state]] # grabs the column corresponding to the state label passed in
        transition = np.random.multinomial(1, col)
        return self.labels[np.argmax(transition)]
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        states = [start]
        for i in range(N-1):
            states.append(self.transition(states[-1]))
        return states
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        if stop not in self.label_dict:
            raise KeyError(f"Stop: {stop} is not in the chain")
        path = [start]
        while path[-1] != stop:
            path.append(self.transition(path[-1]))
        return path
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        x0 = np.random.random((1,self.array.shape[0])).T
        x0 = x0 / np.sum(x0)
        for i in range(1,maxiter+1):
            x_prev = x0
            x0 = np.linalg.matrix_power(self.array, i) @ x0
            if np.linalg.norm(x0 - x_prev, ord = 1) < tol: return x0
        raise ValueError("Markov chain did not converge")

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        file = open(filename)
        lines = file.readlines()
        stripped_lines = [line.strip() for line in lines]
        self.labels = []
        self.labels.append('$tart')
        # get set of unique words in file
        for strip in stripped_lines:
            words = strip.split()
            for word in words:
                if word not in self.labels:
                    self.labels.append(word)
        
        self.labels.append('$top') # labels now contains the unique words in the text file
        self.array = np.zeros((len(self.labels), len(self.labels))) # create zeros matrix
        self.label_dict = dict([])
        for i in range(len(self.labels)):
            self.label_dict[self.labels[i]] = i

        #create the transition matrix
        for strip in stripped_lines:
            words = strip.split()
            words.insert(0, "$tart")
            words.append("$top")
            for i in range(len(words)-1):
                self.array[self.labels.index(words[i+1]), self.labels.index(words[i])] += 1
        self.array[len(self.labels)-1, len(self.labels)-1] = 1
        self.array /= self.array.sum(axis=0)

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        sentence = self.path('$tart', '$top')
        print(sentence)
        sentence.pop(0)
        sentence.pop()
        print(sentence)
        string = ""
        for word in sentence:
            string += f'{word} '
        return string[:-1]
        raise NotImplementedError("Problem 6 Incomplete")


if __name__ == "__main__":
    simple_weather = np.array([[.7, .6], [.3, .4]])
    markov = MarkovChain(simple_weather, ['hot', 'cold'])
    print(markov.label_dict)
    print(markov.labels)
    print(markov.array)
    print(markov.walk('hot', 10))
    print(markov.steady_state())

    print("\n\n*****************************\n\n")
    weather4states = np.array([[0.5, 0.3, 0.1, 0], [0.3, 0.3, 0.3, 0.3], [0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.2]])
    markov = MarkovChain(weather4states, ['hot', 'mild', 'cold', 'freezing'])
    print(markov.walk("hot", 10))
    print(markov.path('hot', 'freezing'))
    print(markov.steady_state())


    os.chdir("/Users/chase/Desktop/Math321Volume2/byu_vol2/MarkovChains")
    print("\n\n*****************************\n\n")
    babbler = SentenceGenerator("yoda.txt")
    print(babbler.babble())
