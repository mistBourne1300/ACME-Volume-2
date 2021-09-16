# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Name>
<Class>
<Date>
"""

from random import choice
import numpy as np
from numpy.core.fromnumeric import transpose


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:
    
    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """
    
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    step_1_int = int(step_1)
    if (step_1_int < 100) or (step_1_int > 999):
        raise ValueError("step one should be a 3 digit number")
    
    if(abs( int(step_1[0]) - int(step_1[2]) ) < 2):
        raise ValueError(f"The first and last digits of step_1 must differ by at least 2: {step_1[0]} and {step_1[2]}")

    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    step_2_int = int(step_2)
    
    if(step_2 != step_1[::-1]):
        raise ValueError(f"step_2 must be the reverse of step_1: {step_1} -> {step_1[::-1]}")
    
    positive_difference = abs(step_1_int - step_2_int)
    step_3 = input("Enter the positive difference of these numbers: ")

    if(int(step_3) != positive_difference):
        raise ValueError(f"step_3 must be the positive difference of step_1 and step_2: {positive_difference}")


    step_4 = input("Enter the reverse of the previous result: ")
    if(step_4 != step_3[::-1]):
        raise ValueError(f"step_4 must be the reverse of step_3: {step_3} -> {step_3[::-1]}")

    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the 
    program is running, the function should catch the exception and 
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """
    
    walk = 0
    directions = [1, -1]
    for i in range(int(max_iters)):
        try:
            walk += choice(directions)
        except KeyboardInterrupt as KI:
            print(f'Process interupted at iteration {i}; walk: {walk}')
    print("Process Completed")
    return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
    """Class for reading in file
        
    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file
        
    """




class ContentFilter(object):   
    # Problem 3

    def openFile(self):
        try:
            self.file = open(self.filename)
        except:
            self.filename = input("Please enter a valid file name: ")
            self.openFile()


    def __init__(self, filename):
        """Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
        self.filename = filename
        self.openFile()
        self.contents = self.file.readlines()
        self.file.close()

            
    
 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """Raise a ValueError if the mode is invalid."""
        if not (mode == 'r' or mode == 'w' or mode == 'x' or mode == 'a'):
            raise ValueError(f"Mode is invalid: {mode}")
        



    def uniform(self, outfile, mode='w', case='upper'):
        """Write the data to the outfile in uniform case."""
        self.check_mode(mode)
        if not (case == "upper" or case == "lower"):
            raise ValueError(f"Case not a valid case: {case}")
        
        self.file = open(outfile, mode)
        if(case == "upper"):
            for line in self.contents:
                self.file.write(line.upper())
        elif(case == "lower"):
            for line in self.contents:
                self.file.write(line.lower())
        self.file.close()


    def reverse(self, outfile, mode='w', unit='word'):
        """Write the data to the outfile in reverse order."""
        self.check_mode(mode)
        if not (unit == "word" or unit == 'line'):
            raise ValueError(f"Unit not a valid unit: {unit}")
        
        self.file = open(outfile, mode)
        if unit == "word":
            for line in self.contents:
                word_list = line.split()
                for word in word_list[::-1]:
                    self.file.write(word + "\n")
        elif(unit == "line"):
            for line in self.contents[::-1]:
                self.file.write(line)
        
        self.file.close()



    def transpose(self, outfile, mode='w'):
        """Write the transposed version of the data to the outfile."""
        self.check_mode(mode)
        transmogrify = []
        for line in self.contents:
            line_list = line.split()
            transmogrify.append(line_list)
        transmogrify = np.array(transmogrify)
        transmogrify = np.transpose(transmogrify)
        print(transmogrify)
        self.file = open(outfile, mode)
        for line in transmogrify:
            for word in line:
                self.file.write(word + " ")
            self.file.write("\n")
        
        self.file.close()


    def __str__(self):
        """String representation: info about the contents of the file."""


if __name__ == "__main__":
    #arithmagic()
    #print(random_walk(100000))
    '''cf = ContentFilter("hello_world.txt")
    cf.uniform("hello_world2.txt")
    cf.reverse("hello_world3.txt", 'w', "word")
    cf = ContentFilter("cf_example2.txt")
    cf.transpose("cf_example2_output.txt", 'w')'''
    pass