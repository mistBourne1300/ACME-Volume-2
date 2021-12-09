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
    # raise error if the first number step_1 is not a 3-digit number.
    if (step_1_int < 100) or (step_1_int > 999):
        raise ValueError("step one should be a 3 digit number")
    
    # raise error if the first number's first and last digits differ by less than $2$.
    if(abs( int(step_1[0]) - int(step_1[2]) ) < 2):
        raise ValueError(f"The first and last digits of step_1 must differ by at least 2: {step_1[0]} and {step_1[2]}")

    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    step_2_int = int(step_2)
    
    # raise error if the second number step_2 is not the reverse of the first number.
    if(step_2 != step_1[::-1]):
        raise ValueError(f"step_2 must be the reverse of step_1: {step_1} -> {step_1[::-1]}")
    
    positive_difference = abs(step_1_int - step_2_int)
    step_3 = input("Enter the positive difference of these numbers: ")

    # raise error if the third number step_3 is not the positive difference of the first two numbers.
    if(int(step_3) != positive_difference):
        raise ValueError(f"step_3 must be the positive difference of step_1 and step_2: {positive_difference}")


    step_4 = input("Enter the reverse of the previous result: ")

    # raise error if the fourth number step_4 is not the reverse of the third number.
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
            walk += choice(directions) # increment / decrement walk
        except KeyboardInterrupt as KI:
            # user terminated process, print interation and return walk
            print(f'Process interupted at iteration {i}; walk: {walk}')
            return walk
    print("Process Completed")
    return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter(object): 
    """Class for reading in file
        
    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file
        
    """  
    
    # Problem 3
    def openFile(self):
        # try opening the file. If it fails, keep prompting for a filename until it works. 
        # Will break if the user inputs more than 1000 wron filenames (will reach python recursion depth)
        try:
            self.file = open(self.filename)
        except:
            self.filename = input("Please enter a valid file name: ")
            self.openFile()

    def calc_stats(self):

        self.num_char = sum([len(line) for line in self.contents]) # calculate the number of characters
        self.num_lines = len(self.contents) # calculate number of lines
        if(self.contents[-1][-1] == "\n"):
            self.num_lines+=1               # if the last line has a trailing newline, need to add one
        
        self.num_alpha = 0
        self.num_numeric = 0
        self.num_whitespace = 0
        for line in self.contents:
            for char in line:
                if char.isalpha(): self.num_alpha += 1 # calculate the number of alphabetic characters
                elif char.isnumeric(): self.num_numeric += 1 # calculate the number of numeric characters
                elif char.isspace(): self.num_whitespace+=1 # calculate the number of whitespace characters
            
        
        
    

    def __init__(self, filename):
        """Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
        self.filename = filename
        self.openFile()
        self.contents = self.file.readlines()
        self.calc_stats()
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
            # write all data in upper format
            for line in self.contents:
                self.file.write(line.upper())
        elif(case == "lower"):
            # write all data in lower format
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
                for word in word_list[::-1]: # will get the reversed list, which we then write to the file
                    self.file.write(word + " ")
                self.file.write("\n")
        elif(unit == "line"):
            for line in self.contents[::-1]: # will get the reversed list, which we then write to the file
                self.file.write(line)
        
        self.file.close()



    def transpose(self, outfile, mode='w'):
        """Write the transposed version of the data to the outfile."""
        self.check_mode(mode)
        # create a numpy array to transpose
        transmogrify = []
        for line in self.contents:
            line_list = line.split()
            transmogrify.append(line_list)
        transmogrify = np.array(transmogrify)
        transmogrify = np.transpose(transmogrify) # get the transposed version of the array
        # write the transposed array to the outfile
        self.file = open(outfile, mode)
        for line in transmogrify:
            for word in line:
                self.file.write(word + " ")
            self.file.write("\n")
        
        self.file.close()


    def __str__(self):
        """String representation: info about the contents of the file."""
        string = f"Source file:\t\t{self.filename}\n"
        string += f'Total characters:\t{self.num_char}\n'
        string += f'Alphabetic characters:\t{self.num_alpha}\n'
        string += f'Numerical characters:\t{self.num_numeric}\n'
        string += f'Whitespace characters:\t{self.num_whitespace}\n'
        string += f'Number of lines:\t{self.num_lines}'
        return string




if __name__ == "__main__":
    # arithmagic()
    # print(random_walk(10000))
    cf = ContentFilter("hello_world.txt")
    cf.uniform("hello_world2.txt")
    cf.reverse("hello_world3.txt", 'w', "word")
    print(cf)
    cf = ContentFilter("cf_example2.txt")
    cf.transpose("cf_example2_output.txt", 'w')
    print(cf.contents)
    print(cf)
    pass