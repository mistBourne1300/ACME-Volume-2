# shell2.py
"""Volume 3: Unix Shell 2.
<Name>
<Class>
<Date>
"""

from glob import glob
import os
import subprocess

# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.
    
    Returns:
        List of filenames that contain the matched string
    """
    list_of_files = []
    for filename in glob(f'**/{file_pattern}', recursive = True):
        with open(filename) as f:
            if target_string in f.read():
                list_of_files.append(filename)
    
    return list_of_files

    raise NotImplementedError("Problem 5 Incomplete")


# Problem 4
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    files_list = glob("**/*.*", recursive = True)
    files_list = sorted(files_list, key = os.path.getsize, reverse = True)
    files_list = files_list[:n]
    subprocess.Popen([f'wc -l < {files_list[-1]} >> smallest.txt'], shell = True)
    return files_list


    
# Problem 6    
def prob6(n = 10):
   """this problem counts to or from n three different ways, and
      returns the resulting lists each integer
   
   Parameters:
       n (int): the integer to count to and down from
   Returns:
       integerCounter (list): list of integers from 0 to the number n
       twoCounter (list): list of integers created by counting down from n by two
       threeCounter (list): list of integers created by counting up to n by 3
   """
   #print what the program is doing
   integerCounter = list()
   twoCounter = list()
   threeCounter = list()
   counter = n
   for i in range(n+1):
       integerCounter.append(i)
       if (i % 2 == 0):
           twoCounter.append(counter - i)
       if (i % 3 == 0):
           threeCounter.append(i)
   #return relevant values
   return integerCounter, twoCounter, threeCounter

