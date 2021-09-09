# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<Name>
<Class>
<Date>
"""
import math

from typing import Collection


class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
        color (str): 
        the backpack's color
        max_size (int): the maximum capacity of the backpack
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        """Set the name and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the backpack's color
            max_size (int): the maximum capacity of the backpack
        """
        self.name = name
        self.contents = []
        self.color = color
        self.max_size = max_size
    
    

    def put(self, item):
        """Add an item to the backpack's list of contents."""
        if(len(self.contents) <= self.max_size):
            self.contents.append(item)
        else:
            print("No Room!")

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)
    
    def dump(self):
        "removes all contents from self.contents and sets it back to an empty list"
        self.contents = []

    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)
    
    def __eq__(self, other):
        return (self.name == other.name) and (self.color == other.color) and (len(self.contents) == len(other.contents))
    
    def __str__(self):
        returnString = "Owner:\t" + self.name
        returnString += "\nColor:\t" + self.color
        returnString += "\nSize:\t" + str(len(self.contents))
        returnString += "\nMax Size:\t" + str(self.max_size)
        returnString += "\nContents:\t" + str(self.contents)
        return returnString


# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    ''' Inherits fromt the Backpack class, but also allows for fuel storage and flying
        
        
        Attributes:
            name (str): the name of the backpack's owner.
            contents (list): the contents of the backpack.
            color (str): the backpack's color
            max_size (int): the maximum capacity of the backpack
            fuel_level (float): the current fuel level
    '''

    def __init__(self, name, color, max_size = 5, fuel = 10.0):
        ''' Initializes a jetpack with name, color, ma
            
            Parameters:
                name (str): name (str): the name of the backpack's owner
                color (str): the backpack's color
                max_size (int): the maximum capacity of the backpack
                fuel (float): the starting fuel level
        '''

        super.__init__(name, color, max_size)
        self.fuel_level = fuel
    
    def fly(self, fuel_burned):
        ''' Burns (decrements) the fuel level by the amount burned. If the amount burned is greater than the fuel level,
            prints Not enough fuel! and does not decrement the fuel
            
            Parameters:
                fuel_burned (float): the amount of fuel to be burned
        '''

        if(fuel_burned > self.fuel_level):
            print("Not enough fuel!")
        else:
            self.fuel_level -= fuel_burned
    
    def dump(self):
        ''' removes all contents from self.contents and sets it back to an empty list
            also dumps the fuel and sets fuel_level to 0.0'''

        self.contents = []
        self.fuel_level = 0.0



# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber:
    ''' Implements a complex number class
        Attributes:
            real (float): the real part of the number
            imag (float): the imaginary part of the number
    '''

    def __init__(self, real, imag):
        ''' Initailizes a new complex number
            Parameters:
                real (float): the real part of the number
                imag (float): the imaginary part of the number
        '''
        
        self.real = real
        self.imag = imag
    
    def conjugate(self):
        return ComplexNumber(self.real, -1* self.imag)

    def __str__(self):
        string = str(self.real)
        if self.imag >= 0:
            string += " +"
        else:
            string += " "
        string += str(self.imag)
        string += "i"
        return string
    
    def __abs__(self):
        return math.sqrt(self.real**2 + self.imag**2)
    
    def __eq__(self, other):
        return (self.real == other.real) and (self.imag == other.imag)
    
    def __add__(self,other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        return ComplexNumber(self.real - other.real, self.imag - other.imag)
    
    def __mul__(self, other):
        return ComplexNumber(self.real * other.real - self.imag * other.imag, self.imag * other.real + self.real * other.imag)
        
    def __truediv__(self, other):
        numerator = self * other.conjugate()
        denominator = other * other.conjugate()
        print(numerator, denominator, sep = "\n")
        return ComplexNumber(numerator.real / denominator.real, numerator.imag / denominator.real)
        

if __name__ == "__main__":
    swiss_army = Backpack("chase", "red", 10)
    woodward = Backpack("Chase", "black", 5)
    print(swiss_army)
    swiss_army.put("red book")
    swiss_army.put("blue book")
    swiss_army.put("Deus Ex Machina")
    print(swiss_army)
    swiss_army.take("Deus Ex Machina")
    print(swiss_army)
    swiss_army.dump()
    print(swiss_army)
    woodward.put("Azathae notebook")
    print(woodward)
    print("woodard == swiss_army: ", woodward == swiss_army)
    
    '''
    complex_num = ComplexNumber(1,0)
    other_c_num = ComplexNumber(0,1)
    print(complex_num, other_c_num, sep = "\n")
    print(complex_num / other_c_num)'''
