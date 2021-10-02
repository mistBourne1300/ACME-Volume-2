# linked_lists.py
"""Volume 2: Linked Lists.
<Name>
<Class>
<Date>
"""
import os


# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        if not (type(data) == int or type(data) == float or type(data == str)):
            raise TypeError(f'data must be int, float or str, not {type(data)}')
        
        self.value = data


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
        self.size += 1

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        curr_node = self.head
        while(curr_node is not self.tail):
            if(curr_node.value == data):
                return curr_node
            curr_node = curr_node.next
        raise ValueError("data not found in list")

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        if(i<0 or i>=self.size):
            raise IndexError("attempt to iterate beyond size of node")
        curr_node = self.head
        for j in range(i):
            curr_node = curr_node.next
        return curr_node

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        return self.size

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        list_str = "["
        if(self.size == 0):
            return list_str + ']'
        curr_node = self.head
        list_str += repr(self.head.value)
        while curr_node.next is not None:
            curr_node = curr_node.next
            list_str += ", "
            list_str += repr(curr_node.value)
        list_str += ']'
        return list_str

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        if(self.size == 0):
            raise ValueError("cannot iterate over an empty list")
        curr_node = self.head
        if(curr_node.value == data):
            if(self.head.next is None):
                self.head = None
                self.tail = None
                self.size -= 1
                return
            self.head = self.head.next
            self.head.prev = None
            self.size -= 1
            return
        
        if self.tail.value == data:
            self.tail = self.tail.prev
            self.tail.next = None
            self.size -= 1
            return 
        
        remove_node = self.find(data)
        remove_node.next.prev = remove_node.prev
        remove_node.prev.next = remove_node.next
        self.size -= 1

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        if index == self.size:
            self.append(data)
            return
        
        insert_node = LinkedListNode(data)
        curr_node = self.get(index)
        
        # found insertion index node. need to insert before it

        insert_node.prev = curr_node.prev
        insert_node.next = curr_node
        curr_node.prev = insert_node
        self.size += 1


# Problem 6: Deque class.
class Deque(LinkedList):
    """ 
        deque class for python. only allows insertion and removal from the ends
        
        Attributes:
            head (LinkedListNode) the head of the deque
            tail (LinkedListNode) the tail of the deque
    """

    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        LinkedList.__init__(self)
    
    def remove(*args, **kwargs):
        """arbitrary remove is not allowed in deque"""
        raise NotImplementedError("use pop() or popleft() for removal")
    
    def insert(*args, **kwargs):
        """arbitrary insert is not allowed in deque"""
        raise NotImplementedError("use append() or appendleft() for insertion")
    
    def pop(self):
        """
            pops the last node off the list and returns its data

            Returns:
                the data the popped node contained
        """
        if(self.size == 0):
            raise ValueError("cannot pop from an empty list")
        
        if(self.head is self.tail):
            # removing from a 1-item list
            last_node = self.head
            self.head = None
            self.tail = None
            self.size -= 1
            return last_node.value
        
        pop_node = self.tail
        self.tail = pop_node.prev
        #print(self.tail.value)
        self.tail.next = None
        self.size -= 1
        return pop_node.value
        
    
    def popleft(self):
        """
            pops the first node off the list and returns its data

            Returns:
                the data the popped node contained
        """
        if(self.size == 0):
            raise ValueError("cannot pop from an empty list")
        
        pop_node = self.head

        LinkedList.remove(self, self.head.value)
        return pop_node.value
        
    
    def appendleft(self, data):
        """
            appends a node containing data to the start of the list
        """
        new_node = LinkedListNode(data)
        self.head.prev = new_node
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    





# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """

    file_in = open(infile, 'r')
    file_out = open(outfile, 'w')
    english_words = file_in.readlines()
    for word in english_words[::-11]:
        file_out.write(word)


if __name__ == "__main__":
    mylist = LinkedList()
    print(mylist)
    print("testing 1-length list removal")
    mylist.append(100)
    print(mylist)
    mylist.remove(100)
    print(mylist)
    print("\n\n")
    for i in range(100):
        mylist.append(i)
    mylist.append("'ello world")
    
    print(mylist)
    print(mylist.get(55).value)
    print(mylist.find(35).value)

    input("press enter to continue")
    os.system("clear")
    
    print("deque class:\n\n")
    dq = Deque()
    print("testing deque 1-item removal")
    dq.append(100)
    print("dq: ", dq)
    print(dq.pop())
    print("after pop(): ", dq)

    dq.append(42)
    print("before: ", dq)
    print(dq.popleft())
    print("after popleft() size = : ", dq.size, dq)

    
    input("press enter to continue")
    os.system("clear")

    
    
    for i in range(10):
        dq.append(i)
    print(dq)
    input("press enter to continue")
    os.system("clear")

    print("popping: ", dq.pop())
    print(dq)
    input("press enter to continue")
    os.system("clear")

    print("poppingleft: ", dq.popleft())
    print(dq)
    input("press enter to continue")
    os.system("clear")

    print("appending 45")
    dq.append(45)
    print(dq)
    input("press enter to continue")
    os.system("clear")

    print("appendingleft: 56")
    dq.appendleft(56)
    print(dq)

    while dq.size > 0:
        print("popping left: ", dq.popleft())
        print(dq)
        print("size: ", dq.size)
        input("press enter to continue")
    
    prob7("english.txt", "reverse_english.txt")


