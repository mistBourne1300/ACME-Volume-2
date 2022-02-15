# regular_expressions.py
"""Volume 3: Regular Expressions.
<Name>
<Class>
<Date>
"""

import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile("python")

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[\_\]\{\&\}\$")

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r'^(Book|Mattress|Grocery) (store|supplier)$')

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # python identifier r'[a-zA-Z_]\w*'
    # r"^[a-zA-z_][a-zA-z0-9_]* *(= *[[0-9]*\.?[0-9]*|'[^']?'|[a-zA-z_][a-zA-z0-9_]*])?$"
    return re.compile(r"^[a-zA-Z_]\w* *(= *(\d*\.?\d*|'[^']*'|[a-zA-Z_]\w*))?$")

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    cryptic = re.compile(r"((if|elif|else|for|while|try|except|finally|with|def|class)[^\n]*)\n")
    return cryptic.sub(r"\1:\n", code)

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """
    # compile objects to get each string type
    names = re.compile(r"^[a-zA-Z]+ [A-Z]\.? ?[a-zA-Z]+")
    phones = re.compile(r"1?-?\(?\d{3}\)?-?\d{3}-\d{4}")
    emails = re.compile(r"\b[^ ]*@[^ ]*\.[a-zA-Z]{3}\b")
    birthdays = re.compile(r"\b\d{,2}/\d{,2}/\d{2,4}")

    # object to get and sub in the area codes
    area_code = re.compile(r"^1?-?\(?(\d{3})-?\)?")

    # get each month, day, and year
    bday1 = re.compile(r"(\d+)")
    file = open(filename)
    contacts = dict()
    for line in file.readlines():
        name = names.findall(line)[0]

        if phones.search(line):
            phone = phones.findall(line)[0]
            phone = area_code.sub(r"(\1)", phone)
        else:
            phone = None
        
        if emails.search(line):
            email = emails.findall(line)[0]
        else:
            email = None

        # i know this is cursed. I have tried. You'll have to deal with it, as I have dealt with it.
        if birthdays.search(line):
            birthday = birthdays.findall(line)[0]
            # print(birthday)
            count = 0
            m,d,y = '','',''
            for match in bday1.finditer(birthday):
                string = match.group(1)
                count += 1
                if count == 1:
                    m = string
                elif count == 2:
                    d = string
                else:
                    y = string
                # print(match.group(1))
            # print(bday1.search(birthday).group(1))
            if len(m) <= 1:
                m = '0' + m
            if len(d) <= 1:
                d = '0' + d
            if len(y) <= 2:
                y = '20' + y
            bday = f'{m}/{d}/{y}'
        else:
            bday = None
        print(f'\n{name}\n\tphone: {phone}\n\temail: {email}\n\tbirthday: {bday}')
        cont = {'birthday':bday, 'email':email, 'phone':phone}
        contacts[name] = cont

    file.close()
    return contacts


if __name__ == "__main__":
    # print(bool(prob1().search("this is python for ya")))
    # print(bool(prob2().search("^{@}(?)[%]{.}(*)[_]{&}$")))

    # print("\nprob4 matches:")
    # lies = prob4()
    # for string in ["Mouse", 'compile', '_123456789', '__x__', 'while', 'max=4.2', "string= ''", "num_guesses", "num_guesses =   4"]:
    #     print(f'{string}: {bool(lies.search(string))}')
    
    # print("\n prob4 non matches:")
    # for string in ['3rats', 'err*r', 'sq(x)', 'sleep()', ' x', '300', 'is_4(value==4)', "pattern = r'^one|two fish$'"]:
    #     print(f'{string}: {bool(lies.search(string))}')
    
    block = """k, i, p = 999, 1, 0
while k > i
    i *= 2
    p += 1
    if k != 999
        print("k should not have changed")
    else
        pass
print(p)"""

    large_block = """if __name__ == "__main__"
    # create lotsa errors
    while not done
        for i in range(1000)
            print("yay more errors")
            if i == 1000 % 10
                done = False
            elif i < 999
                done = True
            else
                print("AAAAAHHHHH")
        
        try
            raise ValueError("nope")
        except
            print("yup")
        
        try
            raise NopeError("yup")
        except NopeError
            print("nope")
        finally
            print(done)
    def nope()
        return "nope"
    class guru
        def __init__()
            self.nope = "nope"
    with name as Error
        "I dunno, just testing"
"""
    # print(prob5(large_block))
    print(prob6())