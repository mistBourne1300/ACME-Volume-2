# test_specs.py
"""Python Essentials: Unit Testing.
<Name>
<Class>
<Date>
"""

from typing import Type
from _pytest.monkeypatch import V

from _pytest.outcomes import _with_exception
import specs
import pytest


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    assert specs.smallest_factor(7) == 7
    assert specs.smallest_factor(5) == 5
    assert specs.smallest_factor(25) == 5
    assert specs.smallest_factor(2) == 2


# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    assert specs.month_length("September") == 30
    assert specs.month_length("April") == 30 
    assert specs.month_length("June") == 30
    assert specs.month_length("November") == 30
    assert specs.month_length("January") == 31
    assert specs.month_length("March") == 31
    assert specs.month_length("May") == 31
    assert specs.month_length("July") == 31
    assert specs.month_length("August") == 31
    assert specs.month_length("October") == 31
    assert specs.month_length("December") == 31
    assert specs.month_length("February") == 28
    assert specs.month_length("February", True)
    assert specs.month_length("Febuary") == None


# Problem 3: write a unit test for specs.operate().
def test_specs_operate():
    with pytest.raises(TypeError) as excinfo:
        specs.operate(1,2,0)
    assert excinfo.value.args[0] == "oper must be a string"

    assert specs.operate(1,2,'+') == 3
    assert specs.operate(1,2,'*') == 2
    assert specs.operate(1,2,'-') == -1
    assert specs.operate(1,2,'/') == .5
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.operate(1,0,'/')
    assert excinfo.value.args[0] == "division by zero is undefined"

    with pytest.raises(ValueError) as excinfo:
        specs.operate(1,2,';')
    assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"


# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    frac_5_1 = specs.Fraction(5,1)
    return frac_1_3, frac_1_2, frac_n2_3, frac_5_1

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_5_1 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_1_0 = specs.Fraction(1,0)
    assert excinfo.value.args[0] == "denominator cannot be zero"

    with pytest.raises(TypeError) as excinfo:
        frac_str1 = specs.Fraction("1", 2)
    assert excinfo.value.args[0] == "numerator and denominator must be integers"

    with pytest.raises(TypeError) as excinfo:
        frac_srt2 = specs.Fraction(1,"2")
    assert excinfo.value.args[0] == "numerator and denominator must be integers"

    with pytest.raises(TypeError) as excinfo:
        frac_str1_2 = specs.Fraction("1","2")
    assert excinfo.value.args[0] == "numerator and denominator must be integers"

    assert frac.numer == 5
    assert frac.denom == 7

def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_5_1 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(frac_5_1) == "5"

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_5_1 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_5_1 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert frac_5_1 == 5

def test_fraction_add(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_5_1 = set_up_fractions
    assert frac_1_3 + frac_n2_3 == -1/3
    assert frac_5_1 + frac_1_2 == 11/2
    assert frac_1_3 + frac_1_2 == 5/6

def test_fraction_sub(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_5_1 = set_up_fractions
    assert frac_1_3 - frac_n2_3 == 1
    assert frac_1_2 - frac_1_3 == 1/6
    assert frac_5_1 - frac_1_2 == 9/2


def test_fraction_mul(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_5_1 = set_up_fractions
    assert frac_5_1 * frac_1_2 == 5/2
    assert frac_n2_3 * frac_1_3 == -2/9
    assert frac_5_1 * frac_n2_3 == -10/3


def test_fraction_truediv(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3, frac_5_1 = set_up_fractions
    assert frac_1_2 / frac_n2_3 == -3/4
    assert frac_5_1 / frac_1_2 == 10
    assert frac_1_3 / frac_5_1 == 1/15
    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_0_1 = specs.Fraction(0,1)
        frac_5_1 / frac_0_1
    assert excinfo.value.args[0] == "cannot divide by zero"
    





# Problem 5: Write test cases for Set.
@pytest.fixture
def set_up_hands():
    hand1 = [   "1022", "1122", "0100", "2021",
                "0010", "2201", "2111", "0020",
                "1102", "0210", "2110", "1020"]
    
    hand2 = [   "2100", "2220", '1012', '0112',
                '0100', '2000', '2201', '0201',
                '0022', '1112', '1111', '0210']
    
    hand_wrong_size = [ "1022", "1122", "0100", "2021",
                        "0010", "2201", "2111", "0020",
                        "1102", "0210", "2110"]
    
    hand_not_unique = [ "2100", "2220", '1012', '0112',
                        '0100', '2000', '2201', '0201',
                        '0022', '1012', '1111', '0210']
    
    hand_not_str = [    "2100", "2220", '1012', '0112',
                        '0100', '2000', '2201', '0201',
                        '0022', 1012, '1111', '0210']
    
    hand_card_length_wrong = [  "2100", "2220", '1012', '0112',
                                '0100', '2000', '2201', '0201',
                                '0022', '112', '1111', '0210']

    hand_card_not_base_3 = [    "2100", "2220", '1012', '0112',
                                '0100', '2003', '2201', '0201',
                                '0022', '1112', '1111', '0210'] 

    return hand1, hand2, hand_wrong_size, hand_not_unique, hand_not_str, hand_card_length_wrong, hand_card_not_base_3

@pytest.fixture
def set_up_sets():
    set1 = ["1111", "1111", "1111"]
    set2 = ["1201", "1210", "1222"]
    set3 = ["0120", "1201", "2012"]
    false_set = ["1201", "1212", "0102"]
    set_a_not_string = [120, "1201", "2012"]
    set_b_not_string = ["0120", 1201, "2012"]
    set_c_not_string = ["0120", "1201", 2012]
    set_a_wrong_length = ["10120", "1201", "2012"]
    set_b_wrong_length = ["0120", "12021", "2012"]
    set_c_wrong_length = ["0120", "1201", "2s012"]
    set_not_base_3 = ["0120", "1201", "4012"]
    return set1, set2, set3, false_set, set_a_not_string, set_b_not_string, set_c_not_string, set_a_wrong_length, set_b_wrong_length, set_c_wrong_length, set_not_base_3

def test_count_sets(set_up_hands):
    hand1 , hand2, hand_wrong_size, hand_not_unique, hand_not_str, hand_card_length_wrong, hand_card_not_base_3 = set_up_hands
    assert specs.count_sets(hand1) == 6 
    assert specs.count_sets(hand2) ==0 
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(hand_wrong_size)
    assert excinfo.value.args[0] == "Hand is not exactly 12 Cards"

    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(hand_not_unique)
    assert excinfo.value.args[0] == 'Cards are not unique'

    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(hand_not_str)
    assert excinfo.value.args[0] == 'Card is not str'

    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(hand_card_length_wrong)
    assert excinfo.value.args[0] == 'Card does not have length 4'

    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(hand_card_not_base_3)
    assert excinfo.value.args[0] == "Card is not base 3 int"

def test_is_set(set_up_sets):
    set1, set2, set3, false_set, set_a_not_string, set_b_not_string, set_c_not_string, set_a_wrong_length, set_b_wrong_length, set_c_wrong_length, set_not_base_3  = set_up_sets
    assert specs.is_set(set1[0], set1[1], set1[2]) == True
    assert specs.is_set(set2[0], set2[1], set2[2]) == True
    assert specs.is_set(set3[0], set3[1], set3[2]) == True
    assert specs.is_set(false_set[0], false_set[1], false_set[2]) == False
    
    with pytest.raises(ValueError) as excinfo:
        specs.is_set(set_a_not_string[0], set_a_not_string[1], set_a_not_string[2])
    assert excinfo.value.args[0] == 'Card is not str'

    with pytest.raises(ValueError) as excinfo:
        specs.is_set(set_b_not_string[0], set_b_not_string[1], set_b_not_string[2])
    assert excinfo.value.args[0] == 'Card is not str'

    with pytest.raises(ValueError) as excinfo:
        specs.is_set(set_c_not_string[0], set_c_not_string[1], set_c_not_string[2])
    assert excinfo.value.args[0] == 'Card is not str'

    with pytest.raises(ValueError) as excinfo:
        specs.is_set(set_a_wrong_length[0], set_a_wrong_length[1], set_a_wrong_length[2])
    assert excinfo.value.args[0] == 'Card does not have length 4'

    with pytest.raises(ValueError) as excinfo:
        specs.is_set(set_b_wrong_length[0], set_b_wrong_length[1], set_b_wrong_length[2])
    assert excinfo.value.args[0] == 'Card does not have length 4'

    with pytest.raises(ValueError) as excinfo:
        specs.is_set(set_c_wrong_length[0], set_c_wrong_length[1], set_c_wrong_length[2])
    assert excinfo.value.args[0] == 'Card does not have length 4'

    with pytest.raises(ValueError) as excinfo:
        specs.is_set(set_not_base_3[0], set_not_base_3[1], set_not_base_3[2])
    assert excinfo.value.args[0] == "Card is not base 3 int"


