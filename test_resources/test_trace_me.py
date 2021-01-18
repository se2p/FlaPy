import os
from typing import Callable, Iterable, List
import unittest
import numpy as np


def test_quick_math():
    i = 2 + 2
    assert i == 4


def my_map(func: Callable, iterable: Iterable) -> List:
    output = []
    for item in iterable:
        output.append(func(item))
    return output


def test_hashing():
    assert my_map(lambda x: x + 1, [1, 2]) == [2, 3]


class TestFixtures(unittest.TestCase):
    def setUp(self):
        self.foo = 42

    def test_foo(self):
        pass


def test_random():
    np.random.uniform(low=0, high=1, size=3)


def test_path():
    os.getcwd()


class SuperClass:
    def overwritten_method(self):
        pass

    def some_method(self):
        pass


class SubClass(SuperClass):
    def overwritten_method(self):
        super().overwritten_method()


def test_super_call():
    c = SubClass()
    c.some_method()
    c.overwritten_method()
