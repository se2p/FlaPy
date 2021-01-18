from unittest import TestCase


def setUpModule():
    print("setUpModule")


class MyCase(TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    def setUp(self):
        print("setUp")

    def test_foo(self):
        print("test_foo")

    def test_bar(self):
        print("test_bar")
