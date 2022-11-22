import sys
import time
import random
from pathlib import Path
from threading import Thread
import requests
import numpy as np


# Luo 2014 Categories:
# * Async Wait
# * Concurrency
# * Test Order Dependency
# * Resource Leak
# * Network
# * Time
# * IO
# * Randomness
# * Floating Point Operation
# * Unordered Collections


def test_not_flaky():
    assert True


def test_concurrency():
    # data race
    global x
    x = 0
    t1 = Thread(target=lambda: repeat(lambda: setx(x + 1), 50000))
    t2 = Thread(target=lambda: repeat(lambda: setx(x + 1), 50000))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert x == 100000


def test_floating_point_operation():
    assert 0.1 * 3 == 0.3  # Fails constantly


def test_random():
    assert random.randint(0, 1) == 1


def test_time():
    t = time.time()
    assert round(t) % 2 == 0


def test_network_remote_connection_failure():
    response = requests.get("http://heise.de", timeout=0.15)
    assert response.status_code == 200


# def test_network_local_bad_socket_management():
#   # open socket that might be already in use
#     pass


def test_io():
    assert Path("/bin").is_dir()


def test_numpy_random():
    assert np.random.randint(0, 2) == 1

def test_unordered_collections():
    """
    This test is flaky due to the non-deterministic builtin-function hash()
    that controls the ordering within sets
    """
    s = {"hello", "world"}
    assert next(s.__iter__()) == "hello"


# PLATFORM DEPENDENCIES


def test_numeric_operation_32bit():
    # regular python has arbitrary precision, so no overflow, but numpy has
    # Windows seems to use 32bit int while Linux uses 64bit -> platform dependency
    a = np.array([2 ** 31 - 1], dtype=int) + 1
    ref = 2 ** 31
    assert a[0] == ref  # Windows: Assertion Error, Linux: passed


def test_numeric_operation_64bit():
    # regular python has arbitrary precision, so no overflow, but numpy has
    # Windows seems to use 32bit int while Linux uses 64bit -> platform dependency
    a = np.array([2 ** 63 - 1], dtype=int) + 1  # Windows: Overflow Error
    ref = 2 ** 63
    assert a[0] == ref  # Linux: Assertion Error


def test_memory_usage():
    assert sys.getsizeof("hello") == 30  # Windows: passed, Linux: failed (54)

# ORDER DEPENDENCIES

x = 0


def test_victim():
    assert x == 0

def test_polluter():
    global x
    x = 5


def test_self_polluter():
    # assert that file does NOT exists and create it if not
    self_polluter_file = Path("/tmp/.minimal_flaky_example_self_polluter_file")
    file_existed = self_polluter_file.exists()
    self_polluter_file.touch(exist_ok=True)
    assert not file_existed

def test_self_statesetter():
    # assert that file exists and create it if not
    self_state_setter_file = Path("/tmp/.minimal_flaky_example_self_state_setter_file")
    file_existed = self_state_setter_file.exists()
    self_state_setter_file.touch(exist_ok=True)
    assert file_existed
