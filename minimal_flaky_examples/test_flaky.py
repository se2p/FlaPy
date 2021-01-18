# Marinov Categories:

# Async Wait (call to server - wait fixed time - assert results)
# Concurrency
# Test Order Dependency
# Resource Leak
# Network
# Time
# IO
# Randomness
# Floating Point Operation
# Unordered Collections

import sys
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import requests
import time
import pathlib
import asyncio
from threading import Thread


# Helper functions


def fib(i):
    if i == 0:
        return 0
    if i == 1:
        return 1
    return fib(i - 1) + fib(i - 2)


async def fib_async(i):
    if i == 0:
        return 0
    if i == 1:
        return 1
    return (await fib_async(i - 1)) + (await fib_async(i - 2))


def setx(v):
    global x
    x = v


def repeat(f, n):
    for _ in range(n):
        f()


def await_fib(n):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(fib_async(n))


# ---------------------

# def test_time_fib():
#     for i in range(10):
#         start = time.time()
#         fib(30)
#         end = time.time()
#         print(end - start)


# CONTROL GROUP


def test_await_NOT_FLAKY():
    # "fib" is part of the trace
    x = await_fib(5)
    assert x == 5


def test_async_wait_NOT_FLAKY():
    # "fib" is NOT part of the trace
    global x
    x = 0
    t = Thread(target=lambda: setx(fib(30)), args=())
    t.start()
    t.join()
    assert x == 832040


def test_async_wait_NOT_FLAKY_2():
    thread_pool = ThreadPoolExecutor()
    future = thread_pool.submit(fib, 30)
    assert future.result() == 832040
    # print(f"Result: {future.result()}")


# ---------------------


def test_async_wait():
    # "fib" is NOT part of the trace
    global x
    x = 0
    Thread(target=lambda: setx(fib(30)), args=()).start()
    # time.sleep(0.447)
    time.sleep(0.855)
    assert x == 832040


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


def test_network_remote_connection_failure():
    response = requests.get("http://heise.de", timeout=0.15)
    assert response.status_code == 200


# def test_network_local_bad_socket_management():
#   # open socket that might be already in use
#     pass


def test_time():
    t = time.time()
    print(t)
    assert round(t) % 2 == 0


def test_io():
    assert pathlib.Path("/bin").is_dir()


def test_randomness():
    # x = random.uniform(-1, 1)
    x = np.random.uniform(-1, 1)
    assert x > 0


def test_floating_point_operation():
    assert 0.1 * 3 == 0.3  # Fails constantly


# Flaky due to non-deterministic builtin-function hash()
def test_unordered_collections():
    s = {"Hello darkness", "my old friend", "I come to talk", "to you again"}
    assert next(s.__iter__()) == "Hello darkness"


# ---------------------


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
