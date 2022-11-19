import sys
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import requests
import time
import pathlib
import asyncio
from threading import Thread


# HELPER FUNCTIONS


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


def await_fib(n):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(fib_async(n))


# CONTROL GROUP (not flaky)


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


# FLAKY


def test_async_wait():
    # "fib" is NOT part of the trace
    global x
    x = 0
    Thread(target=lambda: setx(fib(30)), args=()).start()
    # time.sleep(0.447)
    time.sleep(0.855)
    assert x == 832040





