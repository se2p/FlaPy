""" Provides helper functions """
import sys
from typing import Callable, Type, TypeVar, Any, Union


T = TypeVar("T")
U = TypeVar("U")


def try_default(
    function: Callable[[], T],
    exception: Type[BaseException],
    error_return_val: U,
    finally_: Callable[[], Any] = None,
) -> Union[T, U]:
    """
    Helper function. Try-except is not allowed in lambdas.
    """
    try:
        return function()
    except exception:
        return error_return_val
    finally:
        if finally_:
            finally_()


def eprint(*args, **kwargs):
    """Print on stderr"""
    print(*args, file=sys.stderr, **kwargs)
