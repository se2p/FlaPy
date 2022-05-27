""" Provides helper functions """
import sys
from typing import Callable, Type, TypeVar, Any, Union


T = TypeVar("T")
U = TypeVar("U")


def try_default(
    function: Callable[[], T],
    exception: Type[BaseException],
    error_return_val: Union[U, Callable[[Type[BaseException]], U]],
    finally_: Callable[[], Any] = None,
) -> Union[T, U]:
    """ Helper function. Try-except is not allowed in lambdas.

    function: the function that shall be called. It must not have input parameters -> curry them first
    exception: type (class) of exception, that should be caught
    error_return_val: either a function that shall be called

    """
    try:
        return function()
    except exception as e:
        if callable(error_return_val):
            return error_return_value(e)
        elif isinstance(error_return_val, str) and error_return_val == "ERROR_MESSAGE":
            return f"{type(e).__name__}: {e}"
        elif isinstance(error_return_val, str) and error_return_val == "ERROR_MESSAGE_TUPLE":
            return ("error", f"{type(e).__name__}: {e}")
        else:
            return error_return_val
    finally:
        if finally_:
            finally_()


def eprint(*args, **kwargs):
    """Print on stderr"""
    print(*args, file=sys.stderr, **kwargs)
