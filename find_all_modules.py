#!/usr/bin/env python3
"""Searches for Python module in a given path and prints them to STDOUT."""
import os
import sys
from pkgutil import iter_modules
from typing import Any, Union, Set, List

from setuptools import find_packages


def error_print(*args: Any, **kwargs: Any) -> None:
    """Prints to STDERR.

    Args:
        *args: A list of arguments
        **kwargs: A dict of key-value arguments
    """
    print(*args, file=sys.stderr, **kwargs)


def flatten_nested_list(l: List[Union[Any, List]]) -> List:
    """
    """
    l_flat = []
    for x in l:
        if isinstance(x, list):
            flat_x = flatten_nested_list(x)
            l_flat.extend(flat_x)
        else:
            l_flat.append(x)
    return l_flat


def find_modules(path: Union[str, os.PathLike], prefix=None) -> List[str]:

    if prefix is None:
        prefix = []

    modules = [
        ".".join(prefix + [module.name])
        if not module.ispkg else
        find_modules(f"{path}/{module.name}", prefix=(prefix + [module.name]))
        for module in iter_modules([path])
    ]

    return flatten_nested_list(modules)

    # for package in find_packages(path):
    #     pkg_path = "{}/{}".format(path, package.replace(".", "/"))
    #     for info in iter_modules([pkg_path]):
    #         if not info.ispkg:
    #             modules.add(f"{package}.{info.name}")
    # return modules


if __name__ == '__main__':
    if sys.version_info < (3, 6, 0):
        error_print("Requires at least Python 3.6.0")
        sys.exit(1)
    if not len(sys.argv) == 2:
        error_print("Usage: find_all_packages.py <path/to/project/root>")
        error_print("Searches for all Python modules in <path/to/project/root> and")
        error_print("returns them one module name per line")
        sys.exit(1)
    for module in find_modules(sys.argv[1].strip()):
        print(module)
