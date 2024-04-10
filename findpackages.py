#!/usr/bin/env python3
"""Searches for Python module in a given path and prints them to STDOUT."""
import os
import sys
from pkgutil import iter_modules
from typing import Any, Union, Set

from setuptools import find_packages


def error_print(*args: Any, **kwargs: Any) -> None:
    """Prints to STDERR.

    Args:
        *args: A list of arguments
        **kwargs: A dict of key-value arguments
    """
    print(*args, file=sys.stderr, **kwargs)


def find_modules(path: Union[str, os.PathLike]) -> Set[str]:
    """Finds Python modules under a given path and returns them.

    Args:
        path: The path to search for Python modules

    Returns:
        A set of found modules
    """
    modules: Set[str] = set()
    for package in find_packages(
        path, exclude=[
                "*.tests",
                "*.tests.*",
                "tests.*",
                "tests",
                "test",
                "test.*",
                "*.test.*",
                "*.test",
                "*_test",
                "test_*",
                "*_tests",
                "test_*",
            ]
    ):
        pkg_path = "{}/{}".format(path, package.replace(".", "/"))
        for info in iter_modules([pkg_path]):
            if not info.ispkg:
                modules.add(f"{package}.{info.name}")
    return modules


if __name__ == '__main__':
    if sys.version_info < (3, 6, 0):
        error_print("Requires at least Python 3.6.0")
        sys.exit(1)
    if not len(sys.argv) == 2:
        error_print("Usage: findpackages.py <path/to/project/root>")
        error_print("Searches for all Python modules in <path/to/project/root> and")
        error_print("returns them one module name per line")
        sys.exit(1)
    for module in find_modules(sys.argv[1].strip()):
        print(module.strip())
