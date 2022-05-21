#!/usr/bin/env python3
# pylint: skip-file

import pickle
import sys
from typing import List, Tuple
from unittest.mock import MagicMock
from importlib import import_module
from ast import literal_eval


def initialize_mock(mock_to_value: List[Tuple[str, object]]):
    for to_mock, value in mock_to_value:
        *module_to_mock, func_to_mock = to_mock.split(".")
        module = import_module(".".join(module_to_mock))
        setattr(module, func_to_mock, MagicMock(return_value=value))


def main(args: List[str] = None):
    """The main entry location of the program."""
    if args is None:
        args = sys.argv[1:]

    if len(args) == 0:
        print("Usage: python3 with_mock.py FUNCTION:PICKELED_RETURN_VALUE... -- SCRIPT [ARGS...]")
        sys.exit()

    # Load
    mock_args_str: List[str] = args[: args.index("--")]
    _, script, *script_args = args[args.index("--") :]

    # Mock
    mock_args = [
        (to_mock, pickle.loads(literal_eval(mock_value)))
        for to_mock, mock_value in [x.split(":") for x in mock_args_str]
    ]
    initialize_mock(mock_args)

    # Exec script
    sys.argv = [script, *script_args]
    exec(open(script).read())


if __name__ == "__main__":
    main()
