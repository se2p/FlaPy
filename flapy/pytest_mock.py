# pylint: skip-file
# type: ignore
# flake8: noqa

import sys
import time
from typing import List, Tuple
from importlib import import_module
import pytest


class FixtureRegPlugin:
    def __init__(self, mock_to_value: List[Tuple[str, object]]):
        self.mock_to_value = mock_to_value

    @pytest.fixture(autouse=True)
    def mock_me(self, monkeypatch):
        for to_mock, value in self.mock_to_value:
            *module_to_mock, func_to_mock = to_mock.split(".")
            module = import_module(".".join(module_to_mock))

            def f(*args, **kwargs):
                return value

            monkeypatch.setattr(module, func_to_mock, f)


def main(args: List[str] = None) -> None:
    """The main entry location of the program."""

    if not args:
        args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: pytest_mock.py FUNCTION:PICKELED_RETURN_VALUE...")
        sys.exit()

    pytest.main(plugins=[FixtureRegPlugin()])


if __name__ == "__main__":
    main()
