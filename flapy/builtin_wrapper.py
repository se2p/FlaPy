"""Wrap all imported builtin functions"""
import sys
import types
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.abc import MetaPathFinder, FileLoader
from inspect import isclass, getmembers
from typing import Optional

# TODO: investigate why wrapping those functions sometimes causes TypeError
#  (e.g. kindle-maker: TypeError: mkdir takes at most 2 positional arguments (3 given)
DO_NOT_WRAP = [
    ("os", "open"),
    ("os", "stat"),
    ("os", "mkdir"),
    ("os", "listdir"),
]


def wrap(function):
    """
    Wraps given function
    It is a decorator, but is not being used like one
    Reference: https://blog.sqreen.com/dynamic-instrumentation-agent-for-python/
    """
    # @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except TypeError as ex:
            # print()
            # print("TYPE-ERROR IN WRAPPER")
            # print(f"args = {args}")
            # print(f"kwargs = {kwargs}")
            # print()
            raise ex

    return wrapper


def wrap_module(module):
    """Wraps all builtin functions in the given module"""

    for name, func in getmembers(
        module, lambda o: isinstance(o, types.BuiltinFunctionType),
    ):
        if (module.__name__, name) not in DO_NOT_WRAP:
            # print(f"\t\twrapping {module.__name__}.{name}")
            setattr(module, name, wrap(func))


def wrap_module_recursively(module):
    """Wraps all builtin functions in the given module"""

    for name, obj in getmembers(module):
        if isinstance(obj, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            print(f"\t\twrapping recursively {module.__name__}.{name}")
            try:
                setattr(module, name, wrap(obj))
            except TypeError as exception:
                print(exception.args)
        if isinstance(obj, (types.ModuleType, type)) and obj != type:
            print(f"--- go inside {name}")
            wrap_module_recursively(obj)
            print(f"--- back from {name}")


class InstrumentationLoader(SourceFileLoader):
    """A loader that instruments the module after execution."""

    def exec_module(self, module):
        """
        Instruments the module after it was executed.
        Installs a tracer into the loaded module.
        """
        # print(f"\texec module {module.__name__}")
        super().exec_module(module)
        wrap_module(module)


class InstrumentationFinder(MetaPathFinder):
    """
    A meta path finder which wraps another pathfinder.
    It receives all import requests and intercepts the ones for the modules that
    should be instrumented.
    """

    def __init__(self, original_pathfinder):
        """
        Wraps the given path finder.
        :param original_pathfinder: the original pathfinder that is wrapped.
        :param module_to_instrument: the name of the module, that should be instrumented.
        """
        self._original_pathfinder = original_pathfinder

    def find_spec(self, fullname, path=None, target=None):
        """
        Try to find a spec for the given module.
        If the original path finder accepts the request, we take the spec and replace the loader.
        """
        # print(f"find spec {fullname}")
        spec: ModuleSpec = self._original_pathfinder.find_spec(fullname, path, target)
        if spec is not None:
            if isinstance(spec.loader, FileLoader):
                spec.loader = InstrumentationLoader(spec.loader.name, spec.loader.path)
                return spec
            # print(
            #     "Loader for module under test is not a FileLoader, cannot instrument.",
            #     file=sys.stderr,
            # )

        return None


class ImportHookContextManager:
    """A simple context manager for using the import hook."""

    def __init__(self, hook: Optional[MetaPathFinder]):
        self.hook = hook

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uninstall()

    def uninstall(self):
        """Remove the installed hook."""
        if self.hook is None:
            return

        try:
            sys.meta_path.remove(self.hook)
        except ValueError:
            pass  # already removed


def install_import_hook() -> ImportHookContextManager:
    """
    Install the InstrumentationFinder in the meta path.
    :return a context manager which can be used to uninstall the hook.
    """
    to_wrap = None
    for finder in sys.meta_path:
        if (
            isclass(finder)
            and finder.__name__ == "PathFinder"  # type: ignore
            and hasattr(finder, "find_spec")
        ):
            to_wrap = finder
            break

    if not to_wrap:
        raise RuntimeError("Cannot find a PathFinder in sys.meta_path")

    hook = InstrumentationFinder(to_wrap)
    sys.meta_path.insert(0, hook)
    return ImportHookContextManager(hook)


def install():
    """Install the import hook and manually wrap certain modules"""
    install_import_hook()

    # Manually wrap modules that were already loaded
    # import builtins
    # wrap_module(builtins)

    import time  # pylint: disable=import-outside-toplevel

    wrap_module(time)

    # import os  # pylint: disable=import-outside-toplevel
    #
    # wrap_module(os)
