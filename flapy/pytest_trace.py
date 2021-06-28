#!/usr/bin/env python3
# This project is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This project is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this project.  If not, see <https://www.gnu.org/licenses>.
"""Runs pytest with instrumentation to trace a given function"""
# pylint: disable=wrong-import-order,wrong-import-position
# flake8: noqa: E402

from flapy import (  # type: ignore
    builtin_wrapper,
)

builtin_wrapper.install()

# import numpy as np  # if numpy is imported after builtins in wrapped, this causes errors
import platform
import inspect
import ctypes
import logging
import sys
from multiprocessing import Process, Manager
from trace import _fullmodname  # type: ignore
from typing import Dict, List, Tuple, Optional, IO, Any, Callable, TypeVar, Sequence
from types import FrameType, FunctionType
import signal
from contextlib import contextmanager
import deepdiff  # type: ignore
import re
import pytest  # type: ignore
import json
import hashlib
from pathlib import Path
import os

# TODO: Fix "type: ignore" -> why do I get "Module ... has no attribute ..."?
# from my_garlicsim_py3.garlicsim.general_misc import pickle_tools
from flapy.pickle_tools import dumps_skip, loads_skip  # type: ignore  # pylint: disable=ungrouped-imports
from flapy.copy_fallback import deepcopy  # type: ignore  # pylint: disable=ungrouped-imports
from deepdiff import DeepHash  # type: ignore  # pylint: disable=ungrouped-imports
from flapy.results_parser import FuncDescriptor  # type: ignore  # pylint: disable=ungrouped-imports


T = TypeVar("T")  # pylint: disable=invalid-name

logging.getLogger("deepdiff").setLevel(logging.ERROR)

# Initialize multiprocessing.Manager
MANAGER = Manager()


@contextmanager
def timeout(time):
    """
    Usage: `with timeout(5): foo` in order to execute foo with the given timeout
    """
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    """
    Helper function, because you can't raise an error inside a lambda
    """
    raise TimeoutError("The execution exceeded the timeout I set")


def exec_with_timeout(function: Callable[[], T], time_out: int) -> Optional[T]:
    """
    ALTERNATIVE TO timeout
    Execute function, break after timeout.
    If timeout was exceeded, return None, otherwise return the return-value of the function.
    :param function:
    :param time_out:
    :return:
    """
    return_val = MANAGER.Value(ctypes.c_char_p, None)

    def save_return_value():
        # TODO also propagate exceptions, not only return-value.
        # So far all exceptions lead to retun-value = None and therefore a Timout-Error
        return_val.value = function()

    process = Process(target=save_return_value)
    process.start()
    process.join(time_out)
    process.terminate()
    if return_val.value is None:
        raise TimeoutError(f"Timeout of {time_out}s exceeded")
    return return_val.value


def deephash_fallback(obj: Any) -> str:
    """
    Performs DeepHash with fallback.
    Usually DeepHash(obj)[obj] gives you the hash of obj.
    This does not work in certain cases, e.g. if obj is True or False.
    In this case I take the last hash in the hash-dict which should be the hash of obj.
    """
    hashes: Dict[Any, str] = DeepHash(obj)
    try:
        result = hashes[obj]
    except KeyError:
        result = f"{list(hashes.values())[-1]} (took last item because of KeyError)"
    if isinstance(result, deepdiff.helper.Unprocessed):
        result = "unprocessed"
    return result


def frame_to_classname(frame: FrameType) -> str:
    """
    Retrieving the name of class the stack-frame is in (if it is within a class).
    :param frame:
    :return:
    """
    if "self" in frame.f_locals.keys():
        return frame.f_locals["self"].__class__.__name__
    if "cls" in frame.f_locals.keys():
        try:
            return frame.f_locals["cls"].__name__
        except AttributeError:
            return ""
    return ""


def func_to_descriptor(function: FunctionType) -> Tuple[str, str, str]:
    """
    Retrieves the full name of a given function given.
    Inspired by cpython/Objects/methodobject.c::meth_repr
    :param function: pointer to the function
    :return: Tuple of (module_name, class_name, function_name)
    """
    if hasattr(function, "__self__"):
        module_name = function.__self__.__class__.__module__  # type: ignore
    else:
        module_name = function.__module__
    *class_, func = function.__qualname__.split(".")
    return module_name, ".".join(class_), func


def frame_to_func(frame):
    """Retrieve the pointer to the function you are in given the frame"""
    func_name = frame.f_code.co_name
    if "self" in frame.f_locals.keys():
        return getattr(frame.f_locals["self"].__class__, func_name)
    return getattr(inspect.getmodule(frame), func_name)


def to_hash(
    obj: object,
    hash_timeout: int,
    mod_name: str,
    class_name: Optional[str],
    func_name: str,
) -> str:
    """
    Return a hash value of a given object
    while filter from a black list and applying a timeout
    """
    hash_blacklist = [
        ("git.cmd", "AutoInterrupt", "__getattr__"),
        ("tempfile", None, "_get_candidate_names"),
        ("email.feedparser", "BufferedSubFile", "__iter__"),
        (
            "scipy.optimize._differentialevolution",
            "DifferentialEvolutionSolver",
            "_scale_parameters",
        ),
    ]

    if (mod_name, class_name, func_name) in hash_blacklist:
        return "Not hashed because of blacklist"

    try:
        obj_copy = deepcopy(obj)
    except Exception as ex:  # pylint: disable=broad-except
        return f"Error_while_copying {type(ex)} {ex}"

    try:
        # call_args_hash = ""
        # call_args_hash = call_args

        # copy.deepcopy(call_args)
        # call_args_hash = my_deep_hash(call_args)

        # call_args_hash = exec_with_timeout(
        #     lambda: my_deep_hash(
        #         call_args  # pylint: disable=cell-var-from-loop
        #     ),
        #     hash_timeout,
        # )

        # Windows does not support the alarm signal
        if platform.system() == "Windows":
            return deephash_fallback(obj_copy)
        with timeout(hash_timeout):
            return deephash_fallback(obj_copy)

    except Exception as ex:  # pylint: disable=broad-except
        return f"Error_while_hashing {type(ex)} {ex}"


MEMORY_ADDRESS_PATTERN = re.compile("at 0x[0-9a-f]+")


def to_string(obj: object) -> str:
    """
    Get a string representation of an object
    that contains no memory address and throws no exception
    """
    try:
        return MEMORY_ADDRESS_PATTERN.sub("at MEMORY ADDRESS", repr(obj)).replace(
            "\n", " "
        )
    except Exception as ex:  # pylint: disable=broad-except
        return f"Error_while_to_string {type(ex)} {ex}"


def to_pickle_hash(obj: object) -> Tuple[str, Optional[bytes]]:
    """Pickle and hash given object"""
    try:
        args_pic = dumps_skip(obj)
    except Exception as ex:  # pylint: disable=broad-except
        return f" Error_while_pickling {type(ex)} {ex}", None
    else:
        args_pic_hash = hashlib.sha256(args_pic).hexdigest()
        return args_pic_hash, args_pic


def frame_to_func_descriptor(frame: FrameType) -> Tuple[str, str, str, List[str]]:
    """Retrieve module-name, class-name, function-name and tags including all fallbacks"""
    func_name = frame.f_code.co_name
    filename = frame.f_code.co_filename
    tags = []
    if func_name == "wrapper" and "function" in frame.f_locals:
        mod_name, class_name, unwrapped_func_name = func_to_descriptor(
            frame.f_locals["function"]
        )
        tags.append("#wrapper")
    else:
        unwrapped_func_name = func_name
        try:
            func = frame_to_func(frame)
            mod_name, class_name, _ = func_to_descriptor(func)
        except Exception:  # pylint: disable=broad-except
            tags.append("#naming-fallback-funcPtr")
            mod = inspect.getmodule(frame)
            if mod is not None:
                mod_name = mod.__name__
            else:
                tags.append("#naming-fallback-inspect")
                mod_name = _fullmodname(filename)
            class_name = frame_to_classname(frame)
    return mod_name, class_name, unwrapped_func_name, tags


def message_prefix(event: str, indent: int) -> str:
    """Generate the arrow-like beginning of a trace-entry given event and indent"""
    if event in ["call", "c_call"]:
        return "-" * indent + ">"
    if event in ["return", "c_return"]:
        return "<" + "-" * indent
    return f'WRONG OPTION "{event}"'


def message_args_separator(event: str) -> str:
    """Get arrow-separator ('<=' or '=>') given the event"""
    if event in ["call", "c_call"]:
        return "<="
    if event in ["return", "c_return"]:
        return "=>"
    return f'WRONG OPTION "{event}"'


# pylint: disable=too-many-statements,too-many-nested-blocks
def run_with_trace(
    run: Callable,
    tracedfuncs_to_output: List[Tuple[Tuple[str, ...], IO[str], bool, bool]],
    hash_timeout: int = 5,
    args_cutoff: int = 1000,
    trace_all=False,
) -> Dict[str, str]:
    """
    Execute a command while tracing certain functions and writing the output directly to files.
    :param run: command to be called (without parameters -> curry first)
    :param tracedfuncs_to_output: list of function-names that shall be traced mapped to the
        output-stream the respective trace shall be written to plus two bool value representing if
        the output shall be hashed and string-ed
        EXAMPLE: [('test_ebook_save', <file>, True, False)]
    :param args_cutoff: if the argument is greater than this value, it will be cut off to avoid
        huge output files
    :return: hash_to_pickle
    """
    inside_function: Dict[Tuple[str, ...], bool] = {
        f: False for f, _, _, _ in tracedfuncs_to_output
    }

    if trace_all:
        tracedfuncs_to_output.append(
            (("", ""), open("trace_all.txt", "w"), False, False)
        )
        inside_function.update({("", ""): True})

    indent: Dict[IO[str], int] = {out: 0 for _, out, _, _ in tracedfuncs_to_output}

    # TODO: if two objects have the same key (same hash), the value gets overwritten
    #  (less likely for hashable objects, more likely for Error_while_hashing)
    #  -> check if key already exists, if value is different, append suffix to key
    hash_to_pickle: Dict[str, str] = {}

    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    def tracefunc(frame: FrameType, event: str, arg):
        # TODO: add possibility to trace only by function-name
        #   regardless of file_name and class_name (e.g. for fixtures)
        file_name = os.path.basename(frame.f_code.co_filename)
        func_name = frame.f_code.co_name
        class_name = frame_to_classname(frame)

        if class_name == "":
            func_desc: Tuple[str, ...] = (file_name, func_name)
        else:
            func_desc = (file_name, class_name, func_name)

        if event == "call" and func_desc in inside_function.keys():
            inside_function[func_desc] = True
        if event in ["call", "return", "c_call", "c_return"]:
            for traced_func, output, hash_it, str_it in tracedfuncs_to_output:
                if inside_function[traced_func]:
                    if event in ["call", "c_call"]:
                        indent[output] += 2

                    if event in ["call", "return"]:
                        (
                            mod_name,
                            class_name,
                            unwrapped_func_name,
                            tags,
                        ) = frame_to_func_descriptor(frame)

                        output.write(
                            message_prefix(event, indent[output])
                            + f" {(mod_name, class_name, unwrapped_func_name)} {' '.join(tags)}"
                        )
                        output.flush()
                        # TODO: f_locals is more than just the call argument (all local variables)
                        #  is that a problem?
                        #  possible fix: use frame.f_code.co_varnames
                        # pylint: disable=deprecated-method,cell-var-from-loop
                        if event == "call":
                            arguments = frame.f_locals
                        if event == "return":
                            arguments = arg

                        if hash_it:
                            try:
                                args_pic = dumps_skip(arguments)
                            except Exception as ex:  # pylint: disable=broad-except
                                output.write(" " + message_args_separator(event))
                                output.write(f" Error_while_pickling {type(ex)} {ex}")
                            else:
                                # Load pickeled args again, because dumps_skip will skip over
                                # certain unwanted non-determinisms in fixtures
                                try:
                                    reloaded_args = loads_skip(args_pic)
                                except Exception as ex:  # pylint: disable=broad-except
                                    reloaded_args = arguments
                                    output.write(" #reloading-args-failed")
                                    output.flush()
                                args_pic_hash = to_hash(
                                    reloaded_args,
                                    hash_timeout,
                                    mod_name,
                                    class_name,
                                    unwrapped_func_name,
                                )
                                if (
                                    len(args_pic) > args_cutoff
                                    and (mod_name, class_name, unwrapped_func_name)
                                    not in DONT_CUTOFF
                                ):
                                    args_pic = "Cutoff"
                                hash_to_pickle[args_pic_hash] = str(args_pic)
                                output.write(" " + message_args_separator(event))
                                output.write(f" {args_pic_hash}")
                        if str_it:
                            output.write(f" {to_string(arguments)}")

                    elif event in ["c_call", "c_return"]:
                        output.write(
                            message_prefix(event, indent[output])
                            + f" {func_to_descriptor(arg)} #builtin"
                        )
                        output.flush()

                    output.write("\n")
                    output.flush()

                    if event in ["return", "c_return"]:
                        indent[output] -= 2
        if event == "return" and func_desc in inside_function.keys():
            inside_function[func_desc] = False
        return tracefunc

    sys.setprofile(tracefunc)
    run()
    sys.setprofile(None)
    return hash_to_pickle


PICKLE_FOLDER_NAME = "trace_args_pickels"
HASH_TO_PICKLE_FILENAME = "HashToPickle.json"
DONT_CUTOFF: List[str] = []

PYUNIT_FIXTURES_ON_MODULE: List[str] = [
    "setUpModule",
    "tearDownModule",
]
PYUNIT_FIXTURES_ON_CLASS: List[str] = [
    "setUp",
    "tearDown",
    "setUpClass",
    "tearDownClass",
    "asyncSetUp",
    "asyncTearDown",
]


def main(args: List[str] = None) -> None:  # pylint: disable=too-many-locals
    """
    The main entry location of the program.

    Why not use a proper CLI like argparse or fire?
    -> I want this script to act like pytest, so I need direct access to flags such as '-v'.
        Such flags are being caught by CLI frameworks.
    """

    if not args:
        args = sys.argv[1:]

    if len(args) < 2:
        print("USAGE: pytest_trace.py  FUNCS_TO_TRACE  OUT_FILE  [PYTEST_ARGS]")
        print()
        print(
            "FUNCS_TO_TRACE should be one or multiple functions separated by spaces,\n"
            "so use parentheses to still make them count as one argument\n"
            'Example: "func1 func2"\n'
        )
        print('to trace the entire program, set FUNCS_TO_TRACE to "main"')
        sys.exit()

    print()
    print("Entering pytest_trace")
    print()

    (quoted_funcs_to_be_traced, output_file_name, *pytest_args) = args
    funcs_to_be_traced: Sequence[FuncDescriptor] = [
        tuple(entry.split("::")) for entry in quoted_funcs_to_be_traced.split(" ")
    ]  # Example: [("file', "class", "func"), ("file", "func")]

    # Discard path, only keep filename
    funcs_to_be_traced = [
        (file.split("/")[-1], *func) for file, *func in funcs_to_be_traced
    ]

    # Add pyunit fixtures
    class_fixtures_to_trace: Sequence[FuncDescriptor] = [
        (func[0], func[1], fixture)
        for func in funcs_to_be_traced
        for fixture in PYUNIT_FIXTURES_ON_CLASS
        if len(func) == 3
    ]
    module_fixtures_to_trace: Sequence[FuncDescriptor] = list(
        {
            (func[0], fixture)
            for func in funcs_to_be_traced
            for fixture in PYUNIT_FIXTURES_ON_MODULE
            if len(func) == 3
        }
    )
    funcs_to_be_traced += class_fixtures_to_trace
    funcs_to_be_traced += module_fixtures_to_trace

    hash_to_pickle_filepath = Path(output_file_name).parent / Path(
        HASH_TO_PICKLE_FILENAME
    )
    if not hash_to_pickle_filepath.exists():
        existing_hash_to_pickle = {}
    else:
        with open(hash_to_pickle_filepath) as dict_file:
            existing_hash_to_pickle = json.load(dict_file)

    print(f"Funcs to trace: {funcs_to_be_traced}")
    print(f"Pytest args: {pytest_args}")

    tracefuncs_to_output: List[Tuple[Tuple[str, ...], IO[str], bool, bool]] = []
    for func in funcs_to_be_traced:
        out_file = open(f"{output_file_name}_{func}.txt", "w")
        tracefuncs_to_output.append((func, out_file, False, False))
        # tracefuncs_to_output.append(
        #     (func, open(f"{output_file_name}_{func}__strReturn.txt", "w"), True, True)
        # )

    hash_to_pickle = run_with_trace(
        lambda: pytest.main(pytest_args),
        tracefuncs_to_output,
        # hashing_timeout,
    )
    existing_hash_to_pickle.update(hash_to_pickle)
    print("CLOSING OUTPUT FILES")
    for _, out, _, _ in tracefuncs_to_output:
        out.close()
    print("WRITING JSON DICT")
    with open(hash_to_pickle_filepath, "w") as dict_file:
        json.dump(existing_hash_to_pickle, dict_file, indent=4)


if __name__ == "__main__":
    main()
