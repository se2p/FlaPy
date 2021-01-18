# this import need to be first, so the builtin-wrapper gests installed early
from flapy import pytest_trace

import json
import os
import re
import pytest
from typing import Tuple
from pathlib import Path
from itertools import zip_longest

import test_output
import test_output_reference
import test_resources
import test_resources.test_trace_me
import test_resources.test_trace_me_slowly

out_dir = Path(test_output.__file__).parent / "pytest_trace"
ref_dir = Path(test_output_reference.__file__).parent / "pytest_trace"
hash_to_pickle_filepath = out_dir / "HashToPickle.json"

funcs_to_trace = [
    ("test_trace_me.py", "test_quick_math"),
    ("test_trace_me.py", "test_hashing"),
    ("test_trace_me.py", "TestFixtures", "test_foo"),
    ("test_trace_me.py", "test_path"),
    ("test_trace_me.py", "test_super_call"),
]


@pytest.fixture(scope="session", autouse=True)
def hash_to_pickle_json():
    hash_to_pickle_json = {}
    yield hash_to_pickle_json
    with open(hash_to_pickle_filepath, "w") as dict_file:
        json.dump(hash_to_pickle_json, dict_file, indent=4)


@pytest.mark.parametrize("func_to_trace", funcs_to_trace, ids=str)
def test_trace(func_to_trace: Tuple[str, ...], hash_to_pickle_json):
    out_file = out_dir / f"trace_{func_to_trace}.txt"
    ref_file = ref_dir / f"trace_{func_to_trace}.txt"
    with open(out_file, "w") as output_stream:
        hash_to_pickle = pytest_trace.run_with_trace(
            lambda: pytest.main([test_resources.test_trace_me.__file__, "-vs"]),
            [(func_to_trace, output_stream, True, False)],
        )
        hash_to_pickle_json.update(hash_to_pickle)
    with open(out_file) as out, open(ref_file) as ref:
        for o, r in zip_longest(out, ref):
            assert o == r


def test_builtin(hash_to_pickle_json):
    func_to_trace = ("test_trace_me.py", "test_random")
    out_file = os.path.join(out_dir, f"trace_{func_to_trace}.txt")
    with open(out_file, "w") as output_stream:
        hash_to_pickle = pytest_trace.run_with_trace(
            lambda: pytest.main([test_resources.test_trace_me.__file__, "-s"]),
            [(func_to_trace, output_stream, True, False)],
        )
        hash_to_pickle_json.update(hash_to_pickle)
    reference = r"""--> \('test_resources\.test_trace_me', '', 'test_random'\)  <= \S+
----> \('numpy\.random\.mtrand', 'RandomState', 'uniform'\) #wrapper <= \S+
------> \('numpy\.random\.mtrand', 'RandomState', 'uniform'\) #builtin
--------> \('builtins', '', 'empty'\) #wrapper <= \S+
----------> \('builtins', '', 'empty'\) #builtin
<---------- \('builtins', '', 'empty'\) #builtin
<-------- \('builtins', '', 'empty'\) #wrapper #reloading-args-failed => \S+
<------ \('numpy\.random\.mtrand', 'RandomState', 'uniform'\) #builtin
<---- \('numpy\.random\.mtrand', 'RandomState', 'uniform'\) #wrapper #reloading-args-failed => \S+
<-- \('test_resources\.test_trace_me', '', 'test_random'\)  => \S+
"""
    with open(out_file) as output_stream:
        trace = output_stream.read()
        for trace_line, reference_line in zip(trace.split("\n"), reference.split("\n")):
            assert re.match (reference_line,trace_line)


# def test_entire_program():
#     output_stream = open("{out_dir}/trace_EntireProgram.txt", "w")
#     pytest_trace.run_pytest_with_trace(
#         {"main": output_stream}, [test_resources.test_trace_me.__file__], 10
#     )
#     output_stream.close()


# NO LONGER NEEDED: NOW I USE PICKLE INSTEAD OF HASHING
# def test_timeout():
#     out_file = os.path.join(out_dir, "test_timeout.txt")
#     ref_file = os.path.join(ref_dir, "test_timeout.txt")
#     func_to_trace = "test_expensive_hash"
#     with open(out_file, "w") as output_stream:
#         pytest_trace.run_with_trace(
#             lambda: pytest.main([test_resources.test_trace_me_slowly.__file__, "-s"]),
#             [(func_to_trace, output_stream, True, True)],
#             1,
#         )
#     assert filecmp.cmp(out_file, ref_file)
