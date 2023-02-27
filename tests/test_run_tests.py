import shutil
import os
from flapy import tempfile_seeded
from flapy import tempfile_hardcoded
from flapy import __version__
from flapy.run_tests import FlakyAnalyser
import test_resources
import test_output
from pathlib import Path


def rm_recursively(path: Path):
    """
    Helper function to remove a non-empty directory.
    `shutils.rmtree` does not work in virtualbox-shared-folders
    """
    if path.is_dir():
        for sub_path in path.glob("*"):
            rm_recursively(sub_path)
        path.rmdir()
    else:
        os.remove(path)


def test_tracing():
    generic_test_tracing(FlakyAnalyser)


def test_isolation():
    generic_test_isolation(FlakyAnalyser)


def generic_test_tracing(flaky_analyser: FlakyAnalyser):
    project_name = "test_resources"

    out_dir = Path(os.path.dirname(test_output.__file__)) / "run_tests" / "test_tracing"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "execution.log"

    # Clean out_dir
    for path in out_dir.glob("*"):
        if ".gitignore" not in path.name and "__pycache__" not in path.name:
            rm_recursively(path)

    #tmp_dir = tempfile_seeded.mkdtemp()
    tmp_dir = tempfile_hardcoded.mkdtemp()
    print(f"Using temporary directory {tmp_dir}")

    # fmt: off
    args = [
        "run_tests.py",
        "--logfile", str(log_file.absolute()),
        "--repository", os.path.dirname(test_resources.__file__),
        "--project-name", project_name,
        "--temp", tmp_dir,
        "--number-test-runs", "1",
        "--tests-to-be-run", "test_trace_me.py",
        "--trace",
            "test_trace_me.py::test_hashing "
            "test_trace_me.py::TestFixtures::test_foo "
            "test_trace_me.py::test_random "
            "test_trace_me.py::test_path "
            "test_trace_me.py::test_super_call",
    ]
    # fmt: on
    analyser = flaky_analyser(args)
    analyser.run()

    results_files = list(Path(tmp_dir).glob("*"))
    print(f"Found following results files: {results_files}")
    assert len(results_files) > 0

    # move results to output folder
    for path in results_files:
        # remove '::' from filename
        shutil.move(path, str(path.absolute()).replace("::", "__"))
        shutil.move(str(path.absolute()).replace("::", "__"), out_dir)

    assert (out_dir / f"{project_name}_output0test_trace_me.py.xml").is_file()

    shutil.rmtree(tmp_dir)


def generic_test_isolation(flaky_analyser: FlakyAnalyser):
    project_name = "test_resources"

    out_dir = Path(os.path.dirname(test_output.__file__)) / "run_tests" / "test_isolation"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "execution.log"

    # Clean up out_dir
    for path in out_dir.glob("*"):
        if ".gitignore" not in path.name and "__pycache__" not in path.name:
            rm_recursively(path)

    #tmp_dir = tempfile_seeded.mkdtemp()
    tmp_dir = tempfile_hardcoded.mkdtemp()
    print(f"Using temporary directory {tmp_dir}")

    # fmt: off
    args = [
        "run_tests.py",
        "--logfile", str(log_file.absolute()),
        "--repository", os.path.dirname(test_resources.__file__),
        "--project-name", project_name,
        "--temp", tmp_dir,
        "--number-test-runs", "1",
        "--tests-to-be-run", "test_trace_me.py::test_quick_math"
    ]
    # fmt: on
    analyser = flaky_analyser(args)
    analyser.run()

    results_files = list(Path(tmp_dir).glob("*"))
    print(f"Found following results files: {results_files}")
    assert len(results_files) > 0

    # move results to output folder
    for path in results_files:
        # remove :: from filename
        shutil.copy(path, str(path.absolute()).replace("::", "__"))
        shutil.copy(str(path.absolute()).replace("::", "__"), out_dir)

    # assert junit-xml output exists
    assert (out_dir / f"{project_name}_output0test_trace_me.py__test_quick_math.xml").is_file()

    shutil.rmtree(tmp_dir)
