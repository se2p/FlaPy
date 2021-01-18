import shutil
import os
from flapy import tempfile_seeded
from flapy import __version__
from flapy.analysis import FlakyAnalyser
import test_resources
import test_output
from pathlib import Path


def test_version():
    assert __version__ == "0.1.0"


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
    out_dir = Path(os.path.dirname(test_output.__file__)) / "analysis" / "test_tracing"

    # Clean up out_dir
    log_file = out_dir / "execution.log"
    out_file = out_dir / "output.txt"
    for path in out_dir.glob("*"):
        if ".gitignore" not in path.name and "__pycache__" not in path.name:
            rm_recursively(path)

    tmp_dir = tempfile_seeded.mkdtemp()
    print(f"Using temporary directory {tmp_dir}")

    # fmt: off
    args = [
        "analysis.py",
        "--logfile", str(log_file.absolute()),
        "--repository", os.path.dirname(test_resources.__file__),
        "--temp", tmp_dir,
        "--number-test-runs", "1",
        "--deterministic",
        "--output", str(out_file.absolute()),
        "--trace",
        "test_trace_me.py::test_hashing "
        "test_trace_me.py::TestFixtures::test_foo "
        "test_trace_me.py::test_random "
        "test_trace_me.py::test_path "
        "test_trace_me.py::test_super_call",
        "--tests-to-be-run", "test_trace_me.py"
    ]
    # fmt: on
    analyser = FlakyAnalyser(args)
    analyser.run()

    results_files = list(Path(tmp_dir).glob("*"))
    print(f"Found following results files: {results_files}")
    assert len(results_files) > 0

    # Move results to output folder
    for path in results_files:
        # Remove :: from filename
        shutil.move(path, str(path.absolute()).replace("::", "__"))
        shutil.move(str(path.absolute()).replace("::", "__"), out_dir)

    assert os.path.isfile(out_file)

    shutil.rmtree(tmp_dir)


def test_isolation():
    out_dir = (
        Path(os.path.dirname(test_output.__file__)) / "analysis" / "test_isolation"
    )

    # Clean up out_dir
    log_file = out_dir / "execution.log"
    out_file = out_dir / "output.txt"
    for path in out_dir.glob("*"):
        if ".gitignore" not in path.name and "__pycache__" not in path.name:
            rm_recursively(path)

    tmp_dir = tempfile_seeded.mkdtemp()
    print(f"Using temporary directory {tmp_dir}")

    # fmt: off
    args = [
        "analysis.py",
        "--logfile", str(log_file.absolute()),
        "--repository", os.path.dirname(test_resources.__file__),
        "--temp", tmp_dir,
        "--number-test-runs", "1",
        "--random-order-bucket", "global",
        "--output", str(out_file.absolute()),
        "--tests-to-be-run", "test_trace_me.py::test_quick_math"
    ]
    # fmt: on
    analyser = FlakyAnalyser(args)
    analyser.run()

    results_files = list(Path(tmp_dir).glob("*"))
    print(f"Found following results files: {results_files}")
    assert len(results_files) > 0

    # Move results to output folder
    for path in results_files:
        # Remove :: from filename
        shutil.move(path, str(path.absolute()).replace("::", "__"))
        shutil.move(str(path.absolute()).replace("::", "__"), out_dir)

    assert os.path.isfile(out_file)

    shutil.rmtree(tmp_dir)
