import subprocess
import logging
import shutil
from pathlib import Path
import pandas as pd

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


end_to_end_out_dir = Path("test_output/end_to_end")


def test_end_to_end():
    """
    Test FlaPy via the flapy.sh interface.
    !! This test execution the flapy container from dockerhub, not any local build !!
    """

    end_to_end_out_dir.mkdir(exist_ok=True)

    out_dir = end_to_end_out_dir / "example_results_tiny"
    out_tests_overview = end_to_end_out_dir / "example_results_tiny_to.csv"

    out_dir.mkdir(exist_ok=True)

    # CLEAN
    shutil.rmtree(out_dir, ignore_errors=True)
    out_tests_overview.unlink(missing_ok=True)

    # RUN TESTS
    logging.info("RUNNING TESTS")
    cmd = (
        "./flapy.sh run "
        f"--out-dir {out_dir} --plus-random-runs "
        "flapy_input_example_tiny.csv 1"
    )
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable="/bin/bash"
    )
    out, err = process.communicate()
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    logging.info(f"OUT: {out}")
    logging.info(f"ERROR: {err}")

    # CHECK OUTPUT
    assert len(list(out_dir.rglob("*"))) == 11

    # PARSE
    logging.info("PARSING")
    cmd = (
        "./flapy.sh parse ResultsDirCollection "
        f"--path {out_dir} "
        "get_tests_overview _df "
        f"to_csv --index=False {out_tests_overview}"
    )
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable="/bin/bash"
    )
    out, err = process.communicate()
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    logging.info(f"OUT: {out}")
    logging.info(f"ERROR: {err}")

    # CHECK PARSED OUTPUT
    df = pd.read_csv(out_tests_overview)
    assert len(df) == 14

