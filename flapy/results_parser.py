#!/usr/bin/env python3
import re
import os
import sys
import tarfile
import tempfile
import sqlite3
import warnings

import string
import keyword

from tqdm import tqdm

# import traceback
import logging
import multiprocessing
import fire  # type: ignore

import xml
import yaml
import operator as op
import numpy as np
import pandas as pd  # type: ignore
import xml.etree.ElementTree as ET

from typing import (
    Callable,
    TypeVar,
    Type,
    Tuple,
    List,
    Union,
    IO,
    Optional,
    Dict,
    Set,
    Any,
    Iterable,
)
from functools import partial
from itertools import groupby, starmap
from ast import literal_eval
from pathlib import Path
from abc import ABC, abstractmethod
from functools import lru_cache, reduce
import junitparser
from coverage.numbits import register_sqlite_functions

from flapy.utils import try_default
from flapy.sfl_scoring import tarantula, ochiai, dStar, barinel, op2

# Initialize pandas progress bar methods
tqdm.pandas()

FuncDescriptor = Tuple[str, ...]

default_flaky_keywords = [
    #
    # ASYNC WAIT
    "sleep",
    #
    # CONCURRENCY
    "thread",
    "threading",
    #
    # RESOURCE LEAK
    #
    # IO
    "('builtins', '', 'stat')",
    "('pathlib', 'Path', 'is_dir')",
    #
    # NETWORK
    "requests",
    #
    # TIME
    "time",
    #
    # RANDOMNESS
    "random",
    #
    # FLOATING POINT
    #
    # UNORDERED COLLECTION
    "__hash__",
    "('builtins', 'set', '__contains__')",
]

logging.getLogger().setLevel(logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)
FORMAT = "[%(asctime)s][%(levelname)7s][%(filename)20s:%(lineno)4s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

proj_cols = ["Project_Name", "Project_URL", "Project_Hash"]
test_cols = ["Test_filename", "Test_classname", "Test_name", "Test_parametrization"]
test_cols_without_parametrization = ["Test_filename", "Test_classname", "Test_name"]
test_cols_without_filename = ["Test_classname", "Test_name", "Test_parametrization"]


def read_junit_testcase(test_case: junitparser.TestCase) -> Dict[str, Union[str, int]]:
    """Transform a Junit test case (xml element under the hood) to a dictionary"""
    return {
        "file": test_case._elem.get("file"),
        "class": test_case.classname,
        "name": test_case.name,
        "verdict": try_default(
            lambda: Verdict.from_junitparser(test_case.result),
            junitparser.JUnitXmlError,
            Verdict.PARSE_ERROR,
        ),
        "message": (
            test_case.result.message
            if test_case.result and test_case.result.message
            else "NO MESSAGE"
        ),
        "errors_in_stacktrace": (
            re.findall(
                r"(.*(?:error|exception).*)", test_case.result._elem.text, flags=re.IGNORECASE
            )
            if (
                test_case.result is not None
                and test_case.result._elem is not None
                and test_case.result._elem.text is not None
            )
            else None
        ),
        "errors_in_system_err": (
            re.findall(r"(.*(?:error|exception).*)", test_case.system_err, flags=re.IGNORECASE)
            if test_case.system_err is not None
            else None
        ),
        "type": (test_case.result.type if test_case.result is not None else ""),
    }


def is_empty(openvia: Callable[[str], IO], path: str):
    try:
        with openvia(path) as f:
            next(f)
    except StopIteration:
        return True
    return False


def junitxml_classname_to_modname_and_actual_classname(classname: str) -> Tuple[List[str], str]:
    """The JUnit-XML attribute 'class' contains both the name of the module and the name of the class -> split them by assuming class names start with capital letters.

    EXAMPLE: "tests.test_camera.TestCamera" -> (['tests', 'test_camera'], 'TestCamera')
    """
    if classname == "":
        return [], ""
    split = classname.split(".")
    try:
        # Case there exists a test-class (assume class names are upper case)
        if split[-1][0].isupper():
            *mod, class_ = split
        # Case there is no test-class, just a test-file
        else:
            mod = split
            class_ = ""
        return mod, class_
    except IndexError:
        logging.warning(
            f"junitxml_classname_to_actual_classname: IndexError with classname={classname}"
        )
        return [], ""


def eval_string_to_set(obj):
    if type(obj) == set:
        return obj
    if obj in ["nan", "", "set()"] or pd.isna(
        obj
    ):  # csv (unlike pandas) represents np.NaN as "nan"
        return set()
    return literal_eval(obj)


class PassedFailed:

    columns = [
        "Iteration",
        "Iteration_status",
        #
        "Project_Name",
        "Project_URL",
        "Project_Hash",
        "Test_filename",
        "Test_classname",
        "Test_name",
        "Test_parametrization",
        #
        "Passed_sameOrder",
        "Failed_sameOrder",
        "Error_sameOrder",
        "Skipped_sameOrder",
        "Verdict_sameOrder",
        "Verdicts_sameOrder",
        "numRuns_sameOrder",
        #
        "Passed_randomOrder",
        "Failed_randomOrder",
        "Error_randomOrder",
        "Skipped_randomOrder",
        "Verdict_randomOrder",
        "Verdicts_randomOrder",
        "numRuns_randomOrder",
    ]

    def __init__(self, df: pd.DataFrame):
        self._df = df

        # Project_Hash is allowed to be empty, for example for local copies instead of remote repos
        self._df["Project_Hash"] = self._df["Project_Hash"].fillna("")

        # Some junit-xml files actually had name="" in them
        #   -> replace by NaN so they get ignored in the groupby
        self._df["Test_name"] = self._df["Test_name"].replace("", np.NaN)

        # Rows with NaN are ignored by pd.groupby -> fillna
        self._df["Test_filename"] = self._df["Test_filename"].fillna("")
        self._df["Test_classname"] = self._df["Test_classname"].fillna("")
        self._df["Test_parametrization"] = self._df["Test_parametrization"].fillna("")

    @classmethod
    def load(cls, path: str):
        """Load PassedFailed from CSV file

        :file_name: TODO
        :returns: TODO

        """
        _df = pd.read_csv(
            path,
            # These converters are disabled, because they cause a lot of memory usage.
            # Instead use `eval_string_to_set` on a filtered version.
            # converters={
            #     'Passed_sameOrder': eval_string_to_set,
            #     'Failed_sameOrder': eval_string_to_set,
            #     'Error_sameOrder': eval_string_to_set,
            #     'Skipped_sameOrder': eval_string_to_set,
            #     'Verdicts_sameOrder': eval_string_to_set,
            #     'Passed_randomOrder': eval_string_to_set,
            #     'Failed_randomOrder': eval_string_to_set,
            #     'Error_randomOrder': eval_string_to_set,
            #     'Skipped_randomOrder': eval_string_to_set,
            #     'Verdicts_randomOrder': eval_string_to_set,
            # }
        )
        return cls(_df)

    def add_rerun_column(self) -> pd.DataFrame:
        self._df["ids_sameOrder"] = [
            p.union(f).union(e).union(s)
            for p, f, e, s in zip(
                self._df["Passed_sameOrder"],
                self._df["Failed_sameOrder"],
                self._df["Error_sameOrder"],
                self._df["Skipped_sameOrder"],
            )
        ]
        return self

    def to_tests_overview(self) -> pd.DataFrame:
        logging.info("")
        self._df["Verdict_sameOrder"] = self._df["Verdicts_sameOrder"].apply(
            lambda s: Verdict.decide_overall_verdict(eval_string_to_set(s))
        )
        self._df["Verdict_randomOrder"] = self._df["Verdicts_randomOrder"].apply(
            lambda s: Verdict.decide_overall_verdict(eval_string_to_set(s))
        )

        self._df["Flaky_sameOrder_withinIteration"] = self._df["Verdict_sameOrder"] == Verdict.FLAKY
        self._df["Flaky_randomOrder_withinIteration"] = (
            self._df["Verdict_randomOrder"] == Verdict.FLAKY
        )
        test_overview = self._df.groupby(
            [
                "Project_Name",
                "Project_URL",
                "Project_Hash",
                "Test_filename",
                "Test_classname",
                "Test_name",
                "Test_parametrization",
            ],
            as_index=False,
        ).agg(
            {
                "Verdicts_sameOrder": lambda l: reduce(
                    set.union, map(lambda s: eval_string_to_set(s), l)
                ),
                "Passed_sameOrder": lambda l: reduce(
                    op.add, map(lambda s: len(eval_string_to_set(s)), l)
                ),
                "Failed_sameOrder": lambda l: reduce(
                    op.add, map(lambda s: len(eval_string_to_set(s)), l)
                ),
                "Error_sameOrder": lambda l: reduce(
                    op.add, map(lambda s: len(eval_string_to_set(s)), l)
                ),
                "Skipped_sameOrder": lambda l: reduce(
                    op.add, map(lambda s: len(eval_string_to_set(s)), l)
                ),
                "numRuns_sameOrder": sum,
                #
                "Verdicts_randomOrder": lambda l: reduce(
                    set.union, map(lambda s: eval_string_to_set(s), l)
                ),
                "Passed_randomOrder": lambda l: reduce(
                    op.add, map(lambda s: len(eval_string_to_set(s)), l)
                ),
                "Failed_randomOrder": lambda l: reduce(
                    op.add, map(lambda s: len(eval_string_to_set(s)), l)
                ),
                "Error_randomOrder": lambda l: reduce(
                    op.add, map(lambda s: len(eval_string_to_set(s)), l)
                ),
                "Skipped_randomOrder": lambda l: reduce(
                    op.add, map(lambda s: len(eval_string_to_set(s)), l)
                ),
                "numRuns_randomOrder": sum,
                "Flaky_sameOrder_withinIteration": any,
                "Flaky_randomOrder_withinIteration": any,
            }
        )
        test_overview["Verdict_sameOrder"] = test_overview["Verdicts_sameOrder"].apply(
            Verdict.decide_overall_verdict
        )
        test_overview["Verdict_randomOrder"] = test_overview["Verdicts_randomOrder"].apply(
            Verdict.decide_overall_verdict
        )

        self._df.drop(
            ["Flaky_sameOrder_withinIteration", "Flaky_randomOrder_withinIteration"],
            axis="columns",
            inplace=True,
        )

        # recalculate Order-dependent
        test_overview["Order-dependent"] = (~test_overview["Flaky_sameOrder_withinIteration"]) & (
            test_overview["Flaky_randomOrder_withinIteration"]
        )

        # Infrastructure Flakiness
        #     if a test is order-dependent, it will never be marked as infrastructure flaky,
        #     even if it would fulfill the requirements in the same order test executions
        test_overview["Flaky_Infrastructure"] = (
            (
                # same order
                (test_overview["Verdict_sameOrder"] == Verdict.FLAKY)
                & ~test_overview["Flaky_sameOrder_withinIteration"]
            )
            | (
                # random order
                (test_overview["Verdict_sameOrder"] != Verdict.FLAKY)
                & (test_overview["Verdict_randomOrder"] == Verdict.FLAKY)
                & (~test_overview["Order-dependent"])
            )
        ) & ~test_overview["Order-dependent"]

        test_overview.insert(
            7,
            "flaky?",
            test_overview.apply(
                lambda s: FlakinessType.decide_flakiness_type(
                    s["Flaky_sameOrder_withinIteration"],
                    s["Order-dependent"],
                    s["Flaky_Infrastructure"],
                ),
                axis="columns",
                result_type="reduce",
            ),
        )

        # modname_classname = test_overview['Test_classname'].apply(
        #     junitxml_classname_to_modname_and_actual_classname
        # )
        # test_overview['Test_actual_classname'] = [c for _, c in modname_classname]
        # test_overview['Test_modname[-1]'] = [
        #     m[-1] if len(m) > 0 else '' for m, _ in modname_classname
        # ]
        test_overview["Test_nodeid"] = test_overview.apply(
            lambda s: to_nodeid(s["Test_filename"], s["Test_classname"], s["Test_name"]),
            axis=1,
            result_type="reduce",
        )
        test_overview["Test_nodeid_inclPara"] = (
            test_overview["Test_nodeid"] + test_overview["Test_parametrization"]
        )
        return test_overview

    def __repr__(self):
        return "PassedFailed"


class TestsOverview:
    def __init__(self, df: pd.DataFrame):
        self._df = df.fillna("")

    @classmethod
    def load(cls, file_name: str):
        """Load TestsOverview from CSV file"""
        _df = pd.read_csv(file_name)
        return cls(_df)

    def to_classification_template(self) -> pd.DataFrame:
        """Prepare a manual classification template for all flaky tests"""
        flaky_tests = self._df[self._df["flaky?"] != FlakinessType.NOT_FLAKY]
        classification_template = flaky_tests[
            [
                "Project_Name",
                "Project_URL",
                "Project_Hash",
                "Test_filename",
                "Test_classname",
                "Test_name",
                "Test_parametrization",
                "flaky?",
            ]
        ].drop_duplicates()
        classification_template["Project Domain"] = ""
        classification_template["Category"] = ""
        classification_template["Category sure? 1=yes 4=no"] = ""
        classification_template["Category comment"] = ""
        return classification_template

    def to_flapy_input(
        self, num_runs: int, *, flakiness_type="ANY", all_tests_in_one_run=True
    ) -> pd.DataFrame:
        """Filter for all flaky tests and create a new flapy-input, which executes all these (in isolation).

        :flakiness_type: see `FlakinessType`
        :all_tests_in_one_run: execute all tests within the same project in the same iteration. Tests will still be run separately, but this saves multiple cloning effort.
        """
        if flakiness_type == "ANY":
            df = self._df
        elif flakiness_type == "ANY_FLAKY":
            df = self._df[self._df["flaky?"] != FlakinessType.NOT_FLAKY]
        else:
            df = self._df[self._df["flaky?"] == flakiness_type]

        if all_tests_in_one_run:
            df = df.groupby(proj_cols)["Test_nodeid"].apply(lambda s: " ".join(s)).reset_index()
        else:
            df = df[proj_cols + ["Test_nodeid"]]

        df["Funcs_to_trace"] = ""
        df["Num_runs"] = num_runs
        df["PyPi_tag"] = ""
        df["Tests_to_be_run"] = df["Test_nodeid"]
        return df[
            [
                "Project_Name",
                "Project_URL",
                "Project_Hash",
                "PyPi_tag",
                "Funcs_to_trace",
                "Test_nodeid",
                "Num_runs",
            ]
        ]


class CoverageOverview(object):

    """Output of ResultsDirCollection.get_coverage_overview"""

    def __init__(self, df):
        self._df = df
        self._df["Project_Hash"] = self._df["Project_Hash"].fillna("")

    @classmethod
    def load(cls, path: str):
        return cls(pd.read_csv(path))

    def group_by_project(self) -> pd.DataFrame:
        """
        Compute the average coverage per project,
        taking into account that different iterations
        have different numbers of runs (weighted average).
        """
        df = self._df.copy()
        num_iteration_with_zero_entries = len(df[df["number_of_entries"] < 1])
        logging.warning(f"Dropped {num_iteration_with_zero_entries} iteration with zero coverage entries")
        df = df[df["number_of_entries"] > 0]

        return df.groupby(proj_cols).apply(lambda x: pd.Series({
            "number_of_runs": sum(x["number_of_entries"]),
            "BranchCoverage": np.average(x["BranchCoverage"], weights=x["number_of_entries"]),
            "LineCoverage": np.average(x["LineCoverage"], weights=x["number_of_entries"])
        }))


class MyFileWrapper(ABC):
    def __init__(
        self,
        path_: Union[str, Path],
        project_name: str,
        openvia: Callable[[str], IO] = open,
        tarinfo: tarfile.TarInfo = None,
        archive: tarfile.TarFile = None,
    ):
        self.p: Path = Path(path_)
        self.project_name: str = project_name
        self.openvia = openvia
        self.archive = archive
        self.tarinfo = tarinfo

    @classmethod
    @abstractmethod
    def get_regex(cls, project_name: str) -> str:
        """
        Regex should have the run number as the first and only group
        :param project_name:
        :return:
        """
        pass

    # TODO is it necessary to check for is empty?
    @classmethod
    def is_(cls, path: Path, project_name: str, openvia: Callable[[str], IO]) -> bool:
        return (
            re.match(cls.get_regex(project_name), str(path))
            is not None
            # and
            # (not is_empty(openvia, str(path))
        )

    @lru_cache()
    def get_num(self) -> int:
        num = re.findall(self.get_regex(self.project_name), str(self.p))[0][0]
        return int(num)

    @lru_cache()
    def get_test_to_be_run(self) -> str:
        return re.findall(self.get_regex(self.project_name), str(self.p))[0][1]

    def open(self) -> IO:
        f = self.openvia(str(self.p))
        if f is None:
            raise ValueError("openvia returned None")
        return f

    def read(self) -> str:
        with self.open() as f:
            content = f.read()
            if isinstance(content, bytes):
                content = content.decode()
            return content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.p}')"


class CoverageXmlFile(MyFileWrapper):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/{project_name}_coverage(\d+)(.*)\.xml$"

    def get_order(self) -> str:
        if CoverageXmlFileSameOrder.is_(self.p, self.project_name, self.openvia):
            return "same"
        if CoverageXmlFileRandomOrder.is_(self.p, self.project_name, self.openvia):
            return "random"
        return "COULD_NOT_GET_ORDER"

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        """
        Transform Junit XML files into a table that shows the verdict and message for each run.
        :return:
            EXAMPLE:
                {
                    'num': 0,
                    'order': "same",
                    "BranchCoverage": 0.7,
                    "LineCoverage": 0.6
                }
        """
        try:
            with self.open() as f:
                root = ET.parse(f).getroot()
            return {
                "num": self.get_num(),
                "order": self.get_order(),
                "BranchCoverage": float(root.get("branch-rate")),  # type: ignore
                "LineCoverage": float(root.get("line-rate")),  # type: ignore
            }
        except Exception as ex:
            logging.error(f"{type(ex).__name__} in {self.p}: {ex}")
            return {}


class CoverageXmlFileSameOrder(CoverageXmlFile):
    @classmethod
    def get_regex(cls, project_name: str):
        # 'deterministic' was the legacy name sameOrder
        return rf".*/(?:deterministic|sameOrder)/tmp/{project_name}_coverage(\d+)(.*)\.xml"


class CoverageXmlFileRandomOrder(CoverageXmlFile):
    @classmethod
    def get_regex(cls, project_name: str):
        # 'non-deterministic' was the legacy name randomOrder
        return rf".*/(?:non-deterministic|randomOrder)/tmp/{project_name}_coverage(\d+)(.*)\.xml"


class CoverageSqliteFile(MyFileWrapper):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/{project_name}_coverage(\d+)(.*)\.sqlite$"

    def get_order(self) -> str:
        if CoverageSqliteFileSameOrder.is_(self.p, self.project_name, self.openvia):
            return "same"
        if CoverageSqliteFileRandomOrder.is_(self.p, self.project_name, self.openvia):
            return "random"
        return "COULD_NOT_GET_ORDER"

    def get_linebits(self) -> pd.DataFrame:
        # -- sqlite cannot open a file handle, but needs a file name -> extract db file first
        with tempfile.TemporaryDirectory() as tmpdir:
            self.archive.extract(str(self.p), tmpdir)

            # -- Querrying database
            conn = sqlite3.connect(Path(tmpdir) / self.p)
            register_sqlite_functions(conn)
            df = pd.read_sql_query(
                # "select file_id, context_id, numbits_to_nums(numbits) from line_bits"
                "select path, context, numbits_to_nums(numbits) as lines "
                "from line_bits "
                "join context on context.id=context_id "
                "join file on file.id=file_id",
                conn,
            )
            df["lines"] = df["lines"].apply(literal_eval)
            df = df.replace("", np.NaN)
            return df

    def to_table(self, *, drop_empty_context=False, drop_execution_stages=False) -> pd.DataFrame:
        """
        drop_execution_stages: e.g. 'test_foo|run' -> 'test_foo'; 'test_bar|setup' -> 'test_bar'
            In this case, we also need to group the columns together
            that address the same test via an any operator
        """
        df = self.get_linebits()
        df["context"] = df["context"].fillna("EMPTY_CONTEXT")
        df = df.explode("lines")
        df = df.dropna(subset="lines")
        df["covered"] = True
        df = df.pivot(index=["path", "lines"], columns="context", values="covered")
        df = df.fillna(False)
        if drop_empty_context:
            df.drop(columns="EMPTY_CONTEXT", inplace=True, errors="ignore")
        if drop_execution_stages:
            if not df.empty:
                df.rename(columns=lambda col: re.sub("\|.*", "", col), inplace=True)
                df = df.groupby(level=0, axis=1).any()
                # Groupby somehow removes the index names
                df.index.set_names(["path", "lines"], inplace=True)
        return df


class CoverageSqliteFileSameOrder(CoverageSqliteFile):
    @classmethod
    def get_regex(cls, project_name: str):
        # 'deterministic' was the legacy name sameOrder
        return rf".*/(?:deterministic|sameOrder)/tmp/{project_name}_coverage(\d+)(.*)\.sqlite$"


class CoverageSqliteFileRandomOrder(CoverageSqliteFile):
    @classmethod
    def get_regex(cls, project_name: str):
        # 'non-deterministic' was the legacy name randomOrder
        return (
            rf".*/(?:non-deterministic|randomOrder)/tmp/{project_name}_coverage(\d+)(.*)\.sqlite$"
        )


class JunitXmlFile(MyFileWrapper):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/tmp/{project_name}_output(\d+)(.*)\.xml$"

    @lru_cache()
    def get_order(self) -> str:
        if JunitXmlFileSameOrder.is_(self.p, self.project_name, self.openvia):
            return "same"
        if JunitXmlFileRandomOrder.is_(self.p, self.project_name, self.openvia):
            return "random"
        return "COULD_NOT_GET_ORDER"

    def parse(self) -> Union[junitparser.JUnitXml, junitparser.TestSuite]:
        """
        Read the file and parse it via junitparser.
        Some JUnit XML files have three hiearchie levels: testsuites, testsuite, testcase
        others skip the first one and start with testsuite
        """
        with self.open() as f:
            try:
                return junitparser.JUnitXml.fromfile(f, ET.parse)
            except xml.etree.ElementTree.ParseError as ex:
                logging.error(f"ParseError in {self.p}: {ex}")
                return junitparser.JUnitXml()

    def get_hostname(self) -> str:
        return list(self.parse())[0].hostname

    def get_testcases(self) -> List[junitparser.TestCase]:
        junit_xml = self.parse()
        if isinstance(junit_xml, junitparser.TestSuite):
            test_cases = [case for case in junit_xml]
        else:
            test_cases = [case for suite in junit_xml for case in suite]
        return test_cases

    def to_table(self, include_num_ttbr_order=True) -> List[Dict[str, Union[str, int]]]:
        """
        Transform Junit XML files into a table that shows the verdict and message for each run.
        :return:
            EXAMPLE [
                {
                    'file': 'test_file',
                    'class': 'test_class',
                    'func': 'test_func',
                    'verdict': 'Failed',
                    'errors': '[AssertionError]'
                    'contains_no_space_left': False,
                    'num': 0
                },
                { ... }
            ]
        """
        try:
            test_cases = self.get_testcases()
            # if len(test_cases) == 0:
            #     logging.warning(f"{self.p} contains no testcases")
            if include_num_ttbr_order:
                result: List[Dict[str, Union[str, int]]] = [
                    {
                        **read_junit_testcase(test_case),
                        "num": self.get_num(),
                        "test_to_be_run": self.get_test_to_be_run(),
                        "order": self.get_order(),
                    }
                    for test_case in test_cases
                ]
            else:
                result: List[Dict[str, Union[str, int]]] = [
                    {
                        **read_junit_testcase(test_case),
                    }
                    for test_case in test_cases
                ]

            return result
        except Exception as ex:
            logging.error(f"{type(ex).__name__} in {self.p}: {ex}")
            return []


class JunitXmlFileSameOrder(JunitXmlFile):
    @classmethod
    def get_regex(cls, project_name: str):
        # 'deterministic' was the legacy name sameOrder
        return rf".*/(?:deterministic|sameOrder)/tmp/{project_name}_output(\d+)(.*)\.xml"


class JunitXmlFileRandomOrder(JunitXmlFile):
    @classmethod
    def get_regex(cls, project_name: str):
        # 'non-deterministic' was the legacy name randomOrder
        return rf".*/(?:non-deterministic|randomOrder)/tmp/{project_name}_output(\d+)(.*)\.xml"


class TraceFile(MyFileWrapper):
    """File containing traces"""

    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/{project_name}_trace(\d+)_.+\.txt$"

    def get_test_funcdescriptor(self) -> FuncDescriptor:
        """Extracts the name of a Test from a given trace-file"""
        with self.open() as f:
            try:
                first_line = next(f)
            except StopIteration:
                return ("EMPTY_FILE", "", "")
            call_return, depth, func, _, _ = parse_trace_line(first_line)
            assert call_return == "call"
            assert depth == 1
            return func

    def grep(self, pattern: str) -> Optional[str]:
        """Greps for a string.

        :pattern: String, no regex
        :returns: First line containing the pattern

        """
        with self.open() as f:
            for line in f:
                line = str(line)
                if pattern in line:
                    return line
        return None


T1 = TypeVar("T1", bound=MyFileWrapper)


class Iteration:
    """
    Example: flapy-results_20200101/lineflow
    """

    archive_name = "results.tar.xz"
    meta_file_name = "flapy-iteration-result.yaml"

    def __init__(self, path: Union[str, Path]):
        self.p = Path(path)
        self._archive: Optional[tarfile.TarFile] = None

        # Check if this is a valid iteration
        if not self.p.is_dir():
            raise ValueError(f"{self.p} is not a directory")
        if self.p.name == "run":
            raise ValueError(f"Folders named 'run' are not considered iterations (legacy)")
        if self.p.name.startswith("."):
            raise ValueError(f"Folders whose names start with '.' are not considered iterations")
        if not (self.p / self.archive_name).is_file():
            # TODO: maybe raise this exception later (like inside get_junit_data, or just set
            # status). The meta file might still be present and its information might be interesting
            raise ValueError(f"{self.p} contains no results archive ({self.archive_name})")

        # Read meta info if available (only in newer versions, older versions use separate files)
        self.meta_file = self.p / self.meta_file_name
        if self.meta_file.is_file():
            with open(self.meta_file) as f:
                self.meta_info = yaml.safe_load(f)
        else:
            self.meta_info = None

        # Retrieve basic information to raise heat cache and raise possible errors now
        self.get_project_name()
        self.get_project_url()
        self.get_project_git_hash()

        # Setup cache
        self._results_cache = self.p / "flapy.cache"
        if not self._results_cache.is_dir():
            self._results_cache.mkdir()
        self._junit_cache_file = self._results_cache / "junit_data.csv"

    def has_archive(self):
        """If the results have not been written back (e.g., due to a timeout), there is no resultar.tar.xz, however, the directory with the meta infos is still counted as a failed attempt and therefore an iteration."""
        return (self.p / self.archive_name).is_file()

    @lru_cache()
    def get_project_name(self) -> str:
        if self.meta_info is not None:
            return self.meta_info["project_name"]
        elif (self.p / "project-name.txt").is_file():
            with open(self.p / "project-name.txt") as file:
                return file.read().replace("\n", "")
        else:
            raise ValueError("Could not retrieve project name")

    def get_project_url(self) -> str:
        if self.meta_info is not None:
            return self.meta_info["project_url"]
        elif (self.p / "project-url.txt").is_file():
            with open(self.p / "project-url.txt") as file:
                return file.read().replace("\n", "")
        else:
            raise ValueError("Could not retrieve project URL")

    def get_project_git_hash(self) -> str:
        if self.meta_info is not None:
            return self.meta_info["project_git_hash"]
        elif (self.p / "project-git-hash.txt").is_file():
            with open(self.p / "project-git-hash.txt") as file:
                return file.read().replace("\n", "")
        else:
            raise ValueError("Could not retrieve project hash")

    def get_flapy_git_hash(self) -> str:
        # Flapy used to be called 'flakyanalysis'
        if (self.p / "flakyanalysis-git-hash.txt").is_file():
            with open(self.p / "flakyanalysis-git-hash.txt") as file:
                return file.read().replace("\n", "")
        return "COULD_NOT_GET_FLAKYANALYSIS_GIT_HASH"

    def get_iterations_info(self) -> Dict[str, Any]:
        return {
            "Iteration": self,
            "Project_Name": self.get_project_name(),
            "Project_URL": self.get_project_url(),
            "Project_Hash": self.get_project_git_hash(),
        }

    def get_lines_of_code(self, languages=["Python"], metrics=["code"]) -> Dict[str, Optional[int]]:
        """Read lines-of-code information

        :languages: languages to filter for (Markdown, YAML, Python, ...)
            Must be specified as one string carrying a list, e.g. "[Python, Markdown]"
            If the project does not use this language, the respective keys are mapped to None
        :metrics: type of lines to filter for
            Must be a subset of [files, blank, comment, code, total]
            Must be specified as one string carrying a list
        :returns: Dictionary mapping "language_metric" to the respective number of lines

        """
        LOC_METRICS = ["files", "blank", "comment", "code", "total"]
        for metric in metrics:
            if metric not in LOC_METRICS:
                raise ValueError(f"Unknown metric {metric}. Must be in {LOC_METRICS}")

        loc_df = pd.read_csv(self.p / "loc.csv").set_index("language")
        loc_dict = dict()

        for language in languages:
            for metric in metrics:
                loc_dict[f"{language}_{metric}"] = loc_df[metric].get(language)

        return loc_dict

    def clear_results_cache(self):
        for file_ in self._results_cache.iterdir():
            file_.unlink()

    def clear_junit_data_cache(self):
        if self._junit_cache_file.is_file():
            self._junit_cache_file.unlink()

    def get_junit_data(
        self,
        *,
        include_project_columns: bool = False,
        read_cache=True,
        write_cache=True,
        return_nothing=False,
    ) -> pd.DataFrame:

        did_read_cache = False
        if read_cache and self._junit_cache_file.is_file():
            junit_data: pd.DataFrame = pd.read_csv(self._junit_cache_file)
            did_read_cache = True
        else:
            columns = list(read_junit_testcase(junitparser.TestCase()).keys()) + ["num", "order"]
            junitxml_files = self.get_files(JunitXmlFile)
            junit_data = pd.DataFrame(
                [
                    test_case
                    for junit_xml_file in junitxml_files
                    for test_case in junit_xml_file.to_table()
                ]
            )
            if len(junit_data) == 0:
                junit_data = pd.DataFrame(columns=columns)
        if write_cache and not did_read_cache and len(junit_data) > 0:
            junit_data.to_csv(self._junit_cache_file, index=False)
        if include_project_columns:
            junit_data.insert(
                0,
                "Project_Hash",
                try_default(lambda: self.get_project_git_hash(), Exception, "error"),
            )
            junit_data.insert(0, "Project_URL", self.get_project_url())
            junit_data.insert(0, "Project_Name", self.get_project_name())
            junit_data.insert(0, "Iteration", self.p.name)

        self.close_archive()
        if return_nothing:
            return None
        return junit_data.fillna("")

    def get_passed_failed(
        self, *, read_cache=True, write_cache=True, verdict_cols_to_strings=True
    ) -> pd.DataFrame:
        try:
            junit_data = self.get_junit_data(
                include_project_columns=False, read_cache=read_cache, write_cache=write_cache
            )
        except Exception as e:
            logging.error(f"{type(e).__name__}: {e} in {self.p}")
            return pd.DataFrame(columns=PassedFailed.columns)

        junit_data.insert(0, "Iteration", self.p.name)
        junit_data.insert(1, "Project_Name", self.get_project_name())
        junit_data.insert(2, "Project_URL", self.get_project_url())
        junit_data.insert(3, "Project_Hash", self.get_project_git_hash())

        junit_data["Test_filename"] = junit_data["file"]
        junit_data["Test_classname"] = junit_data["class"]
        junit_data["Test_name"] = [re.sub(r"\[.*\]", "", name) for name in junit_data["name"]]
        junit_data["Test_parametrization"] = [
            re.findall(r"(\[.*\])", name)[0] if len(re.findall(r"(\[.*\])", name)) > 0 else ""
            for name in junit_data["name"]
        ]
        junit_data["Passed"] = junit_data.apply(
            lambda s: s["num"] if s["verdict"] == Verdict.PASS else -1, axis=1
        )
        junit_data["Failed"] = junit_data.apply(
            lambda s: s["num"] if s["verdict"] == Verdict.FAIL else -1, axis=1
        )
        junit_data["Error"] = junit_data.apply(
            lambda s: s["num"] if s["verdict"] == Verdict.ERROR else -1, axis=1
        )
        junit_data["Skipped"] = junit_data.apply(
            lambda s: s["num"] if s["verdict"] == Verdict.SKIP else -1, axis=1
        )

        passed_failed = junit_data.groupby(
            [
                "Iteration",
                "Project_Name",
                "Project_URL",
                "Project_Hash",
                "Test_filename",
                "Test_classname",
                "Test_name",
                "Test_parametrization",
                "order",
            ],
            as_index=False,
            group_keys=False,
        ).agg(
            {
                "Passed": lambda l: {x for x in l if x != -1},
                "Failed": lambda l: {x for x in l if x != -1},
                "Error": lambda l: {x for x in l if x != -1},
                "Skipped": lambda l: {x for x in l if x != -1},
            }
        )
        # Handle cases where the same testcase passed and failed in the same run (notebook tests)
        passed_failed["Passed"] = passed_failed["Passed"] - passed_failed["Failed"]

        passed_failed["Verdicts"] = [
            Verdict.verdict_set_from_num_sets(p, f, e, s)
            for p, f, e, s in zip(
                passed_failed["Passed"],
                passed_failed["Failed"],
                passed_failed["Error"],
                passed_failed["Skipped"],
            )
        ]

        passed_failed["numRuns"] = [
            len(p.union(f).union(e).union(s))
            for p, f, e, s in zip(
                passed_failed["Passed"],
                passed_failed["Failed"],
                passed_failed["Error"],
                passed_failed["Skipped"],
            )
        ]

        passed_failed["Verdict"] = passed_failed["Verdicts"].apply(Verdict.decide_overall_verdict)

        if verdict_cols_to_strings:
            for col in ["Passed", "Failed", "Error", "Skipped", "Verdicts"]:
                passed_failed[col] = passed_failed[col].apply(str)

        passed_failed = pd.merge(
            passed_failed[passed_failed["order"] == "same"],
            passed_failed[passed_failed["order"] == "random"],
            on=[
                "Iteration",
                "Project_Name",
                "Project_URL",
                "Project_Hash",
                "Test_filename",
                "Test_classname",
                "Test_name",
                "Test_parametrization",
            ],
            how="outer",
            suffixes=("_sameOrder", "_randomOrder"),
        )

        passed_failed.insert(1, "Iteration_status", "ok")

        return passed_failed[PassedFailed.columns]

    def get_coverage_raw_data(self) -> List[Dict]:
        """Read branch- and line-coverage from coverage-xml

        :returns: EXAMPLE:
            [
                { 'num': 0, 'order': "same", "BranchCoverage": 0.7, "LineCoverage": 0.6 },
                ...
            ]
        """
        coverage_data = list(
            filter(
                lambda d: len(d) > 0,
                map(CoverageXmlFile.to_dict, self.get_files(CoverageXmlFile)),
            )
        )
        self.close_archive()
        return coverage_data

    def get_coverage_overview(self) -> pd.DataFrame:
        coverage_data = pd.DataFrame(self.get_coverage_raw_data())

        coverage_overview = pd.DataFrame(
            [
                {
                    "Iteration": self.p.name,
                    "Project_Name": self.get_project_name(),
                    "Project_URL": self.get_project_url(),
                    "Project_Hash": self.get_project_git_hash(),
                    "number_of_entries": len(coverage_data),
                }
            ]
        )
        if len(coverage_data) > 0:
            coverage_overview["BranchCoverage"] = coverage_data["BranchCoverage"].mean()
            coverage_overview["LineCoverage"] = coverage_data["LineCoverage"].mean()
        else:
            coverage_overview["BranchCoverage"] = -1
            coverage_overview["LineCoverage"] = -1
        return coverage_overview

    def get_meta_overview(self) -> Dict:
        """Collect meta informations (e.g. runtime) of iteration
        :returns: TODO

        """
        return {
            "Iteration": self.p,
            **self.meta_info,
        }

    def get_files(self, type_: Type[T1]) -> List[T1]:
        if self.has_archive():
            return [
                type_(
                    path_=tarinfo.name,
                    project_name=self.get_project_name(),
                    openvia=self.get_archive().extractfile,  # type: ignore
                    tarinfo=tarinfo,
                    archive=self.get_archive(),
                )
                for tarinfo in self.get_archive_members()
                if type_.is_(
                    path=Path(tarinfo.name),
                    project_name=self.get_project_name(),
                    openvia=self.get_archive().extractfile,  # type: ignore
                )
            ]
        else:
            return []

    def get_junit_files(self) -> List[JunitXmlFile]:
        return self.get_files(JunitXmlFile)

    def get_trace_files(self) -> List[TraceFile]:
        return self.get_files(TraceFile)

    def get_coverage_sqlite_files(self) -> List[CoverageSqliteFile]:
        return self.get_files(CoverageSqliteFile)

    def get_archive(self) -> tarfile.TarFile:
        if self._archive is None:
            self._archive = tarfile.open(self.p / self.archive_name)
        return self._archive

    def get_archive_members(self) -> List[tarfile.TarInfo]:
        try:
            return self.get_archive().getmembers()
        except EOFError:
            logging.warning(f"EOFError in archive.getnames() on {self.p}")
            return []

    def get_archive_names(self) -> List[str]:
        """Wrapper to around self.get_archive().getnames() to avoid EOFError
        :returns: names of files in archive
        """
        try:
            return self.get_archive().getnames()
        except EOFError:
            logging.warning(f"EOFError in archive.getnames() on {self.p}")
            return []

    def find_keywords_in_tracefiles(
        self, keywords: List[str] = default_flaky_keywords
    ) -> Dict[str, Union[str, bool]]:
        logging.debug(f"{self.p.name} {self.get_project_name()}")
        trace_files = self.get_files(TraceFile)
        if len(trace_files) > 0:
            tf = trace_files[0]
            search = {kw: tf.grep(kw) is not None for kw in keywords}
            test_funcdescr = tf.get_test_funcdescriptor()
        else:
            search = {kw: False for kw in keywords}
            test_funcdescr = ("NO TRACE FILES FOUND", "", "")
        self.close_archive()
        return {
            "Project_Name": self.get_project_name(),
            "Project_URL": self.get_project_url(),
            "Project_Hash": self.get_project_git_hash(),
            "Test_filename": test_funcdescr[0],
            "Test_classname": test_funcdescr[1],
            "Test_name": test_funcdescr[2],
            **search,
        }

    def yield_coverage_table(
        self, drop_execution_stages=False, order=None
    ) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Each testsuite execution creates one coverage-sqlite database.
        This method gradually processes the coverage-sqlite database of each testsuite execution to
        the respective coverage table.

        drop_execution_stages: 'test|run' -> 'test'; 'test|setup' -> 'test'
            In this case, we also need to group the columns together
            that address the same test via an any operator

        :returns: generator producing tuples: (coverage-table, test-outcomes)
        """

        junit_data = self.get_junit_data()

        coverage_files = self.get_files(CoverageSqliteFile)
        if order is not None:
            assert order in ["same", "random"], f"unknown order {order}"
            coverage_files = [cf for cf in coverage_files if cf.get_order() == order]
            junit_data = junit_data[junit_data["order"] == order]
        coverage_files = sorted(coverage_files, key=lambda x: x.get_num())

        for f in coverage_files:
            try:
                ct: pd.DataFrame = f.to_table(
                    drop_empty_context=True, drop_execution_stages=drop_execution_stages
                )
                test_outcomes = junit_data[junit_data["num"] == f.get_num()]
                yield (ct, test_outcomes)
            except Exception as e:
                logging.error(f"{type(e).__name__}: {e} | Iteration=({self.p}), File=({f.p})")
        self.close_archive()

    def get_coverage_table(self, *, include_iteration_path=True, order=None) -> pd.DataFrame:
        """ Retrieve table showing which test case executed which line of source code
        (rows = source code lines, columns = test executions).
        Considering only same order executions.
        """

        # cov_tables = [
        #     f.to_table().rename(
        #         columns=lambda c: (self.p.name, f.get_num(), c)
        #         if include_iteration_path
        #         else (f.get_num(), c)
        #     )
        #     for f in sorted(self.get_files(CoverageSqliteFileSameOrder), key=lambda x: x.get_num())
        # ]
        # self.close_archive()

        cov_tables, _ = list(zip(*list(self.yield_coverage_table())))

        if len(cov_tables) > 0:
            cov_table = pd.concat(cov_tables, axis="columns")
            return cov_table
        else:
            return pd.DataFrame()

    def close_archive(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def __repr__(self) -> str:
        return f"Iteration('{self.p}')"


class IterationCollection(ABC):

    """Abstract class, providing analysis methods that can be performed given iterations"""

    tests_overview = None

    @abstractmethod
    def get_iterations(self) -> List[Iteration]:
        """
        Return list of contained iterations.
        TODO: Docstring for get_iterations.
        :returns: TODO

        """
        pass

    @lru_cache()
    def get_iterations_overview(self) -> pd.DataFrame:
        iterations_overview = (
            pd.DataFrame([it.get_iterations_info() for it in self.get_iterations()])
            .set_index(["Project_Name", "Project_URL", "Project_Hash"])
            .sort_index()
        )
        return iterations_overview

    def get_iterations_meta_overview(self):
        """
        Return table with all meta data of each iteration
        """
        return pd.DataFrame(
            [
                {"path": self.p, "Iteration.parent": it.p.parent, "Iteration": it.p, **it.meta_info}
                for it in self.get_iterations()
            ]
        )

    @abstractmethod
    def get_passed_failed(self) -> pd.DataFrame:
        pass

    def get_tests_overview(
        self,
        *,
        read_resultsDir_cache=False,
        write_resultsDir_cache=True,
        read_iteration_cache=True,
        write_iteration_cache=True,
    ) -> TestsOverview:
        # Use cache if possible
        #   I do not use @lru_cache here, because it doesn't work inside pool.map,
        #   since every process hast it's own memory
        if self.tests_overview is None:
            pf = self.get_passed_failed(
                read_resultsDir_cache=read_resultsDir_cache,
                write_resultsDir_cache=write_resultsDir_cache,
                read_iteration_cache=read_iteration_cache,
                write_iteration_cache=write_iteration_cache,
            )
            to = PassedFailed(pf).to_tests_overview()
            self.tests_overview = TestsOverview(to)
        return self.tests_overview

    def get_junit_data(
        self,
        *,  # After this star, there must only be parameters with default values. Needed for fire.
        include_project_columns=True,
        read_cache=True,
        write_cache=True,
        return_nothing=False,
    ) -> pd.DataFrame:
        with multiprocessing.Pool() as pool:
            return pd.concat(
                pool.map(
                    partial(
                        Iteration.get_junit_data,
                        include_project_columns=include_project_columns,
                        read_cache=read_cache,
                        write_cache=write_cache,
                        return_nothing=return_nothing,
                    ),
                    self.get_iterations(),
                )
            )

    def get_coverage_overview(self) -> pd.DataFrame:
        pool = multiprocessing.Pool()
        co = pd.concat(pool.map(Iteration.get_coverage_overview, self.get_iterations()))
        return co

    def get_meta_overview(self) -> pd.DataFrame:
        result = pd.DataFrame([it.get_meta_overview() for it in self.get_iterations()])
        return result

    def get_lines_of_code_overview(self, *, languages=["Python"], metrics=["code"]) -> pd.DataFrame:
        """Collect LoC statistics from all iterations"""
        result = []
        for it in self.get_iterations():
            it_result = it.get_iterations_info()
            try:
                loc = it.get_lines_of_code(languages, metrics)
                it_result.update({"LoC_status": "ok"})
                it_result.update(loc)
            except Exception:
                it_result.update({"LoC_status": "error"})
            result.append(it_result)
        return pd.DataFrame(result)

    def get_accum_coverage_table_for_project(
        self, proj_name: str, proj_url: str, proj_hash: str, order: str
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generating the entire coverage table, which contains all runs of all iterations,
        is often memory-wise not feasible. This functions accumulates the coverage table to
        ct_accum, adding all test executions up and showing for each test case (columns),
        how often it executed each line of source code (rows). To infer, if the test case covered
        a certain line in all its executions, this function also returns the table
        tests_num_executions (tne), that shows how many times each test has been executed.
        """
        # Filter for iterations that executed the specified project
        #   I don't use .loc here, because it has some weird behavior:
        #   sometimes it return a DataFrame, sometimes a Series
        proj_dirs = self.get_iterations_overview().reset_index()
        match = proj_dirs[
            (proj_dirs["Project_Name"] == proj_name)
            & (proj_dirs["Project_URL"] == proj_url)
            & (proj_dirs["Project_Hash"] == proj_hash)
        ]
        proj_iterations = match["Iteration"]

        ct_accum = None
        tests_num_executions = pd.Series()
        for it in proj_iterations:
            logging.debug(it.p.name)
            for ct, _ in it.yield_coverage_table(drop_execution_stages=True, order=order):
                ct = ct.astype(int)

                executed_tests = pd.Series(1, index=ct.columns)
                tests_num_executions = tests_num_executions.add(executed_tests, fill_value=0)

                if ct_accum is None:
                    ct_accum = ct
                else:
                    ct_accum = ct_accum.add(ct, fill_value=0)
        if ct_accum is None:
            logging.warning(f"ct_accum was None in project {proj_name}, {proj_url}, {proj_hash}")
            ct_accum = pd.DataFrame()
        ct_accum = ct_accum.fillna(0).astype(int)
        tests_num_executions.index.name = "Test_nodeid_inclPara"
        tests_num_executions.name = "execution_count"
        return tests_num_executions, ct_accum

    def get_first_coverage_table_for_project(
        self, proj_name: str, proj_url: str, proj_hash: str, order: str
    ):
        proj_dirs = self.get_iterations_overview().reset_index()
        match = proj_dirs[
            (proj_dirs["Project_Name"] == proj_name)
            & (proj_dirs["Project_URL"] == proj_url)
            & (proj_dirs["Project_Hash"] == proj_hash)
        ]
        proj_iterations = match["Iteration"]

        ct_accum = None
        tests_num_executions = pd.Series()

        it = sorted(proj_iterations, key=lambda x: x.p)[0]
        logging.debug(it.p.name)
        ct, _ = next(it.yield_coverage_table(drop_execution_stages=True, order=order))

        ct = ct.astype(int)

        executed_tests = pd.Series(1, index=ct.columns)
        tests_num_executions = tests_num_executions.add(executed_tests, fill_value=0)

        if ct_accum is None:
            ct_accum = ct
        else:
            ct_accum = ct_accum.add(ct, fill_value=0)

        if ct_accum is None:
            logging.warning(f"ct_accum was None in project {proj_name}, {proj_url}, {proj_hash}")
            ct_accum = pd.DataFrame()
        ct_accum = ct_accum.fillna(0).astype(int)
        tests_num_executions.index.name = "Test_nodeid_inclPara"
        tests_num_executions.name = "execution_count"
        return tests_num_executions, ct_accum

    def save_cta_for_project(
        self,
        proj_name: str,
        proj_url: str,
        proj_hash: str,
        cta_save_dir: str,
        flaky_col: str,
        method: str,
    ):
        # Verify inputs
        if flaky_col == "Flaky_sameOrder_withinIteration":
            order = "same"
        elif flaky_col == "Order-dependent":
            order = "random"
        else:
            raise ValueError(
                "Unknown flakiness column, "
                "select 'Flaky_sameOrder_withinIteration' or 'Order-dependent'"
            )

        if method == "accum":
            get_cov_table_method = self.get_accum_coverage_table_for_project
        elif method == "first":
            get_cov_table_method = self.get_first_coverage_table_for_project
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'accum' or 'first'")

        # Calculate coverage table
        try:
            tne, cta = get_cov_table_method(proj_name, proj_url, proj_hash, order)

            to = self.get_tests_overview()._df
            matching_to = to[
                (to["Project_Name"] == proj_name)
                & (to["Project_URL"] == proj_url)
                & (to["Project_Hash"] == proj_hash)
            ]
            flakiness_status = matching_to.set_index("Test_nodeid_inclPara", verify_integrity=True)[
                flaky_col
            ]
        except Exception as e:
            logging.error(
                f"{type(e).__name__}: '{e}' in project {proj_name}, {proj_url}, {proj_hash}"
            )
            logging.error(traceback.format_exc())
            return None

        # Export
        url_without_slash = proj_url.replace("/", " ")
        cta_save_dir = Path(cta_save_dir)
        cta_save_dir.mkdir(exist_ok=True)
        cta.to_csv(cta_save_dir / f"{url_without_slash}@{proj_hash}_cta.csv")
        tne.reset_index().merge(
            flakiness_status.reset_index(), on="Test_nodeid_inclPara", how="outer"
        ).to_csv(cta_save_dir / f"{url_without_slash}@{proj_hash}_TneFs.csv", index=False)

    def save_cta_tables(self, flaky_col: str, method: str, cta_save_dir: str):
        """Save accumulated coverage tables (cta) to directory

        :flaky_col: "Flaky_sameOrder_withinIteration" / "Order-dependent" / "Flaky_Infrastructure"
            You have to provide a this, because it determines which runs are going to be use:
            NOD flaky (Flaky_sameOrder_withinIteration) -> same order runs
            OD flaky (Order-dependent) -> random order runs
        :method: "accum" or "first"

        """
        # Always calculate the tests_overview, even if you don't use it.
        #   Reason: it refreshes the PassedFailed-cache, which the calculation of the
        #   suspiciousness tables needs, however, this should not be done by multiple
        #   threads.
        to = self.get_tests_overview()._df

        projs = to[to[flaky_col]][proj_cols].drop_duplicates().values.tolist()
        logging.info(f"Saving suspiciousness tables for {len(projs)} projects")

        pool = multiprocessing.Pool()
        list(tqdm(
            pool.starmap(
                partial(
                    self.save_cta_for_project,
                    cta_save_dir=cta_save_dir,
                    flaky_col=flaky_col,
                    method=method,
                ),
                projs,
            )
        ))

    def _save_flaky_failures_iteration(self, row: pd.Series, outcomes: List[str], save_dir: str) -> None:
        """Helper function to parallelize save_flaky_failures

        :row: row from PassedFailed CSV of a flaky iteration
        :outcomes: which outcomes are you interested in? e.g. ["Failed_sameOrder"]
        :save_dir: directory to save the junit files to
        :returns: None
        """
        try:
            iteration = Iteration(row["Iteration"])

            for outcome in outcomes:
                files = [
                    x
                    for x in iteration.get_files(JunitXmlFile)
                    if x.get_num() in eval_string_to_set(row[outcome])
                ]
                url_without_slash = row["Project_URL"].replace("/", " ")
                git_hash = row["Project_Hash"]
                test_id = to_nodeid(*row[test_cols]).replace("/", " ")
                proj_savedir = (
                    Path(save_dir) / f"{url_without_slash}@{git_hash[:6]}" / test_id / outcome
                )
                for file_ in files:
                    file_.tarinfo.name = Path(iteration.p.name) / os.path.basename(
                        file_.tarinfo.name
                    )
                    iteration.get_archive().extract(file_.tarinfo, proj_savedir)

            iteration.close_archive()
        except Exception as e:
            logging.error(f"{type(e).__name__}: '{e}' in iteration {row['Iteration']}")

    def save_flaky_failures(
        self, save_dir: str, outcome: str, flaky_types: List[str], proj_csv: str = None
    ) -> None:
        """Extract all junit files in which a flaky test failed.

        This can be useful to find the root cause of the flaky failure.

        :save_dir: directory to save the junit files to
            They will be saved in the following format:

                save_dir/FLAKY_TYPE/PROJ_URL@PROJ_HASH/TEST_NODEID/OUTCOME/ITERATION/JUNIT_FILE

            Slashes in PROJ_URL are being replaced with spaces
            NOTE: if the project-url starts with ".", e.g. a local path ("./minimal_example"), the folder will be hidden

        :outcome: which outcomes do you want to filter for (same, random, any)

        :flaky_type: what kind of flaky tests do you want to consider?
            values: (order-dependent, non-order-dependent, infrastructure flaky, not flaky)
            KNOWN ISSUE: you cannot properly pass a list as an argument, because fire interprets the "-" in "order-dependent"

        :proj_csv: only consider projects in this csv file (needs `proj_cols`)
        """

        # Check input arguments
        if isinstance(flaky_types, str):
            flaky_types = [flaky_types]
        for flaky_type in flaky_types:
            if flaky_type not in FlakinessType.all_types:
                raise ValueError(f"Unknown flaky type '{flaky_type}'")

        logging.info("Generate tests overview")
        passed_failed = PassedFailed(self.get_passed_failed())
        to = passed_failed.to_tests_overview()

        logging.info("Starting export")
        # Filter for specified projects
        if proj_csv is not None:
            proj_df = pd.read_csv(proj_csv)[proj_cols]
            to = to.merge(proj_df)

        if outcome == "random":
            outcomes = [
                # "Passed_randomOrder",
                "Failed_randomOrder",
                "Error_randomOrder",
                # "Skipped_randomOrder",
            ]
        elif outcome == "same":
            outcomes = [
                # "Passed_sameOrder",
                "Failed_sameOrder",
                "Error_sameOrder",
                # "Skipped_sameOrder",
            ]
        elif outcome == "any":
            outcomes = [
                # "Passed_sameOrder",
                "Failed_sameOrder",
                "Error_sameOrder",
                # "Skipped_sameOrder",
                # "Passed_randomOrder",
                "Failed_randomOrder",
                "Error_randomOrder",
                # "Skipped_randomOrder",
            ]
        else:
            raise ValueError(f"Unknown value for argument outcome")

        for f_type in flaky_types:
            # Filter for flaky tests
            flaky_tests = to[to["flaky?"] == f_type][proj_cols + test_cols]
            pf_flaky = passed_failed._df.merge(flaky_tests)

            pool = multiprocessing.Pool()
            pool.map(
                partial(
                    self._save_flaky_failures_iteration,
                    outcomes=outcomes,
                    save_dir=f"{save_dir}/{f_type}",
                ),
                map(lambda x: x[1], pf_flaky.iterrows()),
            )


class ResultsDir(IterationCollection):
    """
    Directory created by one execution of flapy.
    Example: flapy-results_20200101_1430
    """

    def __init__(self, path: Union[str, Path]):
        self.p = Path(path)
        assert self.p.is_dir(), f"Directory {self.p} does not exist"

        # Setup cache
        #   we use the exclamation mark, because it has a early position in the ascii table, so
        #   the cache file appears at the top of the directory
        self._results_cache = self.p / "!flapy.cache"
        if not self._results_cache.is_dir():
            self._results_cache.mkdir()
        self._pf_cache_file = self._results_cache / "passed_failed.csv"

    def get_iterations(self) -> List[Iteration]:
        """
        Return all valid iterations contained in this ResultsDir
        """
        iterations = [
            try_default(lambda: Iteration(path), log_error_info=path) for path in self.p.glob("*")
        ]
        iterations = [it for it in iterations if it is not None]
        return iterations

    def clear_results_cache(self):
        for it in self.get_iterations():
            it.clear_results_cache()

    def clear_junit_data_cache(self):
        for it in self.get_iterations():
            it.clear_junit_data_cache()

    def get_passed_failed(
        self,
        *,
        read_resultsDir_cache=False,
        write_resultsDir_cache=True,
        read_iteration_cache=True,
        write_iteration_cache=True,
    ) -> pd.DataFrame:
        """
        Cached version of self._compute_passed_failed.

        The resultsDir_cache is RESULTS_DIR/!flapy.cache/passed_failed.csv
            and it is procesed within this method.
        The iteration_cache is RESULTS_DIR/ITERATION/results_cache/junit_data.csv
            and it is processed in Iteration.get_junit_data.
        """

        # TODO: fix resultsDir cache
        #   the problem with the resultsDir cache is that iterations might be found EMPTY
        #   just because they haven't finished yet. Then, even after the iteration has finished,
        #   the results-parser will still assume them to be empty, because it reads from the cache.
        #   Just not including empty iterations in the cache also not good, because some iterations
        #   are legitimately empty, even after finishing.
        #   To mitigate the issue, I just disabled reading from the resulsDir cache for now.

        did_read_cache = False
        cache_file = self._pf_cache_file

        if read_resultsDir_cache and cache_file.is_file():
            logging.info(f"Loading cached passed-failed table for ResultsDir {self.p}")
            cache_df: pd.DataFrame = pd.read_csv(cache_file)

            cache_is_hot = self._is_pf_cache_hot(cache_df=cache_df)
            if cache_is_hot:
                passed_failed = cache_df
                did_read_cache = True
            else:
                logging.info(
                    f"Passed-failed cache of Iteration {self.p} is OUTDATED -> recalculate"
                )
                passed_failed = self._compute_passed_failed(
                    read_iteration_cache, write_iteration_cache
                )
        else:
            passed_failed = self._compute_passed_failed(read_iteration_cache, write_iteration_cache)

        if write_resultsDir_cache and not did_read_cache and len(passed_failed) > 0:
            passed_failed.to_csv(cache_file, index=False)

        return passed_failed

    def _is_pf_cache_hot(self, *, cache_df: pd.DataFrame = None):
        """
        Check if cache is still hot (i.e., if pf.csv contains all iterations that are
        actually in this directory)
        """
        if cache_df is None:
            cache_df: pd.DataFrame = pd.read_csv(self._pf_cache_file)

        # Get iterations into the format in which it is saved in passed_failed.csv
        iterations_in_this_dir = [self.p.name + "/" + it.p.name for it in self.get_iterations()]
        return set(iterations_in_this_dir) == set(cache_df["Iteration"])

    def _compute_passed_failed(self, read_iteration_cache, write_iteration_cache):
        """
        Compute passed-failed table parallelized accross iterations.
        """
        if len(self.get_iterations()) == 0:
            logging.warning(f"ResultsDir {self} contains no iterations")
            return pd.DataFrame(columns=PassedFailed.columns)
        with multiprocessing.Pool() as pool:
            passed_failed = pd.concat(
                pool.map(
                    partial(
                        Iteration.get_passed_failed,
                        read_cache=read_iteration_cache,
                        write_cache=write_iteration_cache,
                    ),
                    self.get_iterations(),
                )
            )
        passed_failed["Iteration"] = self.p.name + "/" + passed_failed["Iteration"]
        return passed_failed

    def get_no_space_left(self) -> pd.DataFrame:
        return pd.concat(map(Iteration.get_no_space_left, self.get_iterations()))

    def find_keywords_in_tracefiles(self, *, keywords=default_flaky_keywords) -> pd.DataFrame:
        pool = multiprocessing.Pool()
        result = pool.map(
            Iteration.find_keywords_in_tracefiles,
            self.get_iterations(),
        )

        if len(result) > 0:
            return (
                pd.DataFrame(result)
                .groupby(
                    [
                        "Project_Name",
                        "Project_URL",
                        "Project_Hash",
                        "Test_filename",
                        "Test_classname",
                        "Test_name",
                    ],
                    as_index=False,
                    group_keys=False,
                )
                .any()
            )
        else:
            return pd.DataFrame(
                columns=[
                    "Project_Name",
                    "Project_URL",
                    "Project_Hash",
                    "Test_filename",
                    "Test_classname",
                    "Test_name",
                ]
            )

    def __repr__(self) -> str:
        return f"ResultsDir('{self.p}')"


class ResultsDirCollection(IterationCollection):

    """Directory containing (symlinks to) ResultsDirs. Assumes that it only containes such."""

    def __init__(self, path):
        self.p = Path(path)
        assert self.p.is_dir(), f"Directory {self.p} does not exist"

    def get_results_dirs(self) -> List[ResultsDir]:
        return [ResultsDir(path) for path in self.p.glob("*")]

    def get_iterations(self):
        return [it for rd in self.get_results_dirs() for it in rd.get_iterations()]

    def get_passed_failed(
        self,
        *,
        read_resultsDir_cache=False,
        write_resultsDir_cache=True,
        read_iteration_cache=True,
        write_iteration_cache=True,
    ) -> pd.DataFrame:
        """
        Collect passed-failed information from each ResultsDir and concat them.
        """
        passed_failed = pd.concat(
            [
                rd.get_passed_failed(
                    read_resultsDir_cache=read_resultsDir_cache,
                    write_resultsDir_cache=write_resultsDir_cache,
                    read_iteration_cache=read_iteration_cache,
                    write_iteration_cache=write_iteration_cache,
                )
                for rd in self.get_results_dirs()
            ]
        )
        passed_failed["Iteration"] = self.p.name + "/" + passed_failed["Iteration"]
        return passed_failed


def parse_string_range(s: str):
    """Example "1-3,5,7" -> [1, 2, 3, 5, 7]"""
    result = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            result.extend(range(a, b + 1))
        else:
            a = int(part)
            result.append(a)
    return result


class FlakinessType:
    """
    Enum for flakiness type.
      I don't use actual enums,
      because they are not equal to their string representations,
      but I need this, when re-reading the 'flaky?' column from TestsOverview.csv
    """

    OD = "order-dependent"
    NOD = "non-order-dependent"
    INFR = "infrastructure flaky"
    NOT_FLAKY = "not flaky"

    all_types = [OD, NOD, INFR, NOT_FLAKY]

    @classmethod
    def decide_flakiness_type(
        cls,
        flaky_sameOrder_withinIteration: bool,
        order_dependent: bool,
        infrastructure: bool,
    ) -> str:
        assert flaky_sameOrder_withinIteration + order_dependent + infrastructure < 2
        if flaky_sameOrder_withinIteration:
            return FlakinessType.NOD
        if order_dependent:
            return FlakinessType.OD
        if infrastructure:
            return FlakinessType.INFR
        return FlakinessType.NOT_FLAKY


class Verdict:
    """
    Enum for test verdict.
      I don't use actual enums,
      because they are not equal to their string representations,
      but I need this, when re-reading test verdicts from a .csv file
    """

    PASS = "Passed"
    FAIL = "Failed"
    ERROR = "Error"
    SKIP = "Skipped"
    FLAKY = "Flaky"

    ZERO_RUNS = "ZeroRuns"

    NOT_ANALYSED = "NotAnalysed"
    NOT_MERGED = "NotMerged"
    PARSE_ERROR = "PARSE_ERROR"

    UNDECIDABLE = "Undecidable"

    @staticmethod
    def from_junitparser(
        result: Union[junitparser.Failure, junitparser.Skipped, junitparser.Error, None]
    ) -> str:
        if isinstance(result, junitparser.Failure):
            return Verdict.FAIL
        if isinstance(result, junitparser.Skipped):
            return Verdict.SKIP
        if isinstance(result, junitparser.Error):
            return Verdict.ERROR
        if result is None:
            return Verdict.PASS
        raise ValueError

    @staticmethod
    def verdict_set_from_num_sets(passed, failed, error, skipped: Set[int]) -> Set[str]:
        result = set()
        if len(passed) > 0:
            result.add(Verdict.PASS)
        if len(failed) > 0:
            result.add(Verdict.FAIL)
        if len(error) > 0:
            result.add(Verdict.ERROR)
        if len(skipped) > 0:
            result.add(Verdict.SKIP)
        return result

    @staticmethod
    def decide_overall_verdict(verdicts: Set[str]) -> str:
        """
        Decide the overall verdict of a test given the verdicts of multiple runs.
        :param verdicts: The verdicts of multiple runs of the same test
        :return: The overall verdict:
            "Passed" / "Failed" / "Error" / "Skipped" / "Flaky" / "ZeroRuns"
            "Undecidable" should never happen, this means the algorithm has a problem
        """
        verdicts = {
            x
            for x in verdicts
            if (not pd.isna(x)) and x != Verdict.ZERO_RUNS and x != Verdict.PARSE_ERROR
        }

        if len(verdicts) == 0:
            return Verdict.ZERO_RUNS
        if Verdict.UNDECIDABLE in verdicts:
            return Verdict.UNDECIDABLE
        if verdicts - {Verdict.SKIP} == {Verdict.PASS}:
            return Verdict.PASS
        if verdicts - {Verdict.ERROR, Verdict.SKIP} == {Verdict.FAIL}:
            return Verdict.FAIL
        if (
            (Verdict.PASS in verdicts and (Verdict.FAIL in verdicts or Verdict.ERROR in verdicts))
        ) or (Verdict.FLAKY in verdicts):
            return Verdict.FLAKY
        if verdicts == {Verdict.ERROR}:
            return Verdict.ERROR
        if verdicts == {Verdict.ERROR, Verdict.SKIP}:
            return Verdict.ERROR
        if verdicts == {Verdict.SKIP}:
            return Verdict.SKIP
        return Verdict.UNDECIDABLE


def to_nodeid(filename: str, classname: str, funcname: str, parametrization: str = "") -> str:
    """
    Reconstruct nodeID from variables given by pytest's junit-xml.
    Pytest computes classname and funcname from the nodeID.
    Reference: https://github.com/pytest-dev/pytest/blob/de6c28ed1f26f3ffa937472de2967e03c1da044a/src/_pytest/junitxml.py#L438  # pylint: disable=line-too-long
    """
    if classname == "":
        return f"{filename}::{funcname}{parametrization}"
    split = classname.split(".")
    try:
        # Case there exists a test-class (assume class names are upper case)
        if split[-1][0].isupper():
            *file, class_ = split
            file[-1] = file[-1] + ".py"
            return f"{os.path.join(*file)}::{class_}::{funcname}{parametrization}"
        # Case there is no test-class, just a test-file
        else:
            file = split
            file[-1] = file[-1] + ".py"
            return f"{os.path.join(*file)}::{funcname}{parametrization}"
    except IndexError:
        logging.error(
            f"classname_to_nodeid: IndexError with classname={classname}, funcname={funcname}"
        )
        return funcname


#                            Prefix     Func      Tags         Hash
# EXAMPLE                    <-------- (mod, foo) #wrapper-call <= 421337
parse_pattern = re.compile(r"(<-+|-+>) (\([^)]*\))(.*?)((<=|=>) (.*))?$")


def parse_trace_line(
    line: str, additional_error_information=""
) -> Tuple[str, int, FuncDescriptor, str, str]:
    """
    Parse a line of a tracefile"
    :param additional_error_information:
    :param line: EXAMPLE:
        --> ('domain_validation.test.test_whois_obj', None, 'test_WHOIS_utf_encoding') <= 306f
    :return: EXAMPLE:
        (
            'call',
            1
            "('domain_validation.test.test_whois_obj', None, 'test_WHOIS_utf_encoding')",
            '',
            '306f'
        )
    """
    if isinstance(line, bytes):
        line = line.decode()

    match = parse_pattern.findall(line)
    if len(match) == 0:
        raise ValueError(f'Cannot parse line "{line}" {additional_error_information}')

    prefix, func, tags, _, _, arg_hash = match[0]

    if prefix.startswith("-"):
        call_return = "call"
    elif prefix.startswith("<"):
        call_return = "return"
    else:
        raise ValueError(f'Unrecognized start sequence "{prefix}" {additional_error_information}')
    depth = int(prefix.count("-") / 2)
    func = literal_eval(func)
    func = tuple(["" if x is None else x.replace(" ", "_") for x in func])
    tags = tags.strip()

    return call_return, depth, func, tags, arg_hash


def calculate_suspiciousness_scores_accum(
    cta: pd.DataFrame, tne: pd.Series, test_flakiness_status: pd.Series, sfl_method: str
) -> pd.DataFrame:
    """
    :cta: covered table accumulated
    :tne: tests number executions
        pd.Series mapping a test to the number of times it has been executed
    :test_flakiness_status:
        pd.Series mapping each test to its flakiness status (true / false)
    :sfl_method: how to treat different coverage behaviors
        sffl       -> intersection for flaky tests, union for stable tests
        union      -> union for flaky tests, union for stable tests
        individual -> treat each test execution as a separate test case
    """

    # The following operations cause performance warning, however, avoid these causes longer
    # runtimes -> we suppress them
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    score_df = pd.DataFrame()

    flaky_tests = test_flakiness_status[test_flakiness_status].keys()
    nonFlaky_tests = test_flakiness_status[~test_flakiness_status].keys()

    flaky_total = 0
    nonFlaky_total = 0

    if sfl_method == "sffl":
        score_df[flaky_tests] = cta[flaky_tests].apply(lambda s: s == tne[s.name])
        score_df[nonFlaky_tests] = cta[nonFlaky_tests] > 0
        flaky_total = len(flaky_tests)
        nonFlaky_total = len(nonFlaky_tests)
    elif sfl_method == "union":
        score_df[flaky_tests] = cta[flaky_tests] > 0
        score_df[nonFlaky_tests] = cta[nonFlaky_tests] > 0
        flaky_total = len(flaky_tests)
        nonFlaky_total = len(nonFlaky_tests)
    elif sfl_method == "individual":
        score_df[flaky_tests] = cta[flaky_tests]
        score_df[nonFlaky_tests] = cta[nonFlaky_tests]
        flaky_total = sum(tne[flaky_tests])
        nonFlaky_total = sum(tne[nonFlaky_tests])
    else:
        raise ValueError(f"Unknown sfl_method {sfl_method}, use 'sffl', 'union', or 'individual'")

    # Per line: how my (non)flaky Tests cover it?
    score_df["flaky_covered"] = score_df[flaky_tests].apply(sum, axis="columns")
    score_df["nonFlaky_covered"] = score_df[nonFlaky_tests].apply(sum, axis="columns")

    score_df["flaky_total"] = flaky_total
    score_df["nonFlaky_total"] = nonFlaky_total

    # Calculate suspiciousness scores
    if not score_df.empty:  # the following steps crash on empty dataframes
        score_df["tarantula"] = score_df.apply(
            lambda s: tarantula(
                s["flaky_covered"], s["nonFlaky_covered"], flaky_total, nonFlaky_total
            ),
            axis="columns",
        )
        score_df["ochiai"] = score_df.apply(
            lambda s: ochiai(s["flaky_covered"], s["nonFlaky_covered"], flaky_total),
            axis="columns",
        )
        score_df["dStar"] = score_df.apply(
            lambda s: dStar(s["flaky_covered"], s["nonFlaky_covered"], flaky_total, 2),
            axis="columns",
        )
        score_df["barinel"] = score_df.apply(
            lambda s: barinel(s["flaky_covered"], s["nonFlaky_covered"], flaky_total),
            axis="columns",
        )
        score_df["op2"] = score_df.apply(
            lambda s: op2(s["flaky_covered"], s["nonFlaky_covered"], nonFlaky_total),
            axis="columns",
        )
    else:
        score_df["tarantula"] = []
        score_df["ochiai"] = []
        score_df["dStar"] = []
        score_df["barinel"] = []
        score_df["op2"] = []

    # Encode flakiness status in column name
    score_df.rename(
        columns=lambda x: ("flaky" if test_flakiness_status[x] else "non_flaky", x)
        if x in test_flakiness_status
        else x,
        inplace=True,
    )

    return score_df


class SuspiciousnessTable(object):

    """Output of save_susiciousness_tables_for_project"""

    line_columns = ["path", "lines", "flaky_covered", "nonFlaky_covered", "flaky_total", "nonFlaky_total"]
    sfl_columns = ["tarantula", "ochiai", "dStar", "barinel", "op2"]

    def __init__(self, df: pd.DataFrame, url=None, git_hash=None):
        self._df = df
        self.url = url
        self.git_hash = git_hash

        non_test_columns = self.line_columns + self.sfl_columns + ["Unnamed: 0"]

        # Test columns are tuples `(flaky/non_flaky, TEST_NODEID)` -> evaluate from string
        self.test_columns = set(self._df.columns).difference(non_test_columns)
        self._df.columns = self._df.columns.map(
            lambda c: literal_eval(c) if c in self.test_columns else c
        )
        self.test_columns = set(self._df.columns).difference(non_test_columns)
        self.flaky_tests = [t for fs, t in self.test_columns if fs == "flaky"]
        self.non_flaky_tests = [t for fs, t in self.test_columns if fs == "non_flaky"]

        # path has prefix "workdir/..." -> remove that
        if "path" in self._df.columns:
            self._df["path"] = self._df["path"].apply(
                lambda x: re.sub(
                    "/workdir/(non-deterministic|deterministic|sameOrder|randomOrder)/tmp/(tmp.*?|flapy_repo_copy)/",
                    "",
                    x,
                )
            )

        # calc ranks
        for col in self.sfl_columns:
            self._df[f"{col}_bc_rank"] = self._df[col].rank(method="min", ascending=False)
            self._df[f"{col}_wc_rank"] = self._df[col].rank(method="max", ascending=False)

        # set index
        if not self._df.empty:
            self._df.set_index(["path", "lines"], inplace=True)

    @classmethod
    def load(cls, path_: str):
        """Load SuspiciousnessTable from CSV file"""
        path_ = Path(path_)
        _df = pd.read_csv(path_)
        url, git_hash = re.match("(.*)@(.*).csv", path_.name).groups()
        url = url.replace(" ", "/")
        obj = cls(_df, url, git_hash)
        obj.p = path_
        return obj


class SuspiciousnessDir(object):

    """Output of IterationCollectionL.save_suspiciousness_tables"""

    def __init__(self, path):
        self.p = Path(path)
        assert self.p.is_dir()
        self.suspiciousness_tables = [SuspiciousnessTable.load(path) for path in self.p.glob("*")]
        self.proj_to_st = {(st.url, st.git_hash): st for st in self.suspiciousness_tables}

    def statistic(self):
        statistic_table = pd.DataFrame(
            [
                (
                    st.url,
                    st.git_hash,
                    st._df.empty,
                    len(st.flaky_tests),
                    len(st.non_flaky_tests),
                    # min(st._df["tarantula"]) if not st._df.empty else None,
                    # max(st._df["tarantula"]) if not st._df.empty else None,
                )
                for st in self.suspiciousness_tables
            ],
            columns=[
                "Project_URL",
                "Project_Hash",
                "empty",
                "num_flaky_tests",
                "num_non_flaky_tests",
                # "tarantula_min", "tarantual_max"
            ],
        )
        return statistic_table

    def get_suspiciousness_rank_of_source_code_line(
        self,
        proj_url: str,
        proj_hash: str,
        test_nodeid: str,
        location_file: str,
        location_line,
        drop_not_covered_lines=True
    ) -> Tuple[pd.Series, pd.Series, str]:
        """
        Search for suspiciousness rank of a given source code line
        in a given file of a given project.

        The numbers in the error messages are arbitrary, only their ordering is important for
        `min` in the groupby in `merge_location_info`.

        :drop_not_covered_lines: some lines were not covered by any test case, but just the general
            program execution. Drop these from the suspiciousness table?

        :return: (num_tests, ranks)

            num_tests: how many flaky and non-flaky tests were executed, e.g.

                num_flaky_tests         1
                num_non_flaky_tests    11

            ranks: ranks given to the specivied lines by different approaches, e.g.

                flaky_covered          0
                nonFlaky_covered       0
                tarantula              0
                ochiai                 0
                dStar                  0
                tarantula_bc_rank     11
                tarantula_wc_rank    606
                ochiai_bc_rank        11
                ochiai_wc_rank       606
                dStar_bc_rank         11
                dStar_wc_rank        606

        """

        # Search, if we have a suspiciousness table for the specified project
        st: SuspiciousnessTable = self.proj_to_st.get((proj_url, proj_hash))

        if st is None:
            raise ValueError("8. NO SuspiciousnessTable FOUND")

        st_df = st._df.copy()
        if st_df.empty:
            raise ValueError("6. EMPTY SuspiciousnessTable")
        else:
            if drop_not_covered_lines:
                st_df = st_df[(st_df["flaky_covered"] > 0) | (st_df["nonFlaky_covered"] > 0)]

            # Search for the specified source code file and line in the suspiciousness table
            if (location_file, location_line) not in st_df.index:
                if test_nodeid not in st.flaky_tests:
                    if test_nodeid not in st.non_flaky_tests:
                        error_message = "42. FILE:LINE NOT FOUND + test was not executed"
                    else:
                        error_message = "41. FILE:LINE NOT FOUND + test did not show flaky behavior"
                error_message = "44. FILE:LINE NOT FOUND"
                raise ValueError(error_message)
            else:

                if test_nodeid not in st.flaky_tests:
                    if test_nodeid not in st.non_flaky_tests:
                        status = "2. test was not executed"
                    else:
                        status = "1. test did not show flaky behavior"
                else:
                    status = "0. successfully matched and test showed flaky behavior"

                num_tests = pd.Series({
                    "num_flaky_tests": len(st.flaky_tests),
                    "num_non_flaky_tests": len(st.non_flaky_tests),
                })

                # calc ranks
                for col in SuspiciousnessTable.sfl_columns:
                    st_df[f"{col}_bc_rank"] = st_df[col].rank(method="min", ascending=False)
                    st_df[f"{col}_wc_rank"] = st_df[col].rank(method="max", ascending=False)

                ranks = st_df.loc[(location_file, location_line)][
                    st_df.columns.drop(st.test_columns)
                ]
                return num_tests, ranks, status

    def merge_location_info(
        self, location_file_name: str, loc_file_name: str, *, drop_not_covered_lines=True
    ):
        """

        location_file_name: path to CSV file containing the true fault locations.

        loc_file_name: path to CSV file containing lines-of-code information for the projects.
            This information is necessary to calculate the EXAM score. We cannot just use the
            number of all covered lines, because the test suite might not cover the entire source
            code.
        """
        # Load locations file and explode its location column
        lt = LocationTable.load(location_file_name)
        lt_splitted = lt.split_locations()

        # For each location, determine its rank according to different approaches
        result = []
        for _, row in lt_splitted.iterrows():
            try:
                num_tests, ranks, status = self.get_suspiciousness_rank_of_source_code_line(
                    proj_url=row["Project_URL"],
                    proj_hash=row["Project_Hash"],
                    test_nodeid=row["Test_nodeid_inclPara"],
                    location_file=row["Location_file"],
                    location_line=row["Location_line"],
                    drop_not_covered_lines=drop_not_covered_lines,
                )
                status_info = pd.Series({"status": status})
                result.append(pd.concat([row, num_tests, ranks, status_info]))
            except ValueError as e:
                status_info = pd.Series({"status": str(e)})
                result.append(pd.concat([row, status_info]))
        result_df = pd.DataFrame(result)

        # Group by test
        #   one project/test can have multiple locations -> take the earliest one found
        result_df = (
            result_df.groupby(proj_cols + test_cols_without_filename + ["Test_nodeid_inclPara"])
            .agg(
                {
                    "num_flaky_tests": "first",
                    "num_non_flaky_tests": "first",
                    "status": "min",
                    **{
                        f"{col}_{case}_rank": "min"
                        for col in SuspiciousnessTable.sfl_columns
                        for case in ["bc", "wc"]
                    },
                }
            )
            .reset_index()
        )

        result_df = result_df.merge(lt._df)

        # Compare to total lines of code
        loc = pd.read_csv(loc_file_name)
        loc["Project_Hash"] = loc["Project_Hash"].fillna("")
        loc.rename(columns={"code": "LOC"}, inplace=True)
        loc = loc[loc["language"] == "Python"][["Project_URL", "Project_Hash", "LOC"]]
        result_df = result_df.merge(loc, how="left")

        # Calcualte EXAM scores
        for col in SuspiciousnessTable.sfl_columns:
            result_df[f"EXAM_{col}_bc"] = result_df[f"{col}_bc_rank"] / result_df["LOC"]
            result_df[f"EXAM_{col}_wc"] = result_df[f"{col}_wc_rank"] / result_df["LOC"]

        return result_df


class CtaDir(object):

    """ Output of ResultsDir(Collection).save_cta_tables """

    def __init__(self, path):
        """ """
        self.p = Path(path)
        assert self.p.is_dir()
        self.ctas = [
            try_default(
                lambda: CoverageTableAccumulated.load(path),
                log_error_info=f"path=({path})"
            )
            for path in self.p.glob("*_cta.csv")
        ]
        self.ctas = [x for x in self.ctas if x is not None]

    def calc_and_save_suspiciousness_tables(self, save_dir: str, sfl_method: str):
        """

        :save_dir: path to the directory where the resulting suspiciousness tables shall be saved

        :sfl_method: how to treat different coverage behaviors
            sffl       -> intersection for flaky tests, union for stable tests
            union      -> union for flaky tests, union for stable tests
            individual -> treat each test execution as a separate test case

        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        for cta in tqdm(self.ctas):
            try:
                scores = cta.calc_suspiciousness_scores(sfl_method)
                url_without_slash = cta.url.replace("/", " ")
                scores.to_csv(save_dir / f"{url_without_slash}@{cta.git_hash}.csv")
            except Exception as e:
                logging.error(f"{type(e).__name__}: {e} | cta_file=({cta.p})")
                logging.error(traceback.format_exc())

    def coverage_statistic(self):
        result = []
        for cta in tqdm(self.ctas):
            df = cta.coverage_statistic()
            df["Project_URL"] = cta.url
            df["Project_Hash"] = cta.git_hash
            result.append(df)
        return pd.concat(result)


class CoverageTableAccumulated(object):

    """ """

    def __init__(self, df: pd.DataFrame, tnefs_df: pd.DataFrame, url=None, git_hash=None):
        """ """
        self._df = df
        self.url = url
        self.git_hash = git_hash
        self.tnefs_df = tnefs_df

    @classmethod
    def load(cls, path_: str):
        """ """
        path_ = Path(path_)
        url, git_hash = re.match("(.*)@(.*)_cta.csv", path_.name).groups()
        url = url.replace(" ", "/")
        # load related file containing total-number-of-executions (tne) and flakiness status (fs)
        tnefs_path = Path(str(path_)[:-8] + "_TneFs.csv")
        tnefs_df = pd.read_csv(tnefs_path)
        tnefs_df = tnefs_df.set_index("Test_nodeid_inclPara")

        _df = pd.read_csv(path_)
        _df = _df.set_index(["path", "lines"])

        obj = cls(_df, tnefs_df, url, git_hash)
        obj.p = path_
        return obj

    def calc_suspiciousness_scores(self, sfl_method: str) -> pd.DataFrame:
        """For each statement (= row in this CTA), calculate its suspiciousness

        :sfl_method:
            sffl       -> intersection for flaky tests, union for stable tests
            union      -> union for flaky tests, union for stable tests
            individual -> treat each test execution as a separate test case
        """

        # Get flakiness status column / series
        #   Assume that the TNE has exactly two columns:
        #   "execution_count" and the flakiness status column
        assert len(self.tnefs_df.columns) == 2
        assert "execution_count" in self.tnefs_df.columns
        flaky_col = self.tnefs_df.columns.drop("execution_count")[0]
        flakiness_status = self.tnefs_df[flaky_col].dropna().astype(bool)

        # Log warnings (this was taken calc_suspiciousness_for_proj, probably there's a more elegant
        # way, maybe just self.tnefs_df.dropna() )
        executed_tests_with_cov = list(self.tnefs_df["execution_count"].dropna().index)
        if len(executed_tests_with_cov) == 0:
            logging.warning(f"Project {self.url} has no coverage data")
        else:
            no_coverage_data = set(flakiness_status.index).difference(executed_tests_with_cov)
            no_flakiness_status = set(executed_tests_with_cov).difference(
                set(flakiness_status.index)
            )
            if len(no_coverage_data) > 0:
                logging.warning(
                    f"No coverage data found for project {self.url} in tests {no_coverage_data}"
                )
            if len(no_flakiness_status) > 0:
                logging.warning(
                    f"No flakiness status found for project {self.url} "
                    f"in tests {no_flakiness_status}"
                )

        flakiness_status_and_coverage_available = sorted(
            set(flakiness_status.index).intersection(executed_tests_with_cov)
        )
        scores = calculate_suspiciousness_scores_accum(
            self._df,
            self.tnefs_df["execution_count"],
            flakiness_status[flakiness_status_and_coverage_available],
            sfl_method
        )
        return scores

    def coverage_statistic(self) -> pd.DataFrame:
        """How many lines were covered by not all executions?"""
        result = []
        for test in self._df.columns:
            tne = self.tnefs_df["execution_count"][test]
            cov = self._df[test]
            cov = cov[cov > 0]
            cov = cov.apply(lambda x: "always" if x == tne else "sometimes" if x < tne else "ERROR")
            cov = cov.value_counts()
            result.append(cov)
        cov_df = pd.DataFrame(result)
        cov_df.index.name = "Test_nodeid_inclPara"
        cov_df = cov_df.reset_index()
        cov_df = cov_df.merge(self.tnefs_df.reset_index())
        return cov_df


class LocationTable(object):

    """
    Mapping a flaky tests to the lines in the source code causing the flakiness
    (manually labeled)

    Columns:
        Project_(Name,URL,Hash),
        Test_(filename,classname,funcname,parametrization),
        Location

    Location = file_name:line_range;file_name2:line_range2;...

    line_range is for example `1-3,5,7`

    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._df["Project_Hash"] = self._df["Project_Hash"].fillna("")
        self._df["Test_filename"] = self._df["Test_filename"].fillna("")
        self._df["Test_parametrization"] = self._df["Test_parametrization"].fillna("")

        # compute test nodeid
        self._df["Test_nodeid_inclPara"] = self._df[test_cols].apply(
            lambda s: to_nodeid(*s), axis=1
        )

    def split_locations(self) -> pd.DataFrame:
        """Explode dataframe. Example: Transform foo.py:1,2,3 into three separate rows
        :returns: new dataframe

        """
        _df = self._df.copy()
        # split multiple different files
        _df["Location_split"] = _df["Location"].apply(lambda x: x.split(";"))
        _df = _df.explode("Location_split")

        # split multiple lines in the same files
        _df[["Location_file", "Location_line"]] = _df["Location_split"].str.split(
            ":", expand=True
        )
        _df["Location_line"] = _df["Location_line"].apply(parse_string_range)
        _df = _df.explode("Location_line")
        return _df

    @classmethod
    def load(cls, file_name: str):
        """Load LocationTable from CSV file"""
        path = Path(file_name)
        _df = pd.read_csv(path)
        return cls(_df)


def main() -> None:
    """The main entry location of the program."""
    # TODO improve startup time by making fire not explore libraries
    fire.Fire()


if __name__ == "__main__":
    main()
