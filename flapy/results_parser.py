#!/usr/bin/env python3
import re
import os
import sys
import tarfile
import tempfile
import sqlite3

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

from flapy.utils import try_default, eprint

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
FORMAT = "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)4s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

proj_cols = ["Project_Name", "Project_URL", "Project_Hash"]
test_cols = ["Test_filename", "Test_classname", "Test_funcname", "Test_parametrization"]
test_cols_without_parametrization = ["Test_filename", "Test_classname", "Test_funcname"]
test_cols_without_filename = ["Test_classname", "Test_funcname", "Test_parametrization"]


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
        "result-text": (
            re.findall(r"(\w*Error.*)\n", test_case.result._elem.text)
            if test_case.result and test_case.result._elem
            else []
        ),
        "system-err": (
            re.findall(r"(\w*Error.*)\n", test_case.system_err) if test_case.system_err else []
        ),
    }


def is_empty(openvia: Callable[[str], IO], path: str):
    try:
        with openvia(path) as f:
            next(f)
    except StopIteration:
        return True
    return False


def junitxml_classname_to_modname_and_actual_classname(classname: str) -> Tuple[List[str], str]:
    """ The JUnit-XML attribute 'class' contains both the name of the module and the name of the class -> split them by assuming class names start with capital letters.

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
        print(f"junitxml_classname_to_actual_classname: IndexError with classname={classname}")
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
    def __init__(self, df: pd.DataFrame):
        self._df = df

        # Some junit-xml files actually had name="" in them
        #   -> replace by NaN so they get ignored in the groupby
        self._df["Test_funcname"] = self._df["Test_funcname"].replace("", np.NaN)

        # Rows with NaN are ignored by pd.groupby -> fillna
        self._df["Test_filename"] = self._df["Test_filename"].fillna("")
        self._df["Test_classname"] = self._df["Test_classname"].fillna("")
        self._df["Test_parametrization"] = self._df["Test_parametrization"].fillna("")

    @classmethod
    def load(cls, file_name: str):
        """Load PassedFailed from CSV file

        :file_name: TODO
        :returns: TODO

        """
        _df = pd.read_csv(
            file_name,
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
        self._df["ids_deter"] = [
            p.union(f).union(e).union(s)
            for p, f, e, s in zip(
                self._df["Passed_sameOrder"],
                self._df["Failed_sameOrder"],
                self._df["Error_sameOrder"],
                self._df["Skipped_sameOrder"],
            )
        ]
        return self

    def to_test_overview(self) -> pd.DataFrame:
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
                "Test_funcname",
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
                "#Runs_sameOrder": sum,
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
                "#Runs_randomOrder": sum,
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

        def decide_flakiness_type(
            flaky_sameOrder_withinIteration: bool, order_dependent: bool, infrastructure: bool,
        ) -> str:
            assert flaky_sameOrder_withinIteration + order_dependent + infrastructure < 2
            if flaky_sameOrder_withinIteration:
                return FlakinessType.NOD
            if order_dependent:
                return FlakinessType.OD
            if infrastructure:
                return FlakinessType.INFR
            return FlakinessType.NOT_FLAKY

        test_overview.insert(
            7,
            "flaky?",
            test_overview.apply(
                lambda s: decide_flakiness_type(
                    s["Flaky_sameOrder_withinIteration"],
                    s["Order-dependent"],
                    s["Flaky_Infrastructure"],
                ),
                axis="columns",
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
            lambda s: to_nodeid(s["Test_filename"], s["Test_classname"], s["Test_funcname"]),
            axis=1,
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
                "Test_funcname",
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


class MyFileWrapper(ABC):
    def __init__(
        self,
        path_: Union[str, Path],
        tarinfo: tarfile.TarInfo,
        project_name: str,
        openvia: Callable[[str], IO] = open,
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
        return re.match(cls.get_regex(project_name), str(path)) is not None and not is_empty(
            openvia, str(path)
        )

    # @classmethod
    # def is_(cls, path: Path, project_name: str, openvia: Callable[[str], IO]) -> bool:
    #     return re.match(cls.get_regex(project_name), str(path)) is not None

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.p}')"


class CoverageXmlFile(MyFileWrapper):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/{project_name}_coverage(\d+)(.*)\.xml$"

    def get_order(self) -> str:
        if CoverageXmlFileDeter.is_(self.p, self.project_name, self.openvia):
            return "deter"
        if CoverageXmlFileNonDeter.is_(self.p, self.project_name, self.openvia):
            return "non-deter"
        return "COULD_NOT_GET_ORDER"

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        """
        Transform Junit XML files into a table that shows the verdict and message for each run.
        :return:
            EXAMPLE:
                {
                    'num': 0,
                    'order': "deter",
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
        except Exception:
            # traceback.print_exc()
            # eprint(f"{type(ex)}: ex")
            # eprint(f"    in {self.p}")
            return {}


class CoverageXmlFileDeter(CoverageXmlFile):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/deterministic/tmp/{project_name}_coverage(\d+)(.*)\.xml"


class CoverageXmlFileNonDeter(CoverageXmlFile):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/non-deterministic/tmp/{project_name}_coverage(\d+)(.*)\.xml"


class JunitXmlFile(MyFileWrapper):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/tmp/{project_name}_output(\d+)(.*)\.xml$"

    @lru_cache()
    def get_order(self) -> str:
        if JunitXmlFileDeter.is_(self.p, self.project_name, self.openvia):
            return "deter"
        if JunitXmlFileNonDeter.is_(self.p, self.project_name, self.openvia):
            return "non-deter"
        return "COULD_NOT_GET_ORDER"

    def parse(self) -> junitparser.JUnitXml:
        with self.open() as f:
            try:
                return junitparser.JUnitXml.fromfile(f, ET.parse)
            except xml.etree.ElementTree.ParseError:
                return junitparser.JUnitXml()

    def get_hostname(self) -> str:
        return list(self.parse())[0].hostname

    def get_testcases(self) -> List[junitparser.TestCase]:
        junit_xml = self.parse()
        return [case for suite in junit_xml for case in suite]

    def to_table(self) -> List[Dict[str, Union[str, int]]]:
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
            junit_xml = self.parse()
            test_cases = [case for suite in junit_xml for case in suite]
            # if len(test_cases) == 0:
            #     logging.warning(f"{self.p} contains no testcases")
            result: List[Dict[str, Union[str, int]]] = [
                {
                    **read_junit_testcase(test_case),
                    "num": self.get_num(),
                    "test_to_be_run": self.get_test_to_be_run(),
                    "order": self.get_order(),
                }
                for test_case in test_cases
            ]
            return result
        except Exception as ex:
            # traceback.print_exc()
            eprint(f"{type(ex)} in {self.p}:")
            eprint(f"    {ex}")
            return []


class JunitXmlFileDeter(JunitXmlFile):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/deterministic/tmp/{project_name}_output(\d+)(.*)\.xml"


class JunitXmlFileNonDeter(JunitXmlFile):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/non-deterministic/tmp/{project_name}_output(\d+)(.*)\.xml"


class TraceFile(MyFileWrapper):
    """File containing traces """

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


class IterationResultsMetaInfo:
    def __init__(self, filename):
        with open(filename) as f:
            self._datamap = yaml.safe_load(f)
        self.project_name = self._datamap["project_name"]
        self.project_url = self._datamap["project_url"]
        self.project_git_hash = self._datamap["project_git_hash"]
        self.func_to_trace = self._datamap["func_to_trace"]
        self.tests_to_be_run = self._datamap["tests_to_be_run"]
        self.num_runs = self._datamap["num_runs"]
        self.start_time = self._datamap["start_time"]
        self.end_time = self._datamap["end_time"]
        self.execution_time = self._datamap["execution_time"]


class Iteration:
    """
    Example: flapy-results_20200101/lineflow
    """

    def __init__(self, dir: Union[str, Path]):
        self.p = Path(dir)
        assert Iteration.is_iteration(self.p), "This does not seem like an iteration directory"
        # TODO remove extracted, we don't do that anymore, we use cache instead
        self.use_extracted = (self.p / "local").is_dir()
        self._archive: Optional[tarfile.TarFile] = None

        # Setup cache
        self._results_cache = self.p / "results_cache"
        if not self._results_cache.is_dir():
            self._results_cache.mkdir()
        self._junit_cache_file = self._results_cache / "junit_data.csv"

        # Read meta info if available (only in newer versions, older versions use separate files)
        self.meta_file = self.p / "flapy-iteration-result.yaml"
        if self.meta_file.is_file():
            self.meta_info = IterationResultsMetaInfo(self.meta_file)
        else:
            self.meta_info = None

    @classmethod
    def is_iteration(cls, path: Path) -> bool:
        return (
            path.is_dir()
            and path.name != "run"
            and not path.name.startswith(".")
            and ((path / "local").is_dir() or (path / "results.tar.xz").is_file())
        )

    @lru_cache()
    def get_project_name(self) -> str:
        if self.meta_info is not None:
            return self.meta_info.project_name
        if (self.p / "project-name.txt").is_file():
            with open(self.p / "project-name.txt") as file:
                return file.read().replace("\n", "")
        if self.use_extracted:
            user_dir = next((self.p / "local" / "hdd").iterdir())
            return next(user_dir.iterdir()).name
        else:
            if len(self.get_archive_names()) == 0:
                return self.p.name + "_EMPTY"
            else:
                # local/hdd/user/project_name
                try:
                    return self.get_archive_names()[0].split("/")[3]
                except Exception:
                    logging.error(
                        f"Could not get project name for iteration results {self.p}. This is needed for identifying junit-xml, coverage, and other files, as their name contains the project name."
                    )
                    return "COULD_NOT_GET_PROJECT_NAME"

    def get_project_url(self) -> str:
        if self.meta_info is not None:
            return self.meta_info.project_url
        if (self.p / "project-url.txt").is_file():
            with open(self.p / "project-url.txt") as file:
                return file.read().replace("\n", "")
        return "COULD_NOT_GET_PROJECT_URL"

    def get_project_git_hash(self) -> str:
        if self.meta_info is not None:
            return self.meta_info.project_git_hash
        if (self.p / "project-git-hash.txt").is_file():
            with open(self.p / "project-git-hash.txt") as file:
                return file.read().replace("\n", "")
        return "COULD_NOT_GET_PROJECT_GIT_HASH"

    def get_flapy_git_hash(self) -> str:
        """Flapy used to be called 'flakyanalysis'"""
        if (self.p / "flakyanalysis-git-hash.txt").is_file():
            with open(self.p / "flakyanalysis-git-hash.txt") as file:
                return file.read().replace("\n", "")
        return "COULD_NOT_GET_FLAKYANALYSIS_GIT_HASH"

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
            self.close_archive()
        if write_cache and not did_read_cache:
            junit_data.to_csv(self._junit_cache_file, index=False)
        if include_project_columns:
            junit_data.insert(0, "Project_Hash", self.get_project_git_hash())
            junit_data.insert(0, "Project_URL", self.get_project_url())
            junit_data.insert(0, "Project_Name", self.get_project_name())
            junit_data.insert(0, "Iteration", self.p.name)
        if return_nothing:
            return None
        return junit_data.fillna("")

    def get_passed_failed(self, *, read_cache=True, write_cache=True) -> pd.DataFrame:
        junit_data = self.get_junit_data(
            include_project_columns=True, read_cache=read_cache, write_cache=write_cache
        )
        if len(junit_data) == 0:
            return pd.DataFrame(
                {
                    "Iteration": [self.p.name],
                    "Iteration_EMPTY": [True],
                    "Project_Name": [self.get_project_name()],
                    "Project_URL": [self.get_project_url()],
                    "Project_Hash": [self.get_project_git_hash()],
                }
            )
        junit_data["Test_filename"] = junit_data["file"]
        junit_data["Test_classname"] = junit_data["class"]
        junit_data["Test_funcname"] = [re.sub(r"\[.*\]", "", name) for name in junit_data["name"]]
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
                "Test_funcname",
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

        passed_failed["#Runs"] = [
            len(p.union(f).union(e).union(s))
            for p, f, e, s in zip(
                passed_failed["Passed"],
                passed_failed["Failed"],
                passed_failed["Error"],
                passed_failed["Skipped"],
            )
        ]

        passed_failed["Verdict"] = passed_failed["Verdicts"].apply(Verdict.decide_overall_verdict)

        passed_failed = pd.merge(
            passed_failed[passed_failed["order"] == "deter"],
            passed_failed[passed_failed["order"] == "non-deter"],
            on=[
                "Iteration",
                "Project_Name",
                "Project_URL",
                "Project_Hash",
                "Test_filename",
                "Test_classname",
                "Test_funcname",
                "Test_parametrization",
            ],
            how="outer",
            suffixes=("_sameOrder", "_randomOrder"),
        )

        passed_failed.insert(1, "Iteration_EMPTY", False)

        return passed_failed[
            [
                "Iteration",
                "Iteration_EMPTY",
                #
                "Project_Name",
                "Project_URL",
                "Project_Hash",
                "Test_filename",
                "Test_classname",
                "Test_funcname",
                "Test_parametrization",
                #
                "Passed_sameOrder",
                "Failed_sameOrder",
                "Error_sameOrder",
                "Skipped_sameOrder",
                "Verdict_sameOrder",
                "Verdicts_sameOrder",
                "#Runs_sameOrder",
                #
                "Passed_randomOrder",
                "Failed_randomOrder",
                "Error_randomOrder",
                "Skipped_randomOrder",
                "Verdict_randomOrder",
                "Verdicts_randomOrder",
                "#Runs_randomOrder",
            ]
        ]

    def get_coverage_raw_data(self) -> List[Dict]:
        """Read branch- and line-coverage from coverage-xml

        :returns: EXAMPLE:
            [
                { 'num': 0, 'order': "deter", "BranchCoverage": 0.7, "LineCoverage": 0.6 },
                ...
            ]
        """
        coverage_data = list(
            filter(
                lambda d: len(d) > 0, map(CoverageXmlFile.to_dict, self.get_files(CoverageXmlFile)),
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

    def get_files(self, type_: Type[T1]) -> List[T1]:
        if self.use_extracted:
            return [
                type_(path, self.get_project_name(), openvia=open)
                for path in self.p.rglob("*")
                if type_.is_(path, self.get_project_name(), openvia=open)
            ]
        else:
            return [
                type_(
                    tarinfo.name,
                    tarinfo,
                    self.get_project_name(),
                    openvia=self.get_archive().extractfile,  # type: ignore
                    archive=self.get_archive(),
                )
                for tarinfo in self.get_archive_members()
                if type_.is_(
                    Path(tarinfo.name),
                    self.get_project_name(),
                    openvia=self.get_archive().extractfile,  # type: ignore
                )
            ]

    def get_junit_files(self) -> List[JunitXmlFile]:
        return self.get_files(JunitXmlFile)

    def get_trace_files(self) -> List[TraceFile]:
        return self.get_files(TraceFile)

    def get_archive(self) -> tarfile.TarFile:
        if self._archive is None:
            self._archive = tarfile.open(self.p / "results.tar.xz")
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
            "Test_funcname": test_funcdescr[2],
            **search,
        }

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
            pd.DataFrame(
                [
                    (it, it.get_project_name(), it.get_project_url(), it.get_project_git_hash(),)
                    for it in self.get_iterations()
                ],
                columns=["Iteration", "Project_Name", "Project_URL", "Project_Hash"],
            )
            .set_index(["Project_Name", "Project_URL", "Project_Hash"])
            .sort_index()
        )
        return iterations_overview

    @abstractmethod
    def get_passed_failed(self) -> pd.DataFrame:
        pass

    def get_tests_overview(
        self,
        *,
        read_resultsDir_cache=True,
        write_resultsDir_cache=True,
        read_iteration_cache=True,
        write_iteration_cache=True,
    ) -> TestsOverview:
        # Use cache if possible
        #   I do not use @lru_cache here, because it doesn't work inside pool.map,
        #   since every process hast it's own memory
        if self.tests_overview is None:
            pf = self.get_passed_failed(
                read_iteration_cache=read_iteration_cache,
                write_iteration_cache=write_iteration_cache,
            )
            to = PassedFailed(pf).to_test_overview()
            self.tests_overview = TestsOverview(to)
        return self.tests_overview


class ResultsDir(IterationCollection):
    """
    Directory created by one execution of flapy.
    Example: flapy-results_20200101_1430
    """

    def __init__(self, dir: Union[str, Path]):
        self.p = Path(dir)
        assert self.p.is_dir(), f"Directory {self.p} does not exist"

        # Setup cache
        #   we use the exclamation mark, because it has a early position in the ascii table, so
        #   the cache file appears at the top of the directory
        self._results_cache = self.p / "!flapy.cache"
        if not self._results_cache.is_dir():
            self._results_cache.mkdir()
        self._pf_cache_file = self._results_cache / "passed_failed.csv"

    def get_iterations(self) -> List[Iteration]:
        iterations = [Iteration(path) for path in self.p.glob("*") if Iteration.is_iteration(path)]
        return iterations

    def clear_results_cache(self):
        for dir in self.get_iterations():
            dir.clear_results_cache()

    def clear_junit_data_cache(self):
        for dir in self.get_iterations():
            dir.clear_junit_data_cache()

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

    def get_passed_failed(
        self,
        *,
        read_resultsDir_cache=True,
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

        if write_resultsDir_cache and not did_read_cache:
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
        iterations_in_this_dir = self.get_iterations_overview()["Iteration"].apply(
            lambda it: self.p.name + "/" + it.p.name
        )
        return set(iterations_in_this_dir) == set(cache_df["Iteration"])

    def _compute_passed_failed(self, read_iteration_cache, write_iteration_cache):
        """
        Compute passed-failed table parallelized accross iterations.
        """
        pool = multiprocessing.Pool()
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
        result = pool.map(Iteration.find_keywords_in_tracefiles, self.get_iterations(),)

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
                        "Test_funcname",
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
                    "Test_funcname",
                ]
            )

    def get_coverage_overview(self) -> pd.DataFrame:
        pool = multiprocessing.Pool()
        co = pd.concat(pool.map(Iteration.get_coverage_overview, self.get_iterations()))
        co["Iteration"] = str(self.p) + "/" + co["Iteration"]
        return co

    def __repr__(self) -> str:
        return f"ResultsDir('{self.p}')"


class ResultsDirCollection(IterationCollection):

    """Directory containing (symlinks to) ResultsDirs. Assumes that it only containes such."""

    def __init__(self, dir):
        self.p = Path(dir)
        assert self.p.is_dir(), f"Directory {self.p} does not exist"

    def get_results_dirs(self) -> List[ResultsDir]:
        return [ResultsDir(path) for path in self.p.glob("*")]

    def get_iterations(self):
        return [it for rd in self.get_results_dirs() for it in rd.get_iterations()]

    def get_passed_failed(
        self,
        *,
        read_resultsDir_cache=True,
        write_resultsDir_cache=True,
        read_iteration_cache=True,
        write_iteration_cache=True,
    ) -> pd.DataFrame:
        """
        Collect passed-failed information from each ResultsDir and concat them.
        """
        return pd.concat(
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
        eprint(f"classname_to_nodeid: IndexError with classname={classname}, funcname={funcname}")
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


def main() -> None:
    """The main entry location of the program."""
    # TODO improve startup time by making fire not explore libraries
    fire.Fire()


if __name__ == "__main__":
    main()
