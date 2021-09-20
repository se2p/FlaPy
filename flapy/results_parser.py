#!/usr/bin/env python3
import re
import os
import sys
import tarfile

# import traceback
import logging
import multiprocessing
import fire  # type: ignore

import xml
import operator as op
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
)
from functools import partial
from ast import literal_eval
from pathlib import Path
from abc import abstractmethod
from functools import lru_cache, reduce
from junitparser import (  # type: ignore
    Failure,
    Skipped,
    Error,
    JUnitXml,
    TestCase,
    JUnitXmlError,
)
from flapy.utils import try_default, eprint

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

logging.getLogger().setLevel(logging.DEBUG)


def read_junit_testcase(test_case: TestCase, include_errors) -> Dict[str, Union[str, int]]:
    row = {
        "file": test_case._elem.get("file"),
        "class": test_case.classname,
        "name": test_case.name,
        "verdict": try_default(
            lambda: Verdict.from_junitparser(test_case.result),
            JUnitXmlError,
            Verdict.PARSE_ERROR,
        ),
    }
    if include_errors:
        row.update({
            "message": test_case.result.message
                if test_case.result and test_case.result.message else "NO MESSAGE",
            "result-text": re.findall(r"(\w*Error.*)\n", test_case.result._elem.text)
                if test_case.result and test_case.result._elem else [],
            "system-err": re.findall(r"(\w*Error.*)\n", test_case.system_err)
                if test_case.system_err else [],
        })
    return row


def is_empty(openvia: Callable[[str], IO], path: str):
    try:
        with openvia(path) as f:
            next(f)
    except StopIteration:
        return True
    return False


def junitxml_classname_to_modname_and_actual_classname(
    classname: str,
) -> Tuple[List[str], str]:
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
        print(
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
    def __init__(self, file_: str):
        self._df = pd.read_csv(
            file_,
            # converters={
            #     'Passed_sameOrder': eval_string_to_set,
            #     'Failed_sameOrder': eval_string_to_set,
            #     'Error_sameOrder': eval_string_to_set,
            #     'Skipped_sameOrder': eval_string_to_set,
            #     'Passed_randomOrder': eval_string_to_set,
            #     'Failed_randomOrder': eval_string_to_set,
            #     'Error_randomOrder': eval_string_to_set,
            #     'Skipped_randomOrder': eval_string_to_set,
            # }
        )
        # Some junit-xml files actually had name="" in them
        self._df = self._df[~self._df["Test_funcname"].isna()]
        # Rows with NaN are ignored by pd.groupby -> fillna
        self._df["Test_filename"] = self._df["Test_filename"].fillna("")
        self._df["Test_classname"] = self._df["Test_classname"].fillna("")
        self._df["Test_parametrization"] = self._df["Test_parametrization"].fillna("")

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
        self._df["Verdict_sameOrder"] = self._df["Verdicts_sameOrder"].apply(
            lambda s: Verdict.decide_overall_verdict(eval_string_to_set(s))
        )
        self._df["Verdict_randomOrder"] = self._df["Verdicts_randomOrder"].apply(
            lambda s: Verdict.decide_overall_verdict(eval_string_to_set(s))
        )

        self._df["Flaky_sameOrder_withinIteration"] = (
            self._df["Verdict_sameOrder"] == Verdict.FLAKY
        )
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
                "Flaky_sameOrder_withinIteration": any,
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
                "Flaky_randomOrder_withinIteration": any,
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
            }
        )
        test_overview["Verdict_sameOrder"] = test_overview["Verdicts_sameOrder"].apply(
            Verdict.decide_overall_verdict
        )
        test_overview["Verdict_randomOrder"] = test_overview[
            "Verdicts_randomOrder"
        ].apply(Verdict.decide_overall_verdict)

        self._df.drop(
            ["Flaky_sameOrder_withinIteration", "Flaky_randomOrder_withinIteration"],
            axis="columns",
            inplace=True,
        )

        # recalculate Order-dependent
        test_overview["Order-dependent"] = (
            ~test_overview["Flaky_sameOrder_withinIteration"]
        ) & (test_overview["Flaky_randomOrder_withinIteration"])

        test_overview["Flaky_Infrastructure"] = (
            # deter
            (
                (test_overview["Verdict_sameOrder"] == Verdict.FLAKY)
                & ~test_overview["Flaky_sameOrder_withinIteration"]
            )
            | (
                # non-deter
                (test_overview["Verdict_sameOrder"] != Verdict.FLAKY)
                & (test_overview["Verdict_randomOrder"] == Verdict.FLAKY)
                & (~test_overview["Order-dependent"])
            )
        ) & ~test_overview["Order-dependent"]

        # modname_classname = test_overview['Test_classname'].apply(
        #     junitxml_classname_to_modname_and_actual_classname
        # )
        # test_overview['Test_actual_classname'] = [c for _, c in modname_classname]
        # test_overview['Test_modname[-1]'] = [
        #     m[-1] if len(m) > 0 else '' for m, _ in modname_classname
        # ]
        test_overview["Test_nodeid"] = test_overview.apply(
            lambda s: to_nodeid(
                s["Test_filename"], s["Test_classname"], s["Test_funcname"]
            ),
            axis=1,
        )
        test_overview["Test_nodeid_inclPara"] = (
            test_overview["Test_nodeid"] + test_overview["Test_parametrization"]
        )
        return test_overview

    def __repr__(self):
        return "PassedFailed"


class TestsOverview:
    def __init__(self, file_: str):
        self._df = pd.read_csv(file_).fillna("")

    def get_od_flaky_tests(self):
        self._df["Test_nodeid_inclPara"] = (
            self._df["Test_nodeid"] + self._df["Test_parametrization"]
        )
        od_flaky_tests = self._df[self._df["Order-dependent"]]
        return od_flaky_tests[
            ["Project_Name", "Project_URL", "Project_Hash", "Test_nodeid_inclPara"]
        ].drop_duplicates()

    def to_classification_template(self) -> pd.DataFrame:
        deter_flaky_tests = self._df[self._df["Flaky_sameOrder_withinIteration"]]
        classification_template = deter_flaky_tests[
            [
                "Project_Name",
                "Project_URL",
                "Project_Hash",
                "Test_filename",
                "Test_classname",
                "Test_funcname",
                "Test_parametrization",
            ]
        ].drop_duplicates()
        classification_template["project domain"] = ""
        classification_template["flakiness category"] = ""
        classification_template["sure? 1=yes 4=no"] = ""
        classification_template["flaky fixture?"] = ""
        classification_template["comment"] = ""
        return classification_template

    def to_deter_flaky_tests(self) -> pd.DataFrame:
        deter_flaky_tests = self._df[self._df["Flaky_sameOrder_withinIteration"]]
        return deter_flaky_tests[
            ["Project_Name", "Project_URL", "Project_Hash", "Test_nodeid"]
        ].drop_duplicates()

    def to_od_flaky_tests(self) -> pd.DataFrame:
        od_flaky_tests = self._df[self._df["Order-dependent"]]
        return od_flaky_tests[
            ["Project_Name", "Project_URL", "Project_Hash", "Test_nodeid"]
        ].drop_duplicates()

    def to_flapy_input(self, num_runs: int) -> pd.DataFrame:
        self._df["Test_nodeid"] = self._df.apply(
            lambda s: to_nodeid(
                s["Test_filename"], s["Test_classname"], s["Test_funcname"], s["Test_parametrization"]
            ),
            axis=1,
        )
        self._df["Funcs_to_trace"] = ""
        self._df["Num_runs"] = num_runs
        return self._df[["Project_Name", "Project_URL", "Project_Hash", "Funcs_to_trace", "Test_nodeid", "Num_runs"]]


class CoverageCsv:
    def __init__(self, file_: str):
        self._df = pd.read_csv(file_).fillna("")

    def to_overview(self):
        self._df["BranchCoverage_weighted"] = (
            self._df["BranchCoverage"] * self._df["number_of_entries"]
        )
        self._df["LineCoverage_weighted"] = (
            self._df["LineCoverage"] * self._df["number_of_entries"]
        )

        self._df = self._df.groupby(
            ["Project_Name", "Project_URL", "Project_Hash"], as_index=False
        ).agg(
            {
                "number_of_entries": sum,
                "BranchCoverage_weighted": sum,
                "LineCoverage_weighted": sum,
            }
        )
        self._df["BranchCoverage"] = (
            self._df["BranchCoverage_weighted"] / self._df["number_of_entries"]
        )
        self._df["LineCoverage"] = (
            self._df["LineCoverage_weighted"] / self._df["number_of_entries"]
        )
        self._df.drop("BranchCoverage_weighted", axis="columns")
        self._df.drop("LineCoverage_weighted", axis="columns")
        return self._df.drop_duplicates(["Project_Name", "Project_URL"])


class MyFileWrapper:
    def __init__(
        self,
        path_to_output_file: Union[str, Path],
        project_name: str,
        openvia: Callable[[str], IO] = open,
    ):
        self.p: Path = Path(path_to_output_file)
        self.project_name: str = project_name
        self.openvia = openvia

    @classmethod
    @abstractmethod
    def get_regex(cls, project_name: str):
        """
        Regex should have number as first and only group
        :param project_name:
        :return:
        """
        pass

    @classmethod
    def is_(cls, path: Path, project_name: str, openvia: Callable[[str], IO]) -> bool:
        return re.match(
            cls.get_regex(project_name), str(path)
        ) is not None and not is_empty(openvia, str(path))

    # @classmethod
    # def is_(cls, path: Path, project_name: str, openvia: Callable[[str], IO]) -> bool:
    #     return re.match(cls.get_regex(project_name), str(path)) is not None

    def get_num(self) -> int:
        num = re.findall(self.get_regex(self.project_name), str(self.p))[0]
        return int(num)

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
        return rf".*/{project_name}_coverage(\d+).xml$"

    def get_num(self) -> int:
        num = re.findall(self.get_regex(self.project_name), str(self.p))[0]
        return int(num)

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
        return rf".*/deterministic/tmp/{project_name}_coverage(\d+)\.xml"


class CoverageXmlFileNonDeter(CoverageXmlFile):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/non-deterministic/tmp/{project_name}_coverage(\d+)\.xml"


class JunitXmlFile(MyFileWrapper):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/tmp/{project_name}_output(\d+).*\.xml$"

    def get_num(self) -> int:
        num = re.findall(self.get_regex(self.project_name), str(self.p))[0]
        return int(num)

    def get_order(self) -> str:
        if JunitXmlFileDeter.is_(self.p, self.project_name, self.openvia):
            return "deter"
        if JunitXmlFileNonDeter.is_(self.p, self.project_name, self.openvia):
            return "non-deter"
        return "COULD_NOT_GET_ORDER"

    def parse(self) -> JUnitXml:
        with self.open() as f:
            try:
                return JUnitXml.fromfile(f, ET.parse)
            except xml.etree.ElementTree.ParseError:
                return JUnitXml()

    def get_hostname(self) -> str:
        return list(self.parse())[0].hostname

    def get_testcases(self) -> List[TestCase]:
        junit_xml = self.parse()
        return [case for suite in junit_xml for case in suite]

    def to_table(self, include_errors) -> List[Dict[str, Union[str, int]]]:
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
                # try_default(
                    # lambda:
                    {
                        **read_junit_testcase(test_case, include_errors),
                        "num": self.get_num(),
                        "order": self.get_order(),
                    }#,
                    # Exception,
                    # {},
                # )
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
        return rf".*/deterministic/tmp/{project_name}_output(\d+).*\.xml"


class JunitXmlFileNonDeter(JunitXmlFile):
    @classmethod
    def get_regex(cls, project_name: str):
        return rf".*/non-deterministic/tmp/{project_name}_output(\d+).*\.xml"


class TraceFile(MyFileWrapper):
    """
    File containing traces
    """

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


class ProjectResultsDir:
    """
    Example: flapy-results_20200101/lineflow
    """

    def __init__(self, dir: Union[str, Path]):
        self.p = Path(dir)
        assert ProjectResultsDir.is_project_results_dir(self.p)
        self.use_extracted = (self.p / "local").is_dir()
        self._archive: Optional[tarfile.TarFile] = None
        self._results_cache = self.p / "results_cache"
        if not self._results_cache.is_dir():
            self._results_cache.mkdir()
        self._junit_cache_file = self._results_cache / "junit_data.csv"

    @classmethod
    def is_project_results_dir(cls, path: Path) -> bool:
        return (
            path.is_dir()
            and path.name != "run"
            and not path.name.startswith(".")
            and ((path / "local").is_dir() or (path / "results.tar.xz").is_file())
        )

    @lru_cache()
    def get_project_name(self) -> str:
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
                    return "COULD_NOT_GET_PROJECT_NAME"

    def get_project_url(self) -> str:
        if (self.p / "project-url.txt").is_file():
            with open(self.p / "project-url.txt") as file:
                return file.read().replace("\n", "")
        return "COULD_NOT_GET_PROJECT_URL"

    def get_project_git_hash(self) -> str:
        if (self.p / "project-git-hash.txt").is_file():
            with open(self.p / "project-git-hash.txt") as file:
                return file.read().replace("\n", "")
        return "COULD_NOT_GET_PROJECT_GIT_HASH"

    def get_flakyanalysis_git_hash(self) -> str:
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
        include_errors=True
    ) -> pd.DataFrame:
        did_read_cache = False
        if read_cache and self._junit_cache_file.is_file():
            junit_data: pd.DataFrame = pd.read_csv(self._junit_cache_file)
            did_read_cache = True
        else:
            columns = list(read_junit_testcase(TestCase(), include_errors).keys()) + ["num", "order"]
            junitxml_files = self.get_files(JunitXmlFile)
            junit_data = pd.DataFrame(
                [
                    test_case
                    for junit_file in junitxml_files
                    for test_case in junit_file.to_table(include_errors)
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
        junit_data = junit_data
        if len(junit_data) == 0:
            return pd.DataFrame()
        junit_data["Test_filename"] = junit_data["file"]
        junit_data["Test_classname"] = junit_data["class"]
        junit_data["Test_funcname"] = [
            re.sub(r"\[.*\]", "", name) for name in junit_data["name"]
        ]
        junit_data["Test_parametrization"] = [
            re.findall(r"(\[.*\])", name)[0]
            if len(re.findall(r"(\[.*\])", name)) > 0
            else ""
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

        passed_failed["Verdict"] = passed_failed["Verdicts"].apply(
            Verdict.decide_overall_verdict
        )

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

        return passed_failed[
            [
                "Iteration",
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

    # TODO delete -> not needed -> use PassedFailed.to_test_overview
    def get_test_overview(self) -> pd.DataFrame:
        junit_data = self.get_junit_data(include_project_columns=True)
        junit_data["Test_filename"] = junit_data["file"]
        junit_data["Test_classname"] = junit_data["class"]
        junit_data["Test_funcname"] = [
            re.sub(r"\[.*\]", "", name) for name in junit_data["name"]
        ]
        junit_data["Test_parametrization"] = [
            re.findall(r"(\[.*\])", name)[0]
            if len(re.findall(r"(\[.*\])", name)) > 0
            else ""
            for name in junit_data["name"]
        ]
        junit_data["Passed"] = junit_data["verdict"] == Verdict.PASS
        junit_data["Failed"] = junit_data["verdict"] == Verdict.FAIL
        junit_data["Error"] = junit_data["verdict"] == Verdict.ERROR
        junit_data["Skipped"] = junit_data["verdict"] == Verdict.SKIP
        junit_data["Verdict"] = junit_data["verdict"]

        test_overview = junit_data.groupby(
            [
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
                "Verdict": lambda l: Verdict.decide_overall_verdict(set(l)),
                "Passed": lambda l: sum(l),
                "Failed": lambda l: sum(l),
                "Error": lambda l: sum(l),
                "Skipped": lambda l: sum(l),
            }
        )

        test_overview = test_overview.astype(
            {"Passed": int, "Failed": int, "Error": int, "Skipped": int}
        )

        test_overview = pd.merge(
            test_overview[test_overview["order"] == "deter"],
            test_overview[test_overview["order"] == "non-deter"],
            on=[
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
        test_overview["Order-dependent"] = (
            test_overview["Verdict_sameOrder"] != "Flaky"
        ) & (test_overview["Verdict_randomOrder"] == "Flaky")

        return test_overview

    def get_coverage_raw_data(self) -> List[Dict]:
        """

        :returns: EXAMPLE:
            [
                { 'num': 0, 'order': "deter", "BranchCoverage": 0.7, "LineCoverage": 0.6 },
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
                    name,
                    self.get_project_name(),
                    openvia=self.get_archive().extractfile,  # type: ignore
                )
                for name in self.get_archive_names()
                if type_.is_(
                    Path(name),
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

    def __repr__(self) -> str:
        return f"ProjectResultsDir('{self.p}')"


class ResultsDir:
    """
    Directory created by one execution of flapy.
    Example: flapy-results_20200101
    """

    def __init__(self, dir: Union[str, Path]):
        self.p = Path(dir)
        assert self.p.is_dir()

    def clear_results_cache(self):
        for dir in self.get_project_results_dirs():
            dir.clear_results_cache()

    def clear_junit_data_cache(self):
        for dir in self.get_project_results_dirs():
            dir.clear_junit_data_cache()

    def get_junit_data(
        self,
        *,
        include_project_columns=True,
        read_cache=True,
        write_cache=True,
        return_nothing=False,
    ) -> pd.DataFrame:
        pool = multiprocessing.Pool()
        return pd.concat(
            pool.map(
                partial(
                    ProjectResultsDir.get_junit_data,
                    include_project_columns=include_project_columns,
                    read_cache=read_cache,
                    write_cache=write_cache,
                    return_nothing=return_nothing,
                ),
                self.get_project_results_dirs(),
            )
        )

    @lru_cache()
    def get_project_results_dirs(self) -> List[ProjectResultsDir]:
        project_results_dirs = [
            ProjectResultsDir(path)
            for path in self.p.glob("*")
            if ProjectResultsDir.is_project_results_dir(path)
        ]
        return project_results_dirs

    def get_passed_failed(self, *, read_cache=True, write_cache=True) -> pd.DataFrame:
        pool = multiprocessing.Pool()
        passed_failed = pd.concat(
            pool.map(
                partial(
                    ProjectResultsDir.get_passed_failed,
                    read_cache=read_cache,
                    write_cache=write_cache,
                ),
                self.get_project_results_dirs(),
            )
        )
        passed_failed["Iteration"] = self.p.name + "/" + passed_failed["Iteration"]
        return passed_failed

    def get_project_dirs_overview(self) -> pd.DataFrame:
        proj_dirs_overview = pd.DataFrame(
            [
                (
                    proj_dir.p,
                    proj_dir.get_project_name(),
                    proj_dir.get_project_url(),
                    proj_dir.get_project_git_hash(),
                )
                for proj_dir in self.get_project_results_dirs()
            ],
            columns=["Iteration", "Project_Name", "Project_URL", "Project_Hash"],
        )
        return proj_dirs_overview

    def get_no_space_left(self) -> pd.DataFrame:
        return pd.concat(
            map(ProjectResultsDir.get_no_space_left, self.get_project_results_dirs())
        )

    # TODO delete
    def get_test_overview(self):
        test_overview = pd.concat(
            map(ProjectResultsDir.get_test_overview, self.get_project_results_dirs())
        )

        test_overview = test_overview.groupby(
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
            group_keys=False,
        ).agg(
            {
                "Verdict_sameOrder": lambda l: Verdict.decide_overall_verdict(set(l)),
                "Passed_sameOrder": sum,
                "Failed_sameOrder": sum,
                "Error_sameOrder": sum,
                "Skipped_sameOrder": sum,
                "Verdict_randomOrder": lambda l: Verdict.decide_overall_verdict(set(l)),
                "Passed_randomOrder": sum,
                "Failed_randomOrder": sum,
                "Error_randomOrder": sum,
                "Skipped_randomOrder": sum,
            }
        )
        test_overview = test_overview.astype(
            {
                "Passed_sameOrder": int,
                "Failed_sameOrder": int,
                "Error_sameOrder": int,
                "Skipped_sameOrder": int,
                "Passed_randomOrder": int,
                "Failed_randomOrder": int,
                "Error_randomOrder": int,
                "Skipped_randomOrder": int,
            }
        )
        test_overview["Order-dependent"] = (
            test_overview["Verdict_sameOrder"] != "Flaky"
        ) & (test_overview["Verdict_randomOrder"] == "Flaky")
        test_overview["Test_nodeid"] = test_overview.apply(
            lambda s: to_nodeid(
                s["Test_filename"], s["Test_classname"], s["Test_funcname"]
            ),
            axis=1,
        )
        return test_overview

    def find_keywords_in_tracefiles(
        self, *, keywords=default_flaky_keywords
    ) -> pd.DataFrame:
        pool = multiprocessing.Pool()
        result = pool.map(
            ProjectResultsDir.find_keywords_in_tracefiles,
            self.get_project_results_dirs(),
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
        co = pd.concat(
            pool.map(
                ProjectResultsDir.get_coverage_overview, self.get_project_results_dirs()
            )
        )
        co["Iteration"] = str(self.p) + "/" + co["Iteration"]

        # co["BranchCoverage_weighted"] = co["BranchCoverage"] * co["number_of_entries"]
        # co["LineCoverage_weighted"] = co["LineCoverage"] * co["number_of_entries"]
        #
        # co = co.groupby(["Project_Name", "Project_URL", "Project_Hash"]).agg({
        #     "number_of_entries": sum,
        #     "BranchCoverage_weighted": sum,
        #     "LineCoverage_weighted": sum
        # })
        # co["BranchCoverage_mean"] = co["BranchCoverage_weighted"] / co["number_of_entries"]
        return co

    def __repr__(self) -> str:
        return f"ResultsDir('{self.p}')"


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
    def from_junitparser(result: Union[Failure, Skipped, Error, None]) -> str:
        if isinstance(result, Failure):
            return Verdict.FAIL
        if isinstance(result, Skipped):
            return Verdict.SKIP
        if isinstance(result, Error):
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
            (
                Verdict.PASS in verdicts
                and (Verdict.FAIL in verdicts or Verdict.ERROR in verdicts)
            )
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
        eprint(
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
        raise ValueError(
            f'Unrecognized start sequence "{prefix}" {additional_error_information}'
        )
    depth = int(prefix.count("-") / 2)
    func = literal_eval(func)
    func = tuple(["" if x is None else x.replace(" ", "_") for x in func])
    tags = tags.strip()

    return call_return, depth, func, tags, arg_hash


def main() -> None:
    """The main entry location of the program."""
    fire.Fire()


if __name__ == "__main__":
    main()
