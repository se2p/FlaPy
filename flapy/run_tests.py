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
"""Executes all (specified) tests of a given project"""
import argparse
import contextlib
import logging
import ntpath
import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import (
    Union,
    Callable,
    List,
    Iterable,
    Optional,
    Tuple,
    Any,
    Generator,
    Set,
    Dict,
)

import pipfile  # type: ignore
import virtualenv as virtenv  # type: ignore


class FileUtils:
    """Provides static file utility methods."""

    _copies: List[Any] = []

    @classmethod
    def provide_copy(
        cls,
        src_dir: Union[str, os.PathLike],
        dest_dir: str = None,
    ) -> Union[str, os.PathLike]:
        """Provides a copy of the given source directory and returns the path to it.

        :param src_dir: Path to the source directory
        :param dest_dir: Path to the temporary directory.
            If this option is specified, not a random directory will be created but this one.
        :return: Path to the copied version
        """
        os.mkdir(dest_dir)
        cls.copy_tree(src_dir, dest_dir)
        cls._copies.append(dest_dir)
        return dest_dir

    @classmethod
    def copy_tree(
        cls,
        src: Union[str, os.PathLike],
        dst: Union[str, os.PathLike],
        symlinks: bool = False,
        ignore: Union[
            None,
            Callable[[str, List[str]], Iterable[str]],
            Callable[[Union[str, os.PathLike], List[str]], Iterable[str]],
        ] = None,
    ) -> None:
        """Copies a tree in the filesystem from src to dst.

        :param src: Path to the source
        :param dst: Path to the destination
        :param symlinks: A flag indicating whether symlinks should be copied
        :param ignore: A function for ignoring several files/folders
        """
        for item in os.listdir(src):
            source_path = os.path.join(src, item)
            dest_path = os.path.join(dst, item)
            if os.path.isdir(source_path):
                shutil.copytree(source_path, dest_path, symlinks, ignore)
            else:
                shutil.copy2(source_path, dest_path)

    @classmethod
    def delete_copy(cls, copy_path: Union[str, os.PathLike]) -> None:
        """Deletes a copy.

        Can only delete copies that were generated by this class, for others an
        Exception is raised.

        :param copy_path: Path to the copy that should be deleted
        """
        if copy_path in cls._copies:
            shutil.rmtree(copy_path)
            cls._copies.remove(copy_path)
        else:
            raise Exception(f"Cannot delete copy {copy_path} as it does not exist!")

    @classmethod
    def delete_all_copies(cls) -> None:
        """Deletes all existing copies."""
        for copy in cls._copies:
            try:
                cls.delete_copy(copy)
            except FileNotFoundError:
                pass


class VirtualEnvironment:
    """Wraps a virtual environment."""

    def __init__(self, env_name: str, root_dir) -> None:
        """Creates a new virtual environment in a temporary folder.

        :param env_name: Name of the virtual environment.
        :param root_dir: Directory where the virtual environment will be created.
        """
        self._env_name = env_name
        self._packages: List[str] = []
        self._requirements_files: List[Path] = []
        self._env_dir = f"{root_dir}/flapy_virtual_env"
        virtenv.create_environment(self._env_dir)

    def cleanup(self) -> None:
        """Cleans up the virtual environment."""
        shutil.rmtree(self._env_dir)

    @property
    def env_dir(self) -> Any:
        """Returns the path to the temporary folder the virtual environment is
        installed in.

        :return: The path to the virtual environment folder
        """
        return self._env_dir

    @property
    def env_name(self) -> str:
        """Returns the environment's name

        :return: The environment's name
        """
        return self._env_name

    def add_package_for_installation(self, package_name: str) -> None:
        """Add a package for the installation from PyPI.

        :param package_name: The name of a package on PyPI
        """
        self._packages.append(package_name)

    def add_packages_for_installation(self, package_names: List[str]) -> None:
        """Adds a list of packages for installation from PyPI.

        :param package_names: A list of packages on PyPI
        """
        self._packages.extend(package_names)

    def add_requirements_file_for_installation(self, requirements_file_path: Path) -> None:
        """Add a requirements file to be installed via `pip -r`

        :param requirements_file_path: Path to the requirements_file
        """
        self._requirements_files.append(requirements_file_path)

    def add_requirements_files_for_installation(self, requirements_file_paths: Path) -> None:
        """Add a list of requirements files to be installed via `pip -r`

        :param requirements_file_path: List of paths to the requirements_files
        """
        self._requirements_files.extend(requirements_file_paths)

    def run_commands(self, commands: List[str]) -> Tuple[str, str]:
        """Run commands in the virtual environment setting.

        ATTENTION: Be careful, the commands will be run in a sub-process and can be
        used for possible security flaws! Be sure that you know what you do,
        when executing stuff here!

        :param commands: A list of commands to be executed in the virtual environment
        :return: A tuple of output and error output of the process
        """
        # Source virtual env
        command_list = [
            "source {}".format(os.path.join(self._env_dir, "bin", "activate")),
            "python -V",
        ]
        # Install dependencies
        for reqs_file in self._requirements_files:
            command_list.append(f"pip install -r {reqs_file}")
        for package in self._packages:
            command_list.append(f"pip install {package}")
        # Append other commands
        command_list.extend(commands)
        cmd = ";".join(command_list)
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable="/bin/bash"
        )
        out, err = process.communicate()
        return out.decode("utf-8"), err.decode("utf-8")

    def __str__(self) -> str:
        return f"VirtualEnvironment {self._env_name} in directory {self.env_dir}"

    def __repr__(self) -> str:
        return self.__str__()


@contextlib.contextmanager
def virtualenv(env_name: str, root_dir) -> Generator[VirtualEnvironment, Any, None]:
    """Creates a context around a new virtual environment.

    It creates a virtual environment in a temporary folder and yields and object  of
    the VirtualEnvironment class.

    :param env_name: The name for the virtual environment
    :param root_dir: root folder of the virtual environment
    :return: A VirtualEnvironment object wrapping the virtual environment
    """
    venv = VirtualEnvironment(env_name, root_dir)
    yield venv
    venv.cleanup()


class RandomOrderBucket(Enum):
    """An enum mapping the random bucket values from pytest-random-order.

    See the documentation of pytest-random-order for the meaning of these values.
    """

    CLASS = "class"
    MODULE = "module"
    PACKAGE = "package"
    PARENT = "parent"
    GRANDPARENT = "grandparent"
    GLOBAL = "global"

    def __str__(self):
        return str(self.value)


class PyTestRunner:

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        project_name: str,
        path: Path,
        config: argparse.Namespace,
        logger,
        xml_output_file: Path = None,
        xml_coverage_file: Path = None,
        trace_output_file: Path = None,
        tests_to_be_run: str = "",
    ) -> None:
        self._project_name = project_name
        self._path = Path(path)
        self._config = config
        self._xml_output_file = xml_output_file
        self._xml_coverage_file = xml_coverage_file
        self._trace_output_file = trace_output_file
        self._tests_to_be_run = tests_to_be_run
        self._logger = logger

    def run(self) -> Optional[Tuple[str, str]]:
        """Install dependencies and execute pytest"""

        with virtualenv(self._project_name, self._config.temp) as env:
            old_cwd = Path(os.getcwd())
            os.chdir(self._path)

            # INSTALL PROJECT DEPENDENCIES
            # There are two different methods for dependency installation
            # 1. search for dependencies in typical files like 'requirements.txt'
            # 2. install the project itself (requires pypi-tag to be specified)
            if self._config.pypi_tag in [None, ""]:
                self._logger.info(
                    "no pypi tag specified -> falling back to searching for requirements"
                )
                # packages = self.find_dependencies()
                # env.add_packages_for_installation(packages)

                reqs_files = self.find_requirements_files()
                self._logger.info(
                    f"found the following requirements files: {[str(reqs_file) for reqs_file in reqs_files]}"
                )
                env.add_requirements_files_for_installation(reqs_files)
            else:
                self._logger.info(f"pypi tag found {self._config.pypi_tag}")
                env.add_package_for_installation(f"{self._project_name}=={self._config.pypi_tag}")

            # INSTALL TEST EXECUTION DEPENDENCIES
            env.add_package_for_installation("pytest==6.2.5")
            if self._config.random_order_bucket is not None:
                env.add_package_for_installation("pytest-random-order==1.0.4")

            # START BUILDING COMMAND
            command = ""

            # USE TRACING?
            if self._config.trace not in [None, ""]:
                command += f'pytest_trace "{self._config.trace}" {self._trace_output_file} '
            else:
                command += "pytest "

            # GENERAL PYTEST FLAGS
            command += "-v --rootdir=. "

            # JUNIT XML OUTPUT
            if self._xml_output_file is not None:
                command += f"--junitxml={self._xml_output_file} "

            # RANDOM TEST ORDER?
            if self._config.random_order_bucket is not None:
                command += f"--random-order-bucket={self._config.random_order_bucket} "

            # RANDOM ORDER SEED?
            if self._config.random_order_seed is not None:
                command += f"--random-order-seed={self._config.random_order_seed} "

            # COVERAGE
            if self._xml_coverage_file is not None:
                env.add_package_for_installation("pytest-cov==2.8.1")
                command += f"--cov=. --cov-branch --cov-report xml:{self._xml_coverage_file} "

            # TESTS TO BE RUN
            command += f"{self._tests_to_be_run} "

            # PYTHONPATH=. is necessary to execute tests which are not contained in a module
            command = "PYTHONPATH=. " + command

            # add debug output
            commands = [
                'echo "which python: $(which python)"',
                'echo "which pip:    $(which pip)"',
                'echo "which pytest: $(which pytest)"',
                'echo "python path: "',
                'python -c "import sys; print(sys.path)"',
                command,
            ]

            self._logger.info(f"executing commands {commands}")
            out, err = env.run_commands(commands)
            os.chdir(old_cwd)
            return out, err

    def find_requirements_files(self) -> List[Path]:
        """Search for *requirements*.txt files in the project path

        :returns: List of existing requirements files

        """
        return list(self._path.glob("*requirements*.txt"))

    def find_dependencies(self) -> List[str]:
        """Search for dependencies in common files"""
        packages: List[str] = []
        possible_requirements_filenames = [
            "requirements.txt",
            "dev-requirements.txt",
            "dev_requirements.txt",
            "test-requirements.txt",
            "test_requirements.txt",
            "requirements-dev.txt",
            "requirements_dev.txt",
            "requirements-test.txt",
            "requirements_test.txt",
        ]
        for file_name in possible_requirements_filenames:
            packages.extend(self.read_dependencies_from_requirements_file(self._path / file_name))
        if (self._path / "Pipfile").is_file():
            packages.extend(self.read_dependencies_from_pipfile())
        return packages

    @staticmethod
    def read_dependencies_from_requirements_file(requirements_file: Path) -> List[str]:
        packages: List[str] = []
        if requirements_file.is_file():
            with open(requirements_file) as req_file:
                packages = [
                    line.strip() for line in req_file.readlines() if "requirements" not in line
                ]
        return packages

    def read_dependencies_from_pipfile(self) -> List[str]:
        packages: List[str] = []
        pip_file = pipfile.load(self._path / "Pipfile")
        data = pip_file.data
        if data["default"]:
            for key, _ in data["default"].items():
                packages.append(key)
            if data["develop"]:
                for key, _ in data["develop"].items():
                    packages.append(key)
        return packages

    def __str__(self) -> str:
        return "Runner for project {} in path {} (type {})".format(
            self._project_name, self._path, type(self).__name__
        )

    def __repr__(self) -> str:
        return self.__str__()


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class FlakyAnalyser:
    """Analyses a repository for possible flaky tests."""

    def __init__(self, argv: List[str]) -> None:
        parser = self._create_parser()
        self._config = parser.parse_args(argv[1:])
        self._logger = self._configure_logger()
        self._repo_path = self._config.repository
        self._temp_path = Path(self._config.temp)
        self._flaky_tests: Set[str] = set()
        self._test_cases: Dict[str, str] = {}
        self._tests_to_be_run: str = self._config.tests_to_be_run

    def run(self):
        """
        Runs the tests the required number of times (10 by default) with an instance of
        the given runner_class and creates xml files containing the results.
        """
        self._logger.info(f"Config: {self._config}")
        naming_offset = 0 if self._config.random_order_bucket is None else self._config.num_runs
        repo_copy_dir = self._temp_path / "flapy_repo_copy"

        # TODO add option to run tests_to_be_run one at a time or all togehter
        for test_to_be_run in self._tests_to_be_run.split() or [""]:
            for i in range(self._config.num_runs):
                self._logger.info(
                    f"Iteration {i} of {self._config.num_runs} for project {self._config.project_name} (random={self._config.random_order_bucket})"
                )

                copy: str = FileUtils.provide_copy(
                    src_dir=self._config.project_name, dest_dir=repo_copy_dir
                )

                run_num = i + naming_offset
                ttbr_id = test_to_be_run.replace("/", ".")

                def get_output_filename(keyword, ending) -> Path:
                    return (
                        self._temp_path
                        / f"{self._config.project_name}_{keyword}{run_num}{ttbr_id}.{ending}"
                    )

                xml_output_file: Path = get_output_filename("output", "xml")
                xml_coverage_file: Path = get_output_filename("coverage", "xml")
                trace_file: Path = get_output_filename("trace", "")

                runner = PyTestRunner(
                    project_name=self._config.project_name,
                    path=Path(copy),
                    config=self._config,
                    xml_output_file=xml_output_file,
                    xml_coverage_file=xml_coverage_file,
                    trace_output_file=trace_file,
                    tests_to_be_run=test_to_be_run,
                    logger=self._logger,
                )
                out, err = runner.run()
                self._logger.debug("OUT: %s", out)
                self._logger.debug("ERR: %s", err)

                if not xml_output_file.is_file():
                    self._logger.warning(
                        "Did not create file %s while running the tests.",
                        xml_output_file,
                    )

                FileUtils.delete_copy(copy)

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            fromfile_prefix_chars="@",
            description="""
            An analysing tool for bug-fix commits of Git repositories.
            """,
        )

        parser.add_argument("-l", "--logfile", dest="logfile", help="Path to log file.")
        parser.add_argument(
            "-r",
            "--repository",
            dest="repository",
            help="A path to a folder containing the checked-out version of the " "repository.",
            required=True,
        )
        parser.add_argument(
            "-t",
            "--temp",
            dest="temp",
            help="Path to the temp directory",
            required=True,
        )
        parser.add_argument("--project-name", dest="project_name")
        parser.add_argument(
            "--pypi-tag",
            dest="pypi_tag",
            default=None,
            type=str,
            help="To install the dependencies of the project under test, FlaPy tries to install the project itself from PyPI via its name (arguement --project-name) and its pypi-tag. It is beeing assumed, that this version matches the version currently checked out in the repository. If no PyPI tag is specified, FlaPy searches for dependencies in files like 'requirements.txt'",
        )
        parser.add_argument(
            "-n",
            "--number-test-runs",
            dest="num_runs",
            help="The number of times the tests are run.",
            type=int,
            default=10,
            required=False,
        )
        parser.add_argument(
            "-b",
            "--random-order-bucket",
            dest="random_order_bucket",
            default=None,
            type=RandomOrderBucket,
            choices=list(RandomOrderBucket),
            help="Select the strategy for buckets on random-order test execution.  "
                 "The default value is `module'.  See the documentation of the "
                 "`pytest-random-order' plugin for details on these values.",
        )
        parser.add_argument(
            "-s",
            "--random-order-seed",
            dest="random_order_seed",
            type=int,
            required=False,
            help="An optional seed for random-order test execution.",
        )
        parser.add_argument(
            "-a",
            "--trace",
            dest="trace",
            required=False,
            type=str,
            help="NodeIDs of the functions that shall be traced. "
                 'Example: "tests/test_file.py::test_func1 test_file.py::TestClass::test_func2 '
                 "Note: Only the name of the file (test_file.py) will be used, "
                 "the rest of the path (tests/) is discarded",
        )
        parser.add_argument(
            "--tests-to-be-run",
            dest="tests_to_be_run",
            required=False,
            type=str,
            default="",
            help="NodeIDs of the functions that shall be executed. "
                 "Multiple names must be separated by spaces and "
                 "will be executed each individually in a new pytest run. "
                 'Example: "tests/test_file.py::test_func1 tests/test_file.py::TestClass::test_func2',
        )

        return parser

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger("RepositoryAnalyser")
        logger.setLevel(logging.DEBUG)

        if self._config.logfile:
            file = self._config.logfile
        else:
            file = os.path.join(os.path.dirname("__file__"), "flapy.log")

        log_file = logging.FileHandler(file)
        log_file.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s](%(name)s:%(funcName)s:%(lineno)d: " "%(message)s"
            )
        )
        log_file.setLevel(logging.DEBUG)
        logger.addHandler(log_file)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("[%(levelname)s](%(name)s): %(message)s"))
        logger.addHandler(console)

        return logger


def main(argv: List[str] = None) -> None:
    """The main entry location of the program."""
    if not argv:
        argv = sys.argv
    analyser = FlakyAnalyser(argv)
    analyser.run()


if __name__ == "__main__":
    main(sys.argv)
