# FlaPy

FlaPy is a small tool that allows software developers and researchers to identify flaky tests within a given set of projects by rerunning their test suites.

It is the result of research carried out at the
[Chair of Software Engineering II](https://www.fim.uni-passau.de/lehrstuhl-fuer-software-engineering-ii/)
at the [University of Passau](https://www.uni-passau.de), Germany.


## Using FlaPy


### Installation

System requirements: `docker` (executable without root privileges)

Clone the repository to get the helper scripts:
```bash
git clone https://github.com/se2p/flapy

cd flapy/
```
FlaPy’s main entry point is the script `flapy.sh`, which offers two commands: `run` and `parse`.
The FlaPy docker image will be pulled automatically on first usage.

### Preparing the input-csv

Prepare a CSV file with the following columns (example: `flapy_input_example.csv`):
```
PROJECT_NAME,PROJECT_URL,PROJECT_HASH,PYPI_TAG,FUNCS_TO_TRACE,TESTS_TO_BE_RUN,NUM_RUNS
```

Every line in the input file will result in one execution of the container. We call this an *iteration*.
You can have duplicate lines in this input file to analyze the same project multiple times.
In fact, we actively use this to detect infrastructure flakiness, which might occur only between iterations, not within.
PROJECT_NAME, PROJECT_URL and PROJECT_HASH will be used to uniquely identify a project when accumulating results across multiple iterations.
PROJECT_URL can also be local directory, which will then be copied into the container.
PYPI_TAG is used to install the project itself via pip before executing its testsuite to fetch it's dependencies.
If PYPI_TAG is empty, FlaPy will fall back to searching for requirements in common files like requirements.txt


### Run tests locally

Example (takes ~ 1h):
```bash
#              [OPTIONS...]                                 INPUT_CSV
./flapy.sh run --out-dir example_results --plus-random-runs flapy_input_example.csv
```


### Run tests on SLURM cluster

```bash
./flapy.sh run --out-dir example_results \
  --plus-random-runs \
  --run-on cluster --constraint CONSTRAINT \
  flapy_input_example.csv
```
where `CONSTRAINT` is forwarded to `sbatch --constraint`


### Analyze results

```bash
./flapy.sh parse ResultsDirCollection \
  --dir example_results \
  get_tests_overview _df \
  to_csv --index=False example_results_to.csv
```
Note: the directory specified after `--dir` needs to be accessible from the current working directory since only the current working directory is mounted to the container that is started in the background!!


## Contributing

### Building FlaPy

Clone FlaPy:

```bash
git clone https://github.com/se2p/flapy
cd flapy
```

Building the container image:  
We use containers to run the projects' test suites in an isolated environment.

```bash
docker build -t my_flapy -f Dockerfile .
```
This image can be used together with all existing scripts by changing the `FLAPY_DOCKER_IMAGE` variable in `setup_docker_command.sh` to `localhost/my_flapy`.


### Building and running outside docker

Prerequisites
- Python in at least version 3.8.
- You have installed the latest version of [`poetry`](https://python-poetry.org).
    - `pip install poetry`


Install FlaPy locally:

```bash
poetry install
```


Build FlaPy using the `poetry` tool:  
This command will build two files in the `dist` folder: A `tar.gz` archive and a `whl` Python wheel file.

```bash
poetry build
```


## TODOs

- [ ] Use ordered sets or lists in output csv files to always get the same (string-equivalent) output
    * Many columns in passed_failed.csv are sets and their ordering is different from run to run


## Contact

If you want to contact me, please find our contact details on my
[page at the University of Passau](https://www.fim.uni-passau.de/lehrstuhl-fuer-software-engineering-ii/lehrstuhlteam/).

## License

This project is licensed under the terms of the GNU Lesser General Public License.

