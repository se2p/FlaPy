# FlaPy

FlaPy is a small tool that allows software developers and researchers to identify flaky tests within a given set of projects by rerunning their test suites.

It is the result of research carried out at the
[Chair of Software Engineering II](https://www.fim.uni-passau.de/lehrstuhl-fuer-software-engineering-ii/)
at the [University of Passau](https://www.uni-passau.de), Germany.


## Build and run FlaPy


### Prerequisites

Before you begin, ensure you have met the following requirements:
- Python in at least version 3.8.
- You have installed the latest version of [`poetry`](https://python-poetry.org).
    - `pip install poetry`
- You have [`podman`](https://podman.io/) installed
    - podman is an alternative to docker, which can run rootless containers natively
    - you can use docker instead, but you might have to adjust some scripts, especially `run_container.sh`


### Building FlaPy

Clone FlaPy:

```bash
git clone https://github.com/se2p/flapy
cd flapy
```


Install FlaPy locally:

```bash
poetry install
```


Build FlaPy using the `poetry` tool:  
This command will build two files in the `dist` folder: A `tar.gz` archive and a `whl` Python wheel file.
The wheel will be used in the next step to install FlaPy inside the container.

```bash
poetry build
```


Building the container image:  
We use containers to run the projects' test suites in an isolated environment.

```bash
podman build -t flapy .
```


### Run FlaPy

Results will be written to a folder `flapy-results_DATE_TIME`, e.g. `flapy-results_20220315_1215`
```bash
           # RUN_ON  CSV_FILE         PLUS_RANDOM_RUNS  ADDITIONAL_OPTIONS
./run_csv.sh local   sample_input.csv true              ""
```


### Analyse results


Parse the results (replace `flapy-results_DATE_TIME` with the actual name of the directory)
```bash
poetry run results_parser ResultsDir --dir=flapy-results_DATE_TIME get_passed_failed to_csv --index=False > passed_failed_sample.csv
```
In `passed_failed_sample.csv` you can now find for each iteration and each test case in the project's test suite, which test runs passed, failed, errored, and skip, for both same_order and random_order executions.

To accumulate the results across iterations (group by project and test-case), use:
```bash
poetry run results_parser PassedFailed load passed_failed_sample.csv to_test_overview to_csv --index=False > test_overview_sample.csv
```


## About the input

Prepare a CSV file with the following columns (example: `sample_input.csv`):
```
PROJECT_NAME,PROJECT_URL,PROJECT_HASH,PYPI_TAG,FUNCS_TO_TRACE,TESTS_TO_BE_RUN,NUM_RUNS
```

Every line in the input file will result in one execution of the container. We call this an 'Iteration'.
You can have duplicate lines in this input file to analyze the same project multiple times.
In fact, we actively use this to detect infrastructure flakiness, which might occur only between iterations, not within.
PROJECT_NAME, PROJECT_URL and PROJECT_HASH will be used to uniquely identify a project when accumulating results across multiple iterations.
PYPI_TAG is used to install the project itself via pip before executing its testsuite to fetch it's dependencies.
If PYPI_TAG is empty, FlaPy will fall back to searching for requirements in common files like requirements.txt


## Run on slurm cluster

Exporting the image:  
We do this, so we can deploy the image on our cluster without building it multiple times.

```bash
podman save -o flapy_image.tar localhost/flapy
```

Roll out image to slurm nodes (requires line-separated file `nodes` containing the names of all slurm nodes)
```bash
./exec_on_slurm_nodes.sh nodes load_podman_image.sh flapy.tar
```

Run on cluster
```bash
           # RUN_ON  CSV_FILE         PLUS_RANDOM_RUNS  ADDITIONAL_OPTIONS
./run_csv.sh cluster sample_input.csv true              ""
```


## TODOs

- [ ] Make output deterministic ^^
    * Many columns in passed_failed.csv are sets and their ordering is different from run to run


## Contributors

See the contributors list

## Contact

If you want to contact me, please find our contact details on my
[page at the University of Passau](https://www.fim.uni-passau.de/lehrstuhl-fuer-software-engineering-ii/lehrstuhlteam/).

## License

This project is licensed under the terms of the GNU Lesser General Public License.

