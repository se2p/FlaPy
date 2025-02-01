# FlaPy

FlaPy is a small tool that allows software developers and researchers to identify flaky tests within a given set of projects by rerunning their test suites.

It is the result of research carried out at the
[Chair of Software Engineering II](https://www.fim.uni-passau.de/lehrstuhl-fuer-software-engineering-ii/)
at the [University of Passau](https://www.uni-passau.de), Germany.


### Video tutorial

[![Conference presentation @ ICSE'23 DEMO](https://github.com/se2p/FlaPy/blob/master/images/flapy_demo_video_screenshot.png)](https://youtu.be/ejy-be-FvDY)


### Conference presentation (ICSE'23 DEMO)

[![Conference presentation @ ICSE'23 DEMO](https://github.com/se2p/FlaPy/blob/master/images/flapy_icse_2023_demo_screenshot.png)](https://youtu.be/JUMCW6zZpxc?feature=shared&t=2774)


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
PROJECT_NAME,PROJECT_URL,PROJECT_HASH,PYPI_TAG,FUNCS_TO_TRACE,TESTS_TO_BE_RUN
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
./flapy.sh run --out-dir example_results --plus-random-runs flapy_input_example.csv 5
```

Example (takes ~30s):
```bash
./flapy.sh run --out-dir example_results flapy_input_example_tiny.csv 1
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
  --path example_results \
  get_tests_overview _df \
  to_csv --index=False example_results_to.csv
```
Note: the directory specified after `--path` needs to be accessible from the current working directory since only the current working directory is mounted to the container that is started in the background!!


### Tracing

FlaPy offers an option to trace the execution of a function, i.e., to log all function and method calls made in the course of its execution.
The functions that shall be traced must be specified as a space separated list in the fifth column of the input-csv.
For example `test_flaky.py::test_network_remote_connection_failure test_flaky.py::test_concurrency` in [flapy_input_example_tiny_trace.csv](flapy_input_example_tiny_trace.csv).

Example (takes ~30s):
```bash
./flapy.sh run --out-dir example_results flapy_input_example_tiny_trace.csv
```

Within the resulting results.tar.xz archive, we can now find two extra files:
```
workdir/sameOrder/tmp/flapy_example_trace0test_flaky.py._('test_flaky.py', 'test_concurrency').txt
workdir/sameOrder/tmp/flapy_example_trace0test_flaky.py._('test_flaky.py', 'test_network_remote_connection_failure').txt
```
containing the traces:
```
--> ('test_flaky', '', 'test_network_remote_connection_failure')
----> ('requests.api', '', 'get')
------> ('requests.api', '', 'request')
--------> ('requests.sessions', 'Session', '__init__')
----------> ('requests.utils', '', 'default_headers')
------------> ('requests.utils', '', 'default_user_agent')
<------------ ('requests.utils', '', 'default_user_agent')
------------> ('requests.structures', 'CaseInsensitiveDict', '__init__')
...
```


## SFFL

(Spectrum-based Flaky Fault Localization)

From our paper [Debugging Flaky Tests using Spectrum-based Fault Localization](https://arxiv.org/abs/2305.04735)

### 1. Run tests multiple times while collecting line coverage

Execute `flapy.sh run` with core arguments `--collect-sqlite-coverage-database`

```
./flapy.sh run \
    --out-dir example_results_sffl \
    --core-args "--collect-sqlite-coverage-database" \
    flapy_input_example_sffl.csv 10
```


### 2. Perform fault localization

Execute `flapy.sh parse` to generate the CTA (coverage table accumulated)

(this step only produces an output, if the test actual showed flaky behavior -> if needed, rerun the previous step)
```
./flapy.sh parse \
    ResultsDirCollection --path example_results_sffl \
    save_cta_tables \
        --cta_save_dir example_results_sffl_cta \
        --flaky_col "Flaky_sameOrder_withinIteration" \
        --method="accum"
```

Calculate Suspiciousness scores

```
./flapy.sh parse \
    CtaDir --path example_results_sffl_cta \
    calc_and_save_suspiciousness_tables \
        --save_dir example_results_sffl_cta_sus \
        --sfl_method sffl
```

### 3. Evaluate results

Merge with locations (-> EXAM scores & ranks)

```
./flapy.sh parse \
    SuspiciousnessDir --path example_results_sffl_cta_sus \
    merge_location_info \
        minimal_sffl_example/locations.csv \
        minimal_sffl_example/loc.csv \
    to_csv --index=False | vd --filetype=csv
```
(assumes visidata (vd) to be installed)


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
This image can be used together with all existing scripts by setting the `FLAPY_DOCKER_IMAGE` environment variable.


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

