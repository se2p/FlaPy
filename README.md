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
#              [OPTIONS...]                                 [INPUT_CSV]
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

### Prerequisites

Before you begin, ensure you have met the following requirements:
- Python in at least version 3.8.
- You have installed the latest version of [`poetry`](https://python-poetry.org).
    - `pip install poetry`

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





## Run on Slurm cluster

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


### Clean Slurm nodes

Remove all docker/podman images and containers via
```bash
./exec_on_slurm_nodes.sh nodes clean_podman.sh
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

