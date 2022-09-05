# run_container.sh

Run the image:

```bash
mkdir -p ~/flapy-results-tryout/sample_123

podman run \
    -it \
    -v ~/flapy-results-tryout/sample_123:/results localhost/flapy \
  # PROJECT_NAME   PROJECT_URL                                                 PROJECT_HASH  FUNCS_TO_TRACE  TESTS_TO_BE_RUN  NUM_RUNS  PLUS_RANDOM_RUNS
    sample_project "https://github.com/gruberma/python-tests-minimal-examples" 4865a6a       ""              ""               2         true
```

Parameters:
* PROJECT_NAME: name of the project you want to analyze (can be anything, just for convenience)
* PROJECT_URL: URL leading to your project. Will be used to clone the project via git.
* PROJECT_HASH: the git hash of the commit you want to analyze.
* FUNCS_TO_TRACE: deprecated, leave empty
* TESTS_TO_BE_RUN: deprecated, leave empty
* NUM_RUNS: number of times you would like to execute the test suite of the project
* PLUS_RANDOM_RUNS: would you like to additionally executed the test suite in randomly shuffled order? (also NUM_RUNS times) value: true / false


# clone_and_run_tests.sh

`clone_and_run_tests.sh` builds up the following structure for a Project "PROJ"

```bash
/workdir                           # docker WORKDIR
    /clone_and_run_tests.sh        # copied over by docker
    /FlaPy-0.1.0-py3-none-any.whl  # copied over by docker
    /PROJ                          # REPOSITORY_DIR (cloned repo)
*   /deterministic                 # same order execution output
       /execution.log              #   --logfile
       /output.txt                 #   --output
       /tmp                        #   --temp
           /tmp1234                #   copy of REPOSITORY_DIR, created and deleted by analysis.py
           /...output.xml          #
           /...coverage.xml        #
           /...output.log          #    pytest output (via runexec --output)
*   /non-deterministic             # random order execution outpout
       /execution.log              #   --logfile
       /output.txt                 #   --output
       /tmp                        #   --temp
           /tmp1234                #    copy of REPOSITORY_DIR, created and deleted by analysis.py
           /...output.xml          #
           /...coverage.xml        #
           /...output.log          #    pytest output (via runexec --output)


# * = packaged in an archive called "results.tar.xz" and copied over to ITERATION_RESULTS_DIR
```

# Structure of new scripts

`run_csv.sh`
* for line in csv: (or via array_job)
    `run_line.sh`
        `run_container.sh` (local or on cluster)
        * podman run flapy
            `clone_and_run_tests.sh`
            * clone repo
            * log further infos
            * run `run_tests.py`

