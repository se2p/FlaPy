# clone_and_run_tests.sh

`clone_and_run_tests.sh` builds up the following structure for a Project "PROJ"

```bash
/workdir                           # docker WORKDIR
    /clone_and_run_tests.sh        # copied over by docker
    /FlaPy-0.1.0-py3-none-any.whl  # copied over by docker
    /PROJ                          # REPOSITORY_DIR (cloned repo)
*   /sameOrder                     # same order execution output
       /execution.log              #   --logfile
       /tmp                        #   --temp
           /flapy_repo_copy        #   copy of REPOSITORY_DIR, created and deleted by run_tests.py
           /...output.xml          #
           /...coverage.xml        #
*   /randomOrder                   # random order execution outpout
       /execution.log              #   --logfile
       /tmp                        #   --temp
           /flapy_repo_copy        #    copy of REPOSITORY_DIR, created and deleted by run_tests.py
           /...output.xml          #
           /...coverage.xml        #


# * = packaged in an archive called "results.tar.xz" and copied over to ITERATION_RESULTS_DIR
```

# Structure of scripts

```bash
flapy.sh run
    run_csv.sh  # (run_on, constraint, input_csv, plus_random_runs, num_runs, core_args, out_dir)
    # for line in csv (or via array_job):
        run_line.sh
            run_container.sh
            # podman run flapy
                clone_and_run_tests.sh
                    # clone repo
                    # log further infos
                    run_tests.py
```


```bash
flapy.sh parse
    results_parser.sh
    # run flapy docker image with mounted CWD
```
