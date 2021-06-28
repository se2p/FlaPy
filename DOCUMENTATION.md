# run_execution.sh

`run_execution.sh` builds up the following structure for a Project "PROJ"

```bash
/scratch/${USER}                # SCRATCH
    /flapy                      # SCRATCH_ANALYSIS_DIR
    /flapy-results              # SCRATCH_RESULTS_DIR

/local/hdd/${USER}              # LOCAL
    /PROJ
        /ASDF1234                   # LOCAL_PROJECT_DIR (new HOME)
            /PROJ                   # REPOSITORY_DIR, --repository (here the project is cloned into by run_execution.sh)
            /venv                   #    virtualenv
       *    /execution.log          #    contains execution time, created by run_execution.sh
       *    /deterministic
                /execution.log          # --logfile
                /output.txt             # --output
                /tmp                    # --temp
                    /tmp1234            #    copy of REPOSITORY_DIR, created and deleted by analysis.py
                    /...output.xml      #
                    /...coverage.xml    #
                    /...output.log      #    pytest output (via runexec --output)
       *    /non-deterministic
                /execution.log          # --logfile
                /output.txt             # --output
                /tmp                    # --temp
                    /tmp1234            #    copy of REPOSITORY_DIR, created and deleted by analysis.py
                    /...output.xml      #
                    /...coverage.xml    #
                    /...output.log      #    pytest output (via runexec --output)


* = packaged in an archive called like "PROJ_XXXXX" (project name + random postfix) and copied over to SCRATCH_RESULTS_DIR
```
