FROM registry.hub.docker.com/library/python:3.10.1-bullseye

RUN apt-get update && apt-get install -y sqlite3 cloc

WORKDIR /workdir

# Input: like run_execution_container.sh
# Ouput: write results to mounted dir

# TODO build flapy in this container (multi-stage dockerfile)

# -- INSTALL FLAPY
COPY clone_and_run_tests.sh .
COPY dist/FlaPy-0.2.0-py3-none-any.whl .
RUN ["pip", "install", "FlaPy-0.2.0-py3-none-any.whl"]

ENTRYPOINT ["./clone_and_run_tests.sh"]
