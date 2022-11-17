#!/bin/bash

# -- PARSE ARGS
PROJECT_NAME=$1
PROJECT_URL=$2
PROJECT_HASH=$3
PYPI_TAG=$4
FUNCS_TO_TRACE=$5
TESTS_TO_BE_RUN=$6
NUM_RUNS=$7
PLUS_RANDOM_RUNS=$8
ITERATION_RESULTS_DIR=$9
FLAPY_ARGS=${10}

# -- DEBUG OUTPUT
echo "-- $0 (run_container.sh)"
echo "    Project name:         $PROJECT_NAME"
echo "    Project url:          $PROJECT_URL"
echo "    Project hash:         $PROJECT_HASH"
echo "    PyPi tag:             $PYPI_TAG"
echo "    Funcs to trace:       $FUNCS_TO_TRACE"
echo "    Tests to be run:      $TESTS_TO_BE_RUN"
echo "    Num runs:             $NUM_RUNS"
echo "    Plus random runs:     $PLUS_RANDOM_RUNS"
echo "    Iteration results:    $ITERATION_RESULTS_DIR"
echo "    Flapy Args:           $FLAPY_ARGS"

# -- SET UP ENVIRONMENT (define flapy_docker_command)
echo "-- Prepare for docker command"
source prepare_for_docker_command.sh || exit

# -- INITIALIZE META FILE
META_FILE="$ITERATION_RESULTS_DIR/flapy-iteration-result.yaml"

# -- LOG META INFO
echo "-- Logging Meta info"
flapy_container_id=$(flapy_docker_command images $FLAPY_DOCKER_IMAGE --format "{{.ID}}")
echo "hostname_run_container: $(cat /etc/hostname)"     >> "$META_FILE"
echo "flapy_container_id:     ${flapy_container_id}"    >> "$META_FILE"

# -- EXECUTE CONTAINER
echo "-- Running container"
if [[ $PROJECT_URL == http* ]]
then
    flapy_docker_command run --rm \
        -v "$ITERATION_RESULTS_DIR:/results" \
        $FLAPY_DOCKER_IMAGE \
        "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${PLUS_RANDOM_RUNS}" "${FLAPY_ARGS}"
else
    flapy_docker_command run --rm \
        -v "$ITERATION_RESULTS_DIR:/results" \
        -v "$PROJECT_URL":"$PROJECT_URL" \
        $FLAPY_DOCKER_IMAGE \
        "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${PLUS_RANDOM_RUNS}" "${FLAPY_ARGS}"
fi

