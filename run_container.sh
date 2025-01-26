#!/bin/bash

source utils.sh

# -- PARSE ARGS
DOCKER_FLAGS=$1
PROJECT_NAME=$2
PROJECT_URL=$3
PROJECT_HASH=$4
PYPI_TAG=$5
FUNCS_TO_TRACE=$6
TESTS_TO_BE_RUN=$7
NUM_RUNS=$8
PLUS_RANDOM_RUNS=$9
ITERATION_RESULTS_DIR=${10}
FLAPY_ARGS=${11}

# -- DEBUG OUTPUT
debug_echo "-- $0 (run_container.sh)"
debug_echo "    Project name:         $PROJECT_NAME"
debug_echo "    Project url:          $PROJECT_URL"
debug_echo "    Project hash:         $PROJECT_HASH"
debug_echo "    PyPi tag:             $PYPI_TAG"
debug_echo "    Funcs to trace:       $FUNCS_TO_TRACE"
debug_echo "    Tests to be run:      $TESTS_TO_BE_RUN"
debug_echo "    Num runs:             $NUM_RUNS"
debug_echo "    Plus random runs:     $PLUS_RANDOM_RUNS"
debug_echo "    Iteration results:    $ITERATION_RESULTS_DIR"
debug_echo "    Flapy Args:           $FLAPY_ARGS"

# -- SET UP ENVIRONMENT (define flapy_docker_command)
debug_echo "-- Prepare for docker command"
source prepare_for_docker_command.sh || exit

# -- PULL IMAGE
debug_echo "-- Pulling FlaPY docker image"
./pull_flapy_docker_image.sh

# -- INITIALIZE META FILE
META_FILE="$ITERATION_RESULTS_DIR/flapy-iteration-result.yaml"

# -- LOG META INFO
debug_echo "-- Logging Meta info"
FLAPY_IMAGE_ID=$(flapy_docker_command images $FLAPY_DOCKER_IMAGE --format "{{.ID}}" | head -n 1)
echo "hostname_run_container: $(cat /etc/hostname)"  >> "$META_FILE"
echo "flapy_image_name:       ${FLAPY_DOCKER_IMAGE}" >> "$META_FILE"
echo "flapy_image_id:         ${FLAPY_IMAGE_ID}"     >> "$META_FILE"

# -- EXECUTE CONTAINER
debug_echo "-- Running container"
if [[ $PROJECT_URL == http* ]]
then
    flapy_docker_command run --log-driver=none --rm $DOCKER_FLAGS \
        -v "$ITERATION_RESULTS_DIR:/results" \
        $FLAPY_DOCKER_IMAGE \
        "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${PLUS_RANDOM_RUNS}" "${FLAPY_ARGS}"
else
    PROJECT_URL_ABS_PATH=$(realpath "$PROJECT_URL")
    flapy_docker_command run --log-driver=none --rm $DOCKER_FLAGS \
        -v "$ITERATION_RESULTS_DIR:/results" \
        -v "$PROJECT_URL_ABS_PATH":/project_sources \
        $FLAPY_DOCKER_IMAGE \
        "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${PLUS_RANDOM_RUNS}" "${FLAPY_ARGS}"
fi

