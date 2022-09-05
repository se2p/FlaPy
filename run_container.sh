#!/bin/bash
#SBATCH --job-name=flapy
#SBATCH --time=24:00:00
#SBATCH --mem=8GB

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

# -- PREPARE ENVIRONMENT
unset XDG_RUNTIME_DIR
unset XDG_CONFIG_HOME
# create podman folders
mkdir -p "${PODMAN_HOME}"
mkdir -p "${LOCAL_PODMAN_ROOT}"
# change home (your home dir doesn't get mounted on the cluster)
export HOME=$PODMAN_HOME
alias p='podman --root=$LOCAL_PODMAN_ROOT'

# -- INITIALIZE META FILE
META_FILE="$ITERATION_RESULTS_DIR/flapy-iteration-result.yaml"
touch "$META_FILE"

# -- LOG HOSTNAME
echo "hostname_run_container: $(cat /etc/hostname)"     >> "$META_FILE"

# -- EXECUTE CONTAINER
if [[ $PROJECT_URL == http* ]]
then
    podman --root "${LOCAL_PODMAN_ROOT}" run --rm \
        -v "$ITERATION_RESULTS_DIR:/results" \
        localhost/flapy \
        "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${PLUS_RANDOM_RUNS}" "${FLAPY_ARGS}"
else
    podman --root "${LOCAL_PODMAN_ROOT}" run --rm \
        -v "$ITERATION_RESULTS_DIR:/results" \
        -v "$PROJECT_URL":"$PROJECT_URL" \
        localhost/flapy \
        "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${PLUS_RANDOM_RUNS}" "${FLAPY_ARGS}"
fi

