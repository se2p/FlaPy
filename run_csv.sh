#!/usr/bin/env bash

# -- DOC
# This scripts require LOCAL_PODMAN_ROOT to be set

# -- HELPER FUNCTIONS
DEBUG=1
function debug_echo {
  [[ "${DEBUG}" = 1 ]] && echo "$@"
}

# -- PARSE ARGUMENTS
RUN_ON=$1
CSV_FILE=$2
PLUS_RANDOM_RUNS=$3
FLAPY_ARGS=$4
RESULTS_PARENT_FOLDER=$5

# -- DEBUG OUTPUT
debug_echo "-- $0"
debug_echo "    Run on:            $RUN_ON"
debug_echo "    CSV file:          $CSV_FILE"
debug_echo "    Plus random runs:  $PLUS_RANDOM_RUNS"
debug_echo "    Flapy args:        $FLAPY_ARGS"


# -- INPUT PRE-PROCESSING
if [ -z "${RESULTS_PARENT_FOLDER}" ]; then
    RESULTS_PARENT_FOLDER=$(pwd)
fi
dos2unix "${CSV_FILE}"




# -- CREATE RESULTS_DIR
DATE_TIME=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_PARENT_FOLDER}/flapy-results_${DATE_TIME}"
mkdir -p "${RESULTS_DIR}"

# -- SAVE INPUT FILE
FLAPY_INPUT_META_FOLDER="$RESULTS_DIR/!flapy.input/"
mkdir "${FLAPY_INPUT_META_FOLDER}"
cp "${CSV_FILE}" "${FLAPY_INPUT_META_FOLDER}"

# -- LOG META INFOS
FLAPY_INPUT_META_FILE="$FLAPY_INPUT_META_FOLDER/flapy_input.yaml"
touch "$FLAPY_INPUT_META_FILE"
{
    echo "run_on:             \"$RUN_ON\""
    echo "csv_file:           \"$CSV_FILE\""
    echo "plus_random_runs:   \"$PLUS_RANDOM_RUNS\""
    echo "flapy_args:         \"$FLAPY_ARGS\""
} >> "$FLAPY_INPUT_META_FILE"


# -- CREATE LOG DIR
if [[ $RUN_ON = "cluster" ]]
then
    SLURM_LOG_DIR="${RESULTS_PARENT_FOLDER}/slurm-logs/${DATE_TIME}_run_csv"
    mkdir -p "$SLURM_LOG_DIR"
fi

# -- MAIN
{
    read # ignore header
    while IFS=, read -r PROJECT_NAME PROJECT_URL PROJECT_HASH PYPI_TAG FUNCS_TO_TRACE TESTS_TO_BE_RUN NUM_RUNS; do

        debug_echo "-- $0 (one row)"
        debug_echo "    Project name:      $PROJECT_NAME"
        debug_echo "    Project url:       $PROJECT_URL"
        debug_echo "    Project hash:      $PROJECT_HASH"
        debug_echo "    PyPi tag:          $PYPI_TAG"
        debug_echo "    Funcs to trace:    $FUNCS_TO_TRACE"
        debug_echo "    Tests to be run:   $TESTS_TO_BE_RUN"
        debug_echo "    Num runs:          $NUM_RUNS"

        # Although we have the DATE_TIME already in the RESULTS_DIR, we need it here,
        # because the iterations-result-dirs are sometimes sym-linked to other result-dirs
        ITERATION_RESULTS_DIR=$(
            mktemp -d "${RESULTS_DIR}/${PROJECT_NAME}_${DATE_TIME}__XXXXX"
        )
        ITERATION_NAME=$(basename ${ITERATION_RESULTS_DIR})

        if [[ $RUN_ON = "cluster" ]]
        then
            export PODMAN_HOME=
            export LOCAL_PODMAN_ROOT=
            sbatch \
                -o "$ITERATION_RESULTS_DIR/log.out" \
                -e "$ITERATION_RESULTS_DIR/log.out" \
                -- \
                run_container.sh \
                    "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${PLUS_RANDOM_RUNS}" "${ITERATION_RESULTS_DIR}" "${FLAPY_ARGS}"
        elif [[ $RUN_ON = "local" ]]
        then
            ./run_container.sh \
                "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${PLUS_RANDOM_RUNS}" "${ITERATION_RESULTS_DIR}" "${FLAPY_ARGS}"
        else
            echo "Unknown value '$RUN_ON' for RUN_ON. Please use 'cluster' or 'local'."
            exit
        fi

    done
} <"${CSV_FILE}"
