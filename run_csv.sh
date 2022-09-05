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
debug_echo "    ----"


# -- INPUT PRE-PROCESSING
dos2unix "${CSV_FILE}"
CSV_FILE_LENGTH=$(wc -l < "$CSV_FILE")
debug_echo "    CSV file length:   $CSV_FILE_LENGTH"

# -- CREATE RESULTS_DIR
if [ -z "${RESULTS_PARENT_FOLDER}" ]; then
    RESULTS_PARENT_FOLDER=$(pwd)
fi
DATE_TIME=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_PARENT_FOLDER}/flapy-results_${DATE_TIME}"
mkdir -p "${RESULTS_DIR}"

# -- SAVE INPUT FILE
FLAPY_INPUT_META_FOLDER="$RESULTS_DIR/!flapy.input/"
mkdir "${FLAPY_INPUT_META_FOLDER}"
cp "${CSV_FILE}" "${FLAPY_INPUT_META_FOLDER}/input.csv"

# -- LOG META INFOS
FLAPY_INPUT_META_FILE="$FLAPY_INPUT_META_FOLDER/flapy_input.yaml"
touch "$FLAPY_INPUT_META_FILE"
{
    echo "run_on:             \"$RUN_ON\""
    echo "csv_file:           \"$CSV_FILE\""
    echo "plus_random_runs:   \"$PLUS_RANDOM_RUNS\""
    echo "flapy_args:         \"$FLAPY_ARGS\""
} >> "$FLAPY_INPUT_META_FILE"


# -- EXPORT VARIALBE
#     these variables will be picked up by run_line.sh
export FLAPY_INPUT_CSV_FILE="${FLAPY_INPUT_META_FOLDER}/input.csv"
export FLAPY_INPUT_PLUS_RANDOM_RUNS=$PLUS_RANDOM_RUNS
export FLAPY_INPUT_OTHER_ARGS=$FLAPY_ARGS
export FLAPY_INPUT_RUN_ON=$RUN_ON
export FLAPY_DATE_TIME=$DATE_TIME
export FLAPY_RESULTS_DIR=$RESULTS_DIR

# -- RUN
if [[ $RUN_ON = "cluster" ]]
then
    echo "running on cluster"
    # export PODMAN_HOME=
    # export LOCAL_PODMAN_ROOT=
    sbatch \
        --constraint="" \
        --output "$FLAPY_INPUT_META_FOLDER/log-%a.out" \
        --error  "$FLAPY_INPUT_META_FOLDER/log-%a.out" \
        --array=2-"$CSV_FILE_LENGTH" \
        -- \
        run_line.sh
elif [[ $RUN_ON = "local" ]]
then
    for i in $(seq 2 "$CSV_FILE_LENGTH"); do
        FLAPY_INPUT_CSV_LINE_NUM=$i ./run_line.sh
    done
else
    echo "Unknown value '$RUN_ON' for RUN_ON. Please use 'cluster' or 'local'."
    exit
fi

