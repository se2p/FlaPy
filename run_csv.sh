#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(dirname $0)

source "$SCRIPT_DIR/utils.sh"


# -- PARSE ARGUMENTS
RUN_ON=$1
CONSTRAINT=$2
INPUT_CSV=$3
PLUS_RANDOM_RUNS=$4
NUM_RUNS=$5
CORE_ARGS=$6
OUT_DIR=$7

# -- DEBUG OUTPUT
debug_echo "-- $0"
debug_echo "    Num runs:              $NUM_RUNS"
debug_echo "    Run on:                $RUN_ON"
debug_echo "    Constraint:            $CONSTRAINT"
debug_echo "    Input CSV:             $INPUT_CSV"
debug_echo "    Plus random runs:      $PLUS_RANDOM_RUNS"
debug_echo "    Core args:             $CORE_ARGS"
debug_echo "    Out-dir:               $OUT_DIR"
debug_echo "    ----"

# -- INPUT PRE-PROCESSING
INPUT_CSV_LENGTH=$(wc -l < "$INPUT_CSV")
debug_echo "    input-csv length:      $INPUT_CSV_LENGTH"

# -- CREATE RESULTS_DIR
if [ -z "${OUT_DIR}" ]; then
    OUT_DIR=$(pwd)
else
    OUT_DIR=$(realpath "$OUT_DIR")
fi
DATE_TIME=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${OUT_DIR}/flapy-results_${DATE_TIME}"
mkdir -p "${RESULTS_DIR}"

# -- SAVE INPUT FILE
FLAPY_META_FOLDER="$RESULTS_DIR/!flapy.run/"
mkdir "${FLAPY_META_FOLDER}"
cp "${INPUT_CSV}" "${FLAPY_META_FOLDER}/input.csv"

# -- LOG META INFOS
FLAPY_META_FILE="$FLAPY_META_FOLDER/flapy_run.yaml"
{
    echo "num_runs:               $NUM_RUNS"
    echo "run_on:                 \"$RUN_ON\""
    echo "constraint:             \"$CONSTRAINT\""
    echo "input_csv:              \"$INPUT_CSV\""
    echo "plus_random_runs:       \"$PLUS_RANDOM_RUNS\""
    echo "core_args:              \"$CORE_ARGS\""
    echo "out_dir:                \"$OUT_DIR\""
    echo "input_csv_length:       $INPUT_CSV_LENGTH"
} >> "$FLAPY_META_FILE"


# -- EXPORT VARIABLE
#     these variables will be picked up by run_line.sh
export FLAPY_INPUT_CSV_FILE="${FLAPY_META_FOLDER}/input.csv"
export FLAPY_INPUT_PLUS_RANDOM_RUNS=$PLUS_RANDOM_RUNS
export FLAPY_INPUT_NUM_RUNS=$NUM_RUNS
export FLAPY_INPUT_OTHER_ARGS=$CORE_ARGS
export FLAPY_INPUT_RUN_ON=$RUN_ON
export FLAPY_DATE_TIME=$DATE_TIME
export FLAPY_RESULTS_DIR=$RESULTS_DIR

# -- SBATCH LOG FILES
SBATCH_LOG_FOLDER="$FLAPY_META_FOLDER/sbatch_logs/"
mkdir -p "$SBATCH_LOG_FOLDER"
SBATCH_LOG_FILE_PATTERN="$SBATCH_LOG_FOLDER/log-%a.out"

# -- RUN
if [[ $RUN_ON = "cluster" ]]
then
    debug_echo "running on cluster"
    # export PODMAN_HOME=
    # export LOCAL_PODMAN_ROOT=
    sbatch_info=$(sbatch --parsable \
        --constraint="$CONSTRAINT" \
        --output "$SBATCH_LOG_FILE_PATTERN" \
        --error  "$SBATCH_LOG_FILE_PATTERN" \
        --array=2-"$INPUT_CSV_LENGTH" \
        -- \
        run_line.sh
    )
    debug_echo "sbatch_submission_info: $sbatch_info"
    echo "sbatch_submission_info: \"$sbatch_info\"" >> "$FLAPY_META_FILE"
elif [[ $RUN_ON = "locally" ]]
then
    for i in $(seq 2 "$INPUT_CSV_LENGTH"); do
        FLAPY_INPUT_CSV_LINE_NUM=$i "$SCRIPT_DIR/run_line.sh"
    done
else
    debug_echo "Unknown value '$RUN_ON' for RUN_ON. Please use 'cluster' or 'locally'."
    exit
fi

