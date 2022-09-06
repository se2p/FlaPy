#!/usr/bin/env bash
#SBATCH --job-name=flapy
#SBATCH --time=24:00:00
#SBATCH --mem=8GB

# -- CHECK IF ENVIRONMENT VARIABLES ARE DEFINED
if [[ -z "${FLAPY_INPUT_CSV_FILE}" ]]; then
    echo "ERROR: FLAPY_INPUT_CSV_FILE not defined"
    exit 1
fi
if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
    LINE_NUM=$SLURM_ARRAY_TASK_ID
elif [[ -n "${FLAPY_INPUT_CSV_LINE_NUM}" ]]; then
    LINE_NUM=$FLAPY_INPUT_CSV_LINE_NUM
else
    echo "ERROR: either SLURM_ARRAY_TASK_ID or FLAPY_INPUT_CSV_LINE_NUM must be defined"
    exit 1
fi

echo "-- $0"
echo "    input csv file:      $FLAPY_INPUT_CSV_FILE"
echo "    slurm array task id: $SLURM_ARRAY_TASK_ID"
echo "    input csv line:      $FLAPY_INPUT_CSV_LINE_NUM"

function sighdl {
  kill -INT "${srunPid}" || true
}

# -- READ CSV LINE
csv_line=$(sed "${LINE_NUM}q;d" "${FLAPY_INPUT_CSV_FILE}")

# -- PARSE CSV LINE
IFS=, read -r PROJECT_NAME PROJECT_URL PROJECT_HASH PYPI_TAG FUNCS_TO_TRACE TESTS_TO_BE_RUN NUM_RUNS <<< "${csv_line}"

# -- DEBUG OUTPUT
echo "    ----"
echo "    Project name:      $PROJECT_NAME"
echo "    Project url:       $PROJECT_URL"
echo "    Project hash:      $PROJECT_HASH"
echo "    PyPi tag:          $PYPI_TAG"
echo "    Funcs to trace:    $FUNCS_TO_TRACE"
echo "    Tests to be run:   $TESTS_TO_BE_RUN"
echo "    Num runs:          $NUM_RUNS"

# -- CREATE ITERATION RESULTS DIRECTORY
#     Although we have the DATE_TIME already in the RESULTS_DIR, we need it here,
#     because the iterations-result-dirs are sometimes sym-linked to other result-dirs
ITERATION_RESULTS_DIR=$(
    mktemp -d "${FLAPY_RESULTS_DIR}/${PROJECT_NAME}_${FLAPY_DATE_TIME}__XXXXX"
)
ITERATION_NAME=$(basename ${ITERATION_RESULTS_DIR})

# -- INITIALIZE META FILE
META_FILE="$ITERATION_RESULTS_DIR/flapy-iteration-result.yaml"
touch "$META_FILE"

# -- LOG
echo "input_csv_line_num:     ${LINE_NUM}"   >> "$META_FILE"
echo "hostname_run_line:      $(cat /etc/hostname)"     >> "$META_FILE"

# -- RUN CONTAINER
if [[ $FLAPY_INPUT_RUN_ON = "cluster" ]]; then
    srun \
        --output="$ITERATION_RESULTS_DIR/log.out" \
        --error="$ITERATION_RESULTS_DIR/log.out" \
        -- \
        run_container.sh \
            "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${FLAPY_INPUT_PLUS_RANDOM_RUNS}" "${ITERATION_RESULTS_DIR}" "${FLAPY_INPUT_OTHER_ARGS}" \
        & srunPid=$!
elif [[ $FLAPY_INPUT_RUN_ON = "local" ]]; then
    ./run_container.sh \
        "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${FLAPY_INPUT_PLUS_RANDOM_RUNS}" "${ITERATION_RESULTS_DIR}" "${FLAPY_INPUT_OTHER_ARGS}" \
    & srunPid=$!
else
    echo "Unknown value '$RUN_ON' for RUN_ON. Please use 'cluster' or 'local'."
    exit
fi

trap sighdl INT TERM HUP QUIT

while ! wait; do true; done

