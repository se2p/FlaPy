#!/usr/bin/env bash
#SBATCH --job-name=flapy
#SBATCH --time=24:00:00
#SBATCH --mem=8GB

source utils.sh

# -- CHECK IF ENVIRONMENT VARIABLES ARE DEFINED
if [[ -z "${FLAPY_INPUT_CSV_FILE}" ]]; then
    debug_echo "ERROR: FLAPY_INPUT_CSV_FILE not defined"
    exit 1
fi
if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
    LINE_NUM=$SLURM_ARRAY_TASK_ID
elif [[ -n "${FLAPY_INPUT_CSV_LINE_NUM}" ]]; then
    LINE_NUM=$FLAPY_INPUT_CSV_LINE_NUM
else
    debug_echo "ERROR: either SLURM_ARRAY_TASK_ID or FLAPY_INPUT_CSV_LINE_NUM must be defined"
    exit 1
fi

debug_echo "-- $0"
debug_echo "    input csv file:      $FLAPY_INPUT_CSV_FILE"
debug_echo "    slurm array task id: $SLURM_ARRAY_TASK_ID"
debug_echo "    input csv line num:  $FLAPY_INPUT_CSV_LINE_NUM"

function sighdl {
  kill -INT "${srunPid}" || true
}

# -- READ CSV LINE
csv_line=$(sed "${LINE_NUM}q;d" "${FLAPY_INPUT_CSV_FILE}")

# -- PARSE CSV LINE
IFS=, read -r PROJECT_NAME PROJECT_URL PROJECT_HASH PYPI_TAG FUNCS_TO_TRACE TESTS_TO_BE_RUN NUM_RUNS <<< "${csv_line}"

# -- DEBUG OUTPUT
debug_echo "    ----"
debug_echo "    Project name:      $PROJECT_NAME"
debug_echo "    Project url:       $PROJECT_URL"
debug_echo "    Project hash:      $PROJECT_HASH"
debug_echo "    PyPi tag:          $PYPI_TAG"
debug_echo "    Funcs to trace:    $FUNCS_TO_TRACE"
debug_echo "    Tests to be run:   $TESTS_TO_BE_RUN"
debug_echo "    Num runs:          $NUM_RUNS"

# -- CREATE ITERATION RESULTS DIRECTORY
#     Although we have the DATE_TIME already in the RESULTS_DIR, we need it here,
#     because the iterations-result-dirs are sometimes sym-linked to other result-dirs
ITERATION_RESULTS_DIR="${FLAPY_RESULTS_DIR}/${PROJECT_NAME}_${FLAPY_DATE_TIME}_${LINE_NUM}"
mkdir -p "${ITERATION_RESULTS_DIR}"

# -- INITIALIZE META FILE
META_FILE="$ITERATION_RESULTS_DIR/flapy-iteration-result.yaml"
touch "$META_FILE"

# -- LOG
{
    echo "slurm_array_task_id:    ${SLURM_ARRAY_TASK_ID}"
    echo "slurm_array_job_id:     ${SLURM_ARRAY_JOB_ID}"
    echo "slurm_job_id:           ${SLURM_JOB_ID}"
    echo "input_csv_line_num:     ${LINE_NUM}"
    echo "hostname_run_line:      $(cat /etc/hostname)"

    echo "project_name:           \"$PROJECT_NAME\""
    echo "project_url:            \"$PROJECT_URL\""
    echo "project_git_hash_INPUT: \"$PROJECT_HASH\""
    echo "pypi_tag:               \"$PYPI_TAG\""
    echo "func_to_trace:          \"$FUNCS_TO_TRACE\""
    echo "tests_to_be_run:        \"$TESTS_TO_BE_RUN\""
    echo "num_runs:               $NUM_RUNS"
    echo "plus_random_runs:       $FLAPY_INPUT_PLUS_RANDOM_RUNS"
    echo "flapy args:             \"$FLAPY_INPUT_OTHER_ARGS\""
} >> "$META_FILE"

# -- RUN CONTAINER
if [[ $FLAPY_INPUT_RUN_ON = "cluster" ]]; then
    srun \
        --output="$ITERATION_RESULTS_DIR/log.out" \
        --error="$ITERATION_RESULTS_DIR/log.out" \
        -- \
        run_container.sh \
            "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${FLAPY_INPUT_PLUS_RANDOM_RUNS}" "${ITERATION_RESULTS_DIR}" "${FLAPY_INPUT_OTHER_ARGS}" \
        & srunPid=$!
elif [[ $FLAPY_INPUT_RUN_ON = "locally" ]]; then
    ./run_container.sh \
        "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${PYPI_TAG}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${FLAPY_INPUT_PLUS_RANDOM_RUNS}" "${ITERATION_RESULTS_DIR}" "${FLAPY_INPUT_OTHER_ARGS}" \
    & srunPid=$!
else
    debug_echo "Unknown value '$RUN_ON' for RUN_ON. Please use 'cluster' or 'locally'."
    exit
fi

trap sighdl INT TERM HUP QUIT

while ! wait; do true; done

