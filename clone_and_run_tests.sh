#!/bin/bash
set -e

# -- HELPER FUNCTIONS
function debug_echo {
  [[ "${DEBUG}" = 1 ]] && echo "$@"
}

# -- CONSTANTS
DEBUG=1
RESULT_DIR="/results"

# -- PARSE ARGS
PROJECT_NAME=$1
PROJECT_URL=$2
PROJECT_HASH=$3
PYPI_TAG=$4
FUNCS_TO_TRACE=$5
TESTS_TO_BE_RUN=$6
NUM_RUNS=$7
PLUS_RANDOM_RUNS=$8
FLAPY_ARGS=$9

# -- DEBUG OUTPUT
echo "-- $0"
debug_echo "    Project name:         $PROJECT_NAME"
debug_echo "    Project url:          $PROJECT_URL"
debug_echo "    Project hash:         $PROJECT_HASH"
debug_echo "    PyPi tag:             $PYPI_TAG"
debug_echo "    Tests to be run:      $TESTS_TO_BE_RUN"
debug_echo "    Funcs to trace:       $FUNCS_TO_TRACE"
debug_echo "    Num runs:             $NUM_RUNS"
debug_echo "    Plus random runs:     $PLUS_RANDOM_RUNS"
debug_echo "    Flapy Args:           $FLAPY_ARGS"

# -- INITIALIZE META FILE
META_FILE="$RESULT_DIR/flapy-iteration-result.yaml"
touch "$META_FILE"

CWD=$(pwd)
debug_echo " CWD = ${CWD}"

# -- CLONE / COPY REPO
REPOSITORY_DIR="${CWD}/${PROJECT_NAME}"
debug_echo "Clone Repository into ${REPOSITORY_DIR}"
if [[ $PROJECT_URL == http* ]]
then
    git clone "${PROJECT_URL}" "${REPOSITORY_DIR}"
    if [[ -n "$PROJECT_HASH" ]]; then
        cd "${REPOSITORY_DIR}" || exit 1
        git reset --hard "${PROJECT_HASH}" || exit 1
        cd "${CWD}" || exit 1
    fi
    REPO_HASH=$(git --git-dir="${REPOSITORY_DIR}/.git" rev-parse HEAD)
else
    cp -r "${PROJECT_URL}" "${REPOSITORY_DIR}"
fi


# -- LOG META INFOS (1/2)
START_TIMESTAMP=$(date +%s)
START_DATE=$(date)
{
    echo "project_name:           \"$PROJECT_NAME\""
    echo "project_url:            \"$PROJECT_URL\""
    echo "project_git_hash:       \"$REPO_HASH\""
    echo "project_git_hash_INPUT: \"$PROJECT_HASH\""
    echo "pypi_tag:               \"$PYPI_TAG\""
    echo "func_to_trace:          \"$FUNCS_TO_TRACE\""
    echo "tests_to_be_run:        \"$TESTS_TO_BE_RUN\""
    echo "num_runs:               $NUM_RUNS"
    echo "plus_random_runs:       $PLUS_RANDOM_RUNS"
    echo "flapy args:             \"$FLAPY_ARGS\""
    echo "start_time:             $START_DATE"
} >> $META_FILE

# -- EXECUTE TESTS
#  In the following two paragraphs, "non-deterministic" means executing the project's tests in random
#  order and "deterministic" means executing them in same order (default of pytest).
#  This naming convention is not good, but I keep it for legacy reasons.
if [[ $PLUS_RANDOM_RUNS = true ]]
then
    debug_echo "Run tests in random order mode"
    mkdir -p "${CWD}/non-deterministic"
    mkdir -p "${CWD}/non-deterministic/tmp"
    flapy_run_tests \
      --logfile "${CWD}/non-deterministic/execution.log" \
      --repository "${REPOSITORY_DIR}" \
      --project-name "${PROJECT_NAME}" \
      --pypi-tag "${PYPI_TAG}" \
      --temp "${CWD}/non-deterministic/tmp" \
      --number-test-runs "${NUM_RUNS}" \
      --random-order-bucket global \
      --trace "${FUNCS_TO_TRACE}" \
      --tests-to-be-run "${TESTS_TO_BE_RUN}" \
      $FLAPY_ARGS  # do not double quote here! we want the word splitting
fi

debug_echo "Run tests in same order mode"
mkdir -p "${CWD}/deterministic"
mkdir -p "${CWD}/deterministic/tmp"
flapy_run_tests \
  --logfile "${CWD}/deterministic/execution.log" \
  --repository "${REPOSITORY_DIR}" \
  --project-name "${PROJECT_NAME}" \
  --pypi-tag "${PYPI_TAG}" \
  --temp "${CWD}/deterministic/tmp" \
  --number-test-runs "${NUM_RUNS}" \
  --trace "${FUNCS_TO_TRACE}" \
  --tests-to-be-run "${TESTS_TO_BE_RUN}" \
  $FLAPY_ARGS  # do not double quote here! we want the word splitting

# -- LOG META INFOS (2/2)
END_DATE=$(date)
END_TIMESTAMP=$(date +%s)
RUNTIME=$((END_TIMESTAMP-START_TIMESTAMP))
echo "end_time:               $END_DATE" >> $META_FILE
echo "execution_time:         $RUNTIME"  >> $META_FILE

# -- COPY RESULTS BACK
if [[ $PLUS_RANDOM_RUNS = true ]]
then
    debug_echo "Copy results back (incl random runs)"
    tar cJf "${RESULT_DIR}/results.tar.xz" \
      "${CWD}/non-deterministic" \
      "${CWD}/deterministic"
else
    debug_echo "Copy results back (no random runs)"
    tar cJf "${RESULT_DIR}/results.tar.xz" \
      "${CWD}/deterministic"
fi

debug_echo "Done"
