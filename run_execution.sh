#!/bin/bash

DEBUG=1

function debug_echo {
  [[ "${DEBUG}" = 1 ]] && echo "$@"
}

function print_red {
  echo "\033[0;31m${1}\033[0m"
}

function help_message {
  echo "${0} -n|--project-name <project_name> -u|--project-url <project_url> -p|--python-path </path/to/python>"
}

PROJECT_NAME=$1
PROJECT_URL=$2
PROJECT_HASH=$3
FUNC_TO_TRACE=$4
TESTS_TO_BE_RUN=$5
NUM_RUNS=$6
LOCAL_PROJECT_DIR=$7
mkdir -p "${LOCAL_PROJECT_DIR}"
if [ -f /usr/bin/python3.7 ] ; then
    PYTHON_PATH="/usr/bin/python3.7"
else
    PYTHON_PATH="/usr/bin/python3"
fi

echo "Project name:      $PROJECT_NAME"
echo "Project url:       $PROJECT_URL"
echo "Project hash:      $PROJECT_HASH"
echo "Funcs to trace:    $FUNC_TO_TRACE"
echo "Num runs:          $NUM_RUNS"
echo "Local project dir: $LOCAL_PROJECT_DIR"


START_TIME=$(date +%s)

CWD=$(pwd)
SCRATCH_ANALYSIS_DIR=$(pwd)
SCRATCH_RESULTS_DIR="$(pwd)/flapy-results"
mkdir -p "${SCRATCH_RESULTS_DIR}"
RESULT_DIR=$(mktemp -d "${SCRATCH_RESULTS_DIR}/${PROJECT_NAME}__XXXXX")

debug_echo "Project ${PROJECT_NAME}"

# Why set pip cache?
# => You don't have a home-directory on the cluster,
#    but pip needs a caching directory
#
# In case `virtualenv` is not found, the altered HOME might be the reason for it
# solution: use global installation of virtualenv, not local one
export PIP_CACHE_DIR="/local/hdd/gruberma/.cache/pip"

debug_echo "Change directory to local"
cd "${LOCAL_PROJECT_DIR}" || exit 1

debug_echo "Create Virtual Environment"
python3 -m virtualenv -p "${PYTHON_PATH}" "${LOCAL_PROJECT_DIR}/venv"

source "${LOCAL_PROJECT_DIR}/venv/bin/activate"

debug_echo "Installing FlaPy"
pip install "${SCRATCH_ANALYSIS_DIR}/dist/FlaPy-0.1.0-py3-none-any.whl"
# Note: FlaPy creates another virtualenv inside this one
# Why use two virtualenvs inside each other?
# => The dependencies of FlaPy might clash with the dependencies of the analysed project
#   -- Stephan

debug_echo "Using $(python --version) in $(which python)"

REPOSITORY_DIR="${LOCAL_PROJECT_DIR}/${PROJECT_NAME}"
debug_echo "Checkout Repository into ${REPOSITORY_DIR}"
git clone "${PROJECT_URL}" "${REPOSITORY_DIR}"
cd "${REPOSITORY_DIR}" || exit 1
git reset --hard "${PROJECT_HASH}" || exit 1
cd "${LOCAL_PROJECT_DIR}" || exit 1

# Log further information
echo "$PROJECT_NAME" > "$RESULT_DIR/project-name.txt"
echo "$PROJECT_URL" > "$RESULT_DIR/project-url.txt"
echo "$FUNC_TO_TRACE" > "$RESULT_DIR/func_to_trace.txt"
echo "$TESTS_TO_BE_RUN" > "$RESULT_DIR/tests_to_be_run.txt"
echo "$NUM_RUNS" > "$RESULT_DIR/num_runs.txt"
git --git-dir="${REPOSITORY_DIR}/.git" rev-parse HEAD > "${RESULT_DIR}/project-git-hash.txt"
git --git-dir="${SCRATCH_ANALYSIS_DIR}/.git" rev-parse HEAD > "${RESULT_DIR}/flakyanalysis-git-hash.txt"

debug_echo "Run Analysis in non-deterministic mode"
mkdir -p "${LOCAL_PROJECT_DIR}/non-deterministic"
mkdir -p "${LOCAL_PROJECT_DIR}/non-deterministic/tmp"
flakyanalysis \
  --logfile "${LOCAL_PROJECT_DIR}/non-deterministic/execution.log" \
  --repository "${REPOSITORY_DIR}" \
  --temp "${LOCAL_PROJECT_DIR}/non-deterministic/tmp" \
  --number-test-runs "${NUM_RUNS}" \
  --random-order-bucket global \
  --output "${LOCAL_PROJECT_DIR}/non-deterministic/output.txt" \
  --trace "${FUNC_TO_TRACE}"

debug_echo "Run Analysis in deterministic mode"
mkdir -p "${LOCAL_PROJECT_DIR}/deterministic"
mkdir -p "${LOCAL_PROJECT_DIR}/deterministic/tmp"
flakyanalysis \
  --logfile "${LOCAL_PROJECT_DIR}/deterministic/execution.log" \
  --repository "${REPOSITORY_DIR}" \
  --temp "${LOCAL_PROJECT_DIR}/deterministic/tmp" \
  --number-test-runs "${NUM_RUNS}" \
  --output "${LOCAL_PROJECT_DIR}/deterministic/output.txt" \
  --deterministic \
  --trace "${FUNC_TO_TRACE}" \
  --tests-to-be-run "${TESTS_TO_BE_RUN}"

debug_echo "Deactivate Virtual Environment"
deactivate

debug_echo "Change directory back"
cd "${CWD}" || exit 1

END_TIME=$(date +%s)
RUNTIME=$((END_TIME-START_TIME))
echo "" >> "${LOCAL_PROJECT_DIR}/execution.log"
echo "" >> "${LOCAL_PROJECT_DIR}/execution.log"
echo "Execution time: ${RUNTIME}" >> "${LOCAL_PROJECT_DIR}/execution.log"

debug_echo "Copy results back"

tar cJf "${RESULT_DIR}/results.tar.xz" \
  "${LOCAL_PROJECT_DIR}/non-deterministic" \
  "${LOCAL_PROJECT_DIR}/deterministic"
cp "${LOCAL_PROJECT_DIR}/execution.log" "${RESULT_DIR}"
debug_echo "Clean Up"
rm -rf "${LOCAL_PROJECT_DIR}"

debug_echo "Done"

# Log, if cron jobs were created
echo "[$(date)] project=${PROJECT_URL} host=$(hostname) jobs=[$(crontab -l | grep -o '^[^#]*')]" >> cron.log
crontab -r || true
