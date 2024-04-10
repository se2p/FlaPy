#!/bin/bash
set -e

source utils.sh

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
debug_echo "-- $0"
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
    cp -r /project_sources "${REPOSITORY_DIR}"
fi

# -- MEASURE LINES OF CODE
LOC_FILE="${RESULT_DIR}/loc.csv"
CLOC_DBFILE="${CWD}/flapy_cloc.sqlite"
cloc --sql - "$REPOSITORY_DIR" | sqlite3 "${CLOC_DBFILE}"
# Query cloc results
SQL="select Language                  ,
            count(File)   as files    ,
            sum(nBlank)   as blank    ,
            sum(nComment) as comment  ,
            sum(nCode)    as code     ,
            sum(nBlank)+sum(nComment)+sum(nCode) as Total
    from t group by Language order by code desc;
"
# Write header
echo "language,files,blank,comment,code,total" > "$LOC_FILE"
# Write cloc information
(echo "${SQL}" | sqlite3 -csv "${CLOC_DBFILE}") >> "$LOC_FILE"

# -- MEASURE SIZE
DISK_USAGE=$(du -s "$REPOSITORY_DIR" | cut -f 1)

# -- LOG META INFOS (1/2)
START_TIMESTAMP=$(date +%s)
START_DATE=$(date)
{
    echo "project_git_hash:       \"$REPO_HASH\""
    echo "disk_usage:             $DISK_USAGE"
    echo "start_time:             $START_DATE"
} >> $META_FILE

# -- EXPLORE MODULES
python3 ./findpackages.py "$REPOSITORY_DIR" > "${RESULT_DIR}/found_modules.txt"
python3 ./find_all_modules.py "$REPOSITORY_DIR" > "${RESULT_DIR}/found_modules_all.txt"

# -- EXECUTE TESTS
debug_echo "Run tests in same order mode"
mkdir -p "${CWD}/sameOrder"
mkdir -p "${CWD}/sameOrder/tmp"
flapy_run_tests \
  --logfile "${CWD}/sameOrder/execution.log" \
  --repository "${REPOSITORY_DIR}" \
  --project-name "${PROJECT_NAME}" \
  --pypi-tag "${PYPI_TAG}" \
  --temp "${CWD}/sameOrder/tmp" \
  --number-test-runs "${NUM_RUNS}" \
  --trace "${FUNCS_TO_TRACE}" \
  --tests-to-be-run "${TESTS_TO_BE_RUN}" \
  $FLAPY_ARGS  # do not double quote here! we want the word splitting

if [[ $PLUS_RANDOM_RUNS = true ]]
then
    debug_echo "Run tests in random order mode"
    mkdir -p "${CWD}/randomOrder"
    mkdir -p "${CWD}/randomOrder/tmp"
    flapy_run_tests \
      --logfile "${CWD}/randomOrder/execution.log" \
      --repository "${REPOSITORY_DIR}" \
      --project-name "${PROJECT_NAME}" \
      --pypi-tag "${PYPI_TAG}" \
      --temp "${CWD}/randomOrder/tmp" \
      --number-test-runs "${NUM_RUNS}" \
      --random-order-bucket class \
      --trace "${FUNCS_TO_TRACE}" \
      --tests-to-be-run "${TESTS_TO_BE_RUN}" \
      $FLAPY_ARGS  # do not double quote here! we want the word splitting
fi

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
      "${CWD}/randomOrder" \
      "${CWD}/sameOrder" \
      "$CLOC_DBFILE"
else
    debug_echo "Copy results back (no random runs)"
    tar cJf "${RESULT_DIR}/results.tar.xz" \
      "${CWD}/sameOrder" \
      "$CLOC_DBFILE"
fi

debug_echo "Done"
