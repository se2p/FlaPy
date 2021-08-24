#!/bin/bash
#SBATCH --partition=anywhere
#SBATCH --job-name=flapy
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --array=1-1

n=${SLURM_ARRAY_TASK_ID}
csv_file=$1
csv_line=$(sed "${n}q;d" "${csv_file}")

PROJECT_NAME=$(echo "${csv_line}" | cut -d',' -f1)
PROJECT_URL=$(echo "${csv_line}" | cut -d',' -f2)
PROJECT_HASH=$(echo "${csv_line}" | cut -d',' -f3)
FUNCS_TO_TRACE=$(echo "${csv_line}" | cut -d',' -f4)
TESTS_TO_BE_RUN=$(echo "${csv_line}" | cut -d',' -f5)
NUM_RUNS=$(echo "${csv_line}" | cut -d',' -f6)

LOCAL="/local/hdd/${USER}"
mkdir -p "${LOCAL}/${PROJECT_NAME}"

# Pick a postfix for the repo-dir to allow multiple runs of the same project and the same machine
if [[ ${FUNCS_TO_TRACE} == "" ]]; then
    # In case tracing is deactivated, the postfix is a random number.
    #  (It would be better to choose an unused name within the target directory)
    LOCAL_PROJECT_DIR=$(mktemp -d "${LOCAL}/${PROJECT_NAME}/XXXXX")
    REPO_POSTFIX=$(basename "${LOCAL_PROJECT_DIR}")
else
    # In case tracing is activated, postfix should be consistent accross multiple executions
    # , because it might influence the trace -> use hash of funcs-to-trace
    REPO_POSTFIX=$(echo "$FUNCS_TO_TRACE" | md5sum | head -c 8)
    LOCAL_PROJECT_DIR="${LOCAL}/${PROJECT_NAME}/${REPO_POSTFIX}"
fi
mkdir -p "${LOCAL_PROJECT_DIR}"

echo "Run Jobs for ${PROJECT_NAME}"
echo "URL ${PROJECT_URL}"
echo "Functions to trace: ${FUNCS_TO_TRACE}"

function sighdl {
  kill -INT "${srunPid}" || true
}

mkdir -p "flapy-results/run"
OUT_FILE="flapy-results/run/${PROJECT_NAME}-${REPO_POSTFIX}.txt"

srun \
  --user-cgroups=on \
  --output="${OUT_FILE}" \
  --error="${OUT_FILE}" \
  -- \
  ./run_execution.sh "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${LOCAL_PROJECT_DIR}" \
  & srunPid=$!

trap sighdl INT TERM HUP QUIT

while ! wait; do true; done
