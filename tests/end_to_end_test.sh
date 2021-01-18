#!/usr/bin/env bash

FUNCS_TO_TRACE=""
PROJECT_NAME="test_resources"

LOCAL="/local/hdd/${USER}"
mkdir -p "${LOCAL}/${PROJECT_NAME}"

# Pick a postfix for the repo-dir to allow multiple runs of the same project and the same machine
if [[ ${FUNCS_TO_TRACE} == "" ]]; then
    # In case tracing is deactivated, the postfix is a random number.
    #  (It would be better to choose an unused name within the target directory)
    LOCAL_PROJECT_DIR=$(mktemp -d "${LOCAL}/${PROJECT_NAME}/XXXXX")
    REPO_POSTFIX=$(basename ${LOCAL_PROJECT_DIR})
else
    # In case tracing is activated, postfix should be consistent accross multiple executions
    # , because it might influence the trace -> use hash of funcs-to-trace
    REPO_POSTFIX=$(echo FUNCS_TO_TRACE | md5sum | head -c 8)
    LOCAL_PROJECT_DIR=$(mktemp -d "${LOCAL}/${PROJECT_NAME}/${REPO_POSTFIX}")
    mkdir -p "${LOCAL_PROJECT_DIR}"
fi


#                    Name               Url                           Hash   Trace TestsToRun NumRuns PostFix
../run_execution.sh "${PROJECT_NAME}" "$(realpath ../test_resources)" master ""    ""         1       "${LOCAL_PROJECT_DIR}"

