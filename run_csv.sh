#!/usr/bin/env bash


if [[ "${PWD:0:6}" = "/home/" ]]
then
    echo "ERROR: DO NOT EXECUTE THIS IN /HOME"
    echo "    To prevent the executed code from accessing your data, we use runexec and hide the home directory."
    echo "    Therefore, the flapy will not produce any results if executed in the home directory."
    exit
fi

CSV_FILE=$1

dos2unix "${CSV_FILE}"

LOCAL="$(pwd)/local/hdd/${USER}"

while IFS=, read PROJECT_NAME PROJECT_URL PROJECT_HASH FUNCS_TO_TRACE TESTS_TO_BE_RUN NUM_RUNS; do

        echo "Project name:      $PROJECT_NAME"
        echo "Project url:       $PROJECT_URL"
        echo "Project hash:      $PROJECT_HASH"
        echo "Funcs to trace:    $FUNCS_TO_TRACE"
        echo "Num runs:          $NUM_RUNS"

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
            LOCAL_PROJECT_DIR="${LOCAL}/${PROJECT_NAME}/${REPO_POSTFIX}"
        fi
        mkdir -p "${LOCAL_PROJECT_DIR}"

        ./run_execution.sh "${PROJECT_NAME}" "${PROJECT_URL}" "${PROJECT_HASH}" "${FUNCS_TO_TRACE}" "${TESTS_TO_BE_RUN}" "${NUM_RUNS}" "${LOCAL_PROJECT_DIR}"

done <"${CSV_FILE}"
