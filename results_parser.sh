#!/usr/bin/env bash

SCRIPT_DIR=$(dirname $0)

source "$SCRIPT_DIR/utils.sh"

export DEBUG=0

debug_echo "-- Prepare for docker command"
source "$SCRIPT_DIR/prepare_for_docker_command.sh" || exit

flapy_docker_command run --rm -it --entrypoint=results_parser \
    -v "$(pwd)":/mounted_cwd --workdir /mounted_cwd \
    $FLAPY_DOCKER_IMAGE "$@"

