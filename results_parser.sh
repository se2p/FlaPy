#!/usr/bin/env bash

source utils.sh

debug_echo "-- Prepare for docker command"
source prepare_for_docker_command.sh || exit

flapy_docker_command run --rm --entrypoint=results_parser \
    -v "$(pwd)":/mounted_cwd --workdir /mounted_cwd \
    $FLAPY_DOCKER_IMAGE "$@"

