#!/usr/bin/env bash

source utils.sh

# -- SET UP ENVIRONMENT (define flapy_docker_command)
debug_echo "-- Prepare for docker command"
source prepare_for_docker_command.sh || exit

debug_echo "-- Removing containers"
flapy_docker_command rm -vf $(flapy_docker_command ps -aq)

debug_echo "-- Echo image+container info"
./echo_flapy_docker_info.sh
