#!/usr/bin/env bash

# -- SET UP ENVIRONMENT (define flapy_docker_command)
echo "-- Prepare for docker command"
source prepare_for_docker_command.sh || exit

echo "-- Removing containers"
flapy_docker_command rm -vf $(flapy_docker_command ps -aq)

echo "-- Echo image+container info"
./echo_flapy_docker_info.sh
