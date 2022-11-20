#!/usr/bin/env bash

source utils.sh

# -- SET UP ENVIRONMENT (define flapy_docker_command)
debug_echo "-- Prepare for docker command"
source prepare_for_docker_command.sh || exit


debug_echo "-- IMAGES"
flapy_docker_command images

debug_echo

# echo all containers
debug_echo "-- CONTAINERS"
flapy_docker_command ps -a
