#!/usr/bin/env bash

# -- SET UP ENVIRONMENT (define flapy_docker_command)
echo "-- Prepare for docker command"
source prepare_for_docker_command.sh || exit


echo "-- IMAGES"
flapy_docker_command images

echo

# echo all containers
echo "-- CONTAINERS"
flapy_docker_command ps -a
