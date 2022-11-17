#!/bin/bash

# -- SET UP ENVIRONMENT (define flapy_docker_command and FLAPY_DOCKER_IMAGE)
echo "-- Prepare for docker command"
source prepare_for_docker_command.sh || exit

flapy_docker_command images $FLAPY_DOCKER_IMAGE --format "{{.ID}}"
