#!/bin/bash

# -- SET UP ENVIRONMENT (define flapy_docker_command and FLAPY_DOCKER_IMAGE)
echo "-- Preparing slurm node"
source prepare_slurm_node.sh || exit

flapy_docker_command images $FLAPY_DOCKER_IMAGE --format "{{.ID}}"
