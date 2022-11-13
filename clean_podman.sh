#!/usr/bin/env bash

# -- SET UP ENVIRONMENT (define flapy_docker_command)
echo "-- Preparing slurm node"
source prepare_slurm_node.sh || exit

echo "-- Removing containers"
flapy_docker_command rm -v --force --all

echo "-- Removing images"
flapy_docker_command rmi --force --all

echo "-- Echo podman info"
./echo_podman_info.sh
