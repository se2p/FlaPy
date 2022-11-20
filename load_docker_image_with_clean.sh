#!/usr/bin/env bash

#SBATCH --job-name="load docker image with clean"
#SBATCH --time=01:00:00
#SBATCH --mem-bind=local
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1

source utils.sh

DOCKER_IMAGE=$1

# -- SET UP ENVIRONMENT (define flapy_docker_command)
debug_echo "-- Prepare for docker command"
source prepare_for_docker_command.sh || exit

debug_echo "-- Cleaning images and containers"
./docker_rm_all_images_and_containers.sh

./load_docker_image.sh

debug_echo "-- $0: Done!"
