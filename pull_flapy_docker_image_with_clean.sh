#!/usr/bin/env bash

#SBATCH --job-name="pull docker image with clean"
#SBATCH --time=01:00:00
#SBATCH --mem-bind=local
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1

echo "-- Node $HOSTNAME"

# -- SET UP ENVIRONMENT (define flapy_docker_command)
echo "-- Prepare for docker command"
source prepare_for_docker_command.sh || exit

echo "-- Cleaning images and containers"
./docker_rm_all_images_and_containers.sh

./pull_flapy_docker_image.sh

echo "-- $0: Done!"
