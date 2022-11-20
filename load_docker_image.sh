#!/usr/bin/env bash

#SBATCH --job-name="load docker image"
#SBATCH --time=01:00:00
#SBATCH --mem-bind=local
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1

source utils.sh

DOCKER_IMAGE=$1

debug_echo "-- Node $HOSTNAME"

# -- SET UP ENVIRONMENT (define flapy_docker_command)
debug_echo "-- Prepare for docker command"
source prepare_for_docker_command.sh || exit

debug_echo "-- Loading image ${DOCKER_IMAGE}"
date -Iseconds
flapy_docker_command load -i "${DOCKER_IMAGE}"
date -Iseconds

debug_echo "-- Echo image+container info"
./echo_flapy_docker_info.sh

debug_echo "-- $0: Done!"
