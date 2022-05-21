#!/usr/bin/env bash

#SBATCH --job-name="load docker image"
#SBATCH --time=01:00:00
#SBATCH --mem-bind=local
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1

DOCKER_IMAGE=$1

# -- Requires LOCAL_PODMAN_ROOT and PODMAN_HOME to be set

echo "-- Node $HOSTNAME"

echo "-- (Un)setting global variables"
unset XDG_RUNTIME_DIR
unset XDG_CONFIG_HOME
export HOME=$PODMAN_HOME

echo "-- Creating ${LOCAL_PODMAN_ROOT}"
mkdir -p "${LOCAL_PODMAN_ROOT}"
echo "-- Creating ${PODMAN_HOME}"
mkdir -p "${PODMAN_HOME}"

./clean_podman.sh

echo "-- Loading image ${DOCKER_IMAGE}"
date -Iseconds
podman --root "${LOCAL_PODMAN_ROOT}" load -i "${DOCKER_IMAGE}"
date -Iseconds

echo "-- Done!"
