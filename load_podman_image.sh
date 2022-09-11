#!/usr/bin/env bash

#SBATCH --job-name="load docker image"
#SBATCH --time=01:00:00
#SBATCH --mem-bind=local
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1

DOCKER_IMAGE=$1

if [ -z ${LOCAL_PODMAN_ROOT+x} ]; then
    echo "LOCAL_PODMAN_ROOT not set, exiting"
    exit
fi
if [ -z ${PODMAN_HOME+x} ]; then
    echo "PODMAN_HOME not set, exiting"
    exit
fi
if [ -z ${LOCAL_TMP+x} ]; then
    echo "LOCAL_TMP not set, exiting"
    exit
fi


echo "-- Node $HOSTNAME"

echo "-- (Un)setting global variables"
unset XDG_RUNTIME_DIR
unset XDG_CONFIG_HOME
export HOME=$PODMAN_HOME

echo "-- Creating LOCAL_PODMAN_ROOT dir: ${LOCAL_PODMAN_ROOT}"
mkdir -p "${LOCAL_PODMAN_ROOT}"
echo "-- Creating PODMAN_HOME dir: ${PODMAN_HOME}"
mkdir -p "${PODMAN_HOME}"
echo "-- Creating LOCAL_TMP dir: ${LOCAL_TMP}"
mkdir -p "${LOCAL_TMP}"

echo "-- Cleaning podman"
./clean_podman.sh

echo "-- Loading image ${DOCKER_IMAGE}"
date -Iseconds
TMPDIR=$LOCAL_TMP podman --root "${LOCAL_PODMAN_ROOT}" load -i "${DOCKER_IMAGE}"
date -Iseconds

echo "-- Echo podman info"
./echo_podman_info.sh

echo "-- Done!"
