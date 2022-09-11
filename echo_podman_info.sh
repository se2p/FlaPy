#!/usr/bin/env bash

unset XDG_RUNTIME_DIR
unset XDG_CONFIG_HOME
export HOME=$PODMAN_HOME

echo "-- IMAGES (podman images)"
podman --root "${LOCAL_PODMAN_ROOT}" images

echo

# echo all containers
echo "-- CONTAINERS (podman ps -a)"
podman --root "${LOCAL_PODMAN_ROOT}" ps -a
