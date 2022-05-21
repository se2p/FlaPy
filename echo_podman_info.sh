#!/usr/bin/env bash

unset XDG_RUNTIME_DIR
unset XDG_CONFIG_HOME
export HOME=$PODMAN_HOME

# echo all images
podman --root "${LOCAL_PODMAN_ROOT}" images

echo

# echo all containers
podman --root "${LOCAL_PODMAN_ROOT}" ps -a
