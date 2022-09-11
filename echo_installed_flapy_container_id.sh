#!/bin/bash

unset XDG_RUNTIME_DIR
unset XDG_CONFIG_HOME
export HOME=$PODMAN_HOME

podman --root="${LOCAL_PODMAN_ROOT}" images localhost/flapy --format "{{.ID}}"
