# MEANT TO BE SOURCED

# This script needs to do two things:
# 1. define variable FLAPY_DOCKER_IMAGE
# 2. define function flapy_docker_command

echo "-- Define FlaPy docker image"
export FLAPY_DOCKER_IMAGE="registry.hub.docker.com/gruberma/flapy"

echo "-- Creating alias 'flapy_docker_command'"
function flapy_docker_command {
    # Possible modifications to the normal podman command:
    # * Setting a different tmp-dir to avoid overflowing /tmp by prepending TMPDIR=...
    # * Use 'podman' instead of docker
    docker $@
}
