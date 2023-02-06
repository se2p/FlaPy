# MEANT TO BE SOURCED

# This script needs to do two things:
# 1. define variable FLAPY_DOCKER_IMAGE
# 2. define function flapy_docker_command

# This script can either be modified, or a custom script can be used instead by setting the
# FLAPY_DOCKER_COMMAND_SETUP_SCRIPT variable

SCRIPT_DIR=$(dirname $0)

source "$SCRIPT_DIR/utils.sh"

debug_echo "-- Define FlaPy docker image"
export FLAPY_DOCKER_IMAGE="registry.hub.docker.com/gruberma/flapy"

debug_echo "-- Creating alias 'flapy_docker_command'"
function flapy_docker_command {
    # Possible modifications to the normal docker command:
    # * Setting a different tmp-dir to avoid overflowing /tmp by prepending TMPDIR=...
    # * Use 'podman' instead of docker
    sudo docker "$@"
}
