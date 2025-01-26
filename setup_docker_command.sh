# MEANT TO BE SOURCED

# This script needs to do two things:
# 1. define variable FLAPY_DOCKER_IMAGE
# 2. define function flapy_docker_command

# This script can either be modified, or a custom script can be used instead by setting the
# FLAPY_DOCKER_COMMAND_SETUP_SCRIPT variable

SCRIPT_DIR=$(dirname $0)

source "$SCRIPT_DIR/utils.sh"

if [[ -n $FLAPY_DOCKER_IMAGE ]]; then
    debug_echo "-- setup_docker_command: Using custom image: $FLAPY_DOCKER_IMAGE"
else
    debug_echo "-- setup_docker_command: FLAPY_DOCKER_IMAGE not set, using default image"
    export FLAPY_DOCKER_IMAGE="registry.hub.docker.com/gruberma/flapy"
fi

debug_echo "-- setup_docker_command: Creating alias 'flapy_docker_command'"
function flapy_docker_command {
    # Possible modifications to the normal docker command:
    # * Setting a different tmp-dir to avoid overflowing /tmp by prepending TMPDIR=...
    # * Use 'podman' instead of docker
    docker "$@"
}
