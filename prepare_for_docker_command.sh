# MEANT TO BE SOURCED

SCRIPT_DIR=$(dirname $0)

source "$SCRIPT_DIR/utils.sh"

if [[ -n $FLAPY_DOCKER_COMMAND_SETUP_SCRIPT ]]; then
    debug_echo "-- prepare_for_docker_command: Using custom setup script"
    source $FLAPY_DOCKER_COMMAND_SETUP_SCRIPT
else
    debug_echo "-- prepare_for_docker_command: Using default setup script"
    source "$SCRIPT_DIR/setup_docker_command.sh"
fi
