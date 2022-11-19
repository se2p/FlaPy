# MEANT TO BE SOURCED

if [[ -n $FLAPY_DOCKER_COMMAND_SETUP_SCRIPT ]]; then
    echo "-- prepare_for_docker_command: Using custom setup script"
    source $FLAPY_DOCKER_COMMAND_SETUP_SCRIPT
else
    echo "-- prepare_for_docker_command: Using default setup script"
    source setup_docker_command.sh
fi
