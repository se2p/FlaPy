echo "-- (Un)setting global variables"
unset XDG_RUNTIME_DIR
unset XDG_CONFIG_HOME
export HOME=$PODMAN_HOME

echo "-- Creating alias 'p'"
alias p='podman --root=$LOCAL_PODMAN_ROOT'
