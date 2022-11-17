# MEANT TO BE SOURCED

# -- CHECK IF ENVIRONMENT VARIABLES ARE DEFINED
if [ -z ${LOCAL_PODMAN_ROOT+x} ]; then
    echo "LOCAL_PODMAN_ROOT not set, exiting"
    return 1
fi
if [ -z ${PODMAN_HOME+x} ]; then
    echo "PODMAN_HOME not set, exiting"
    return 1
fi
if [ -z ${LOCAL_TMP+x} ]; then
    echo "LOCAL_TMP not set, exiting"
    return 1
fi

echo "-- Creating LOCAL_PODMAN_ROOT dir: ${LOCAL_PODMAN_ROOT}"
mkdir -p "${LOCAL_PODMAN_ROOT}"
echo "-- Creating PODMAN_HOME dir: ${PODMAN_HOME}"
mkdir -p "${PODMAN_HOME}"
echo "-- Creating LOCAL_TMP dir: ${LOCAL_TMP}"
mkdir -p "${LOCAL_TMP}"

echo "-- (Un)setting global variables (necessary on slurm nodes)"
unset XDG_RUNTIME_DIR
unset XDG_CONFIG_HOME

echo "-- Re-directing HOME (necessary on slurm nodes)"
export HOME=$PODMAN_HOME

echo "-- Define FlaPy docker image"
export FLAPY_DOCKER_IMAGE="registry.hub.docker.com/gruberma/flapy"

echo "-- Creating alias 'flapy_docker_command'"
function flapy_docker_command {
    # Two modifications to the normal podman command:
    # 1. Setting a different tmp-dir to avoid overflowing /tmp, which doesn't have much space
    # 2. Setting a different root, since we have no home directory on slurm
    TMPDIR="$LOCAL_TMP" podman --root="$LOCAL_PODMAN_ROOT" "$@"
}
alias p='flapy_docker_command'
