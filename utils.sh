# MEANT TO BE SOURCED


# -- HELPER FUNCTIONS

function debug_echo {
    if [ "${DEBUG}" != 0 ]; then
        >&2 echo "$@"
    fi
}
