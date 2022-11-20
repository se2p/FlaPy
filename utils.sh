# MEANT TO BE SOURCED

DEBUG=1

# -- HELPER FUNCTIONS

function debug_echo {
  [[ "${DEBUG}" = 1 ]] && >&2 echo "$@"
}
