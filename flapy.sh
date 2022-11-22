#!/usr/bin/env bash

source utils.sh

HELP_MESSAGE="Usage: $0 COMMAND ARGS

available commands: 'run', 'parse'"

# -- CHECK NUMBER OF ARGUMENTS
if [[ "$#" -lt 1 ]]; then
    debug_echo "$HELP_MESSAGE"
    exit 1
fi

COMMAND=$1
ARGS="${@:2}"
echo $ARGS

if [ "$COMMAND" == "run" ]; then
    # TODO: use ARGS here and getopts in ./run_csv.sh
    # problem: arguments get expanded, specifically the additional arguments -> I have to escape
    # them again
    ./run_csv.sh "$2" "$3" "$4" "$5" "$6" "$7"
elif [ "$COMMAND" == "parse" ]; then
    ./results_parser.sh $ARGS
else
    debug_echo "Unknown command '$COMMAND'"
    debug_echo "available commands: 'run', 'parse'"
    exit 1
fi

