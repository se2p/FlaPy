#!/usr/bin/env bash

SCRIPT_DIR=$(dirname $0)

source "$SCRIPT_DIR/utils.sh"

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

    HELP_MESSAGE="Usage: ./flapy.sh run RUN_ON  CONSTRAINT  INPUT_CSV  PLUS_RANDOM_RUNS  FLAPY_ARGS  [OUT_DIR]

        RUN_ON must be either 'locally' or 'cluster'
        CONSTRAINT is the \`sbatch --constraint\` in case RUN_ON == 'cluster'
        INPUT_CSV is the flapy input csv file,
            which must have the following columns in the following order:
            PROJECT_NAME, PROJECT_URL, PROJECT_HASH, PYPI_TAG, FUNCS_TO_TRACE, TESTS_TO_RUN, NUM_RUNS
        PLUS_RANDOM_RUNS must be 'true' or 'false'
        FLAPY_ARGS can contain the following, but must always be provided, even as empty string.
            Must always be one string.
            Available options:
            --random-order-seed <seed>
        OUT_DIR is the parent folder of the output results directory.
            If this option is not provided, the current directory is used

    Example (takes ~30min): ./flapy.sh run locally \"\" flapy_input_example.csv false \"\" example_results

    Example (takes ~30s): ./flapy.sh run locally \"\" flapy_input_example_tiny.csv false \"\" example_results_tiny"

    # -- CHECK NUMBER OF ARGUMENTS
    if [ "$#" -lt 6 ]; then
        debug_echo "$HELP_MESSAGE"
        exit 1
    else
        "$SCRIPT_DIR/run_csv.sh" "$2" "$3" "$4" "$5" "$6" "$7"
    fi
elif [ "$COMMAND" == "parse" ]; then
    $SCRIPT_DIR/results_parser.sh $ARGS
else
    debug_echo "Unknown command '$COMMAND'"
    debug_echo "available commands: 'run', 'parse'"
    exit 1
fi

