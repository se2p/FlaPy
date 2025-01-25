#!/usr/bin/env bash

SCRIPT_DIR=$(dirname $0)

source "$SCRIPT_DIR/utils.sh"

HELP_MESSAGE="Usage: $0 COMMAND ARGS

available commands: 'sample', 'run', 'parse', 'fetch-all-pypi-projects'"

# -- CHECK NUMBER OF ARGUMENTS
if [[ "$#" -lt 1 ]]; then
    debug_echo "$HELP_MESSAGE"
    exit 1
fi

COMMAND=$1
ARGS="${@:2}"

if [ "$COMMAND" == "run" ]; then
    # TODO: use ARGS here and getopts in ./run_csv.sh
    # problem: arguments get expanded, specifically the additional arguments -> I have to escape
    # them again

    RUN_HELP_MESSAGE="Usage: ./flapy.sh run [OPTION]... INPUT_CSV NUM_RUNS

INPUT_CSV

    The input-csv file must have the following columns in the following order:
    PROJECT_NAME, PROJECT_URL, PROJECT_HASH, PYPI_TAG, FUNCS_TO_TRACE, TESTS_TO_RUN


NUM_RUNS

    Number of times the test suites should be executed.


OPTIONS

        -r, --run-on RUN_ON
            RUN_ON must be either 'locally' or 'cluster'
            if this option is not specified, RUN_ON defaults to 'locally'

        -c, --constraint CONSTRAINT
            CONSTRAINT is the \`sbatch --constraint\` in case RUN_ON == 'cluster'

        -p, --plus-random-runs
            Additional to the same-order runs, conduct the same number of runs where the
            tests are shuffeled on class level (pytest-random-order plugin)

        -a, --core-args CORE_ARGS
            Used for extensions to the core test execution process.
            Must always be one string, even if multiple options are specified, e.g. \"--foo --bar\".
            Currently available options:

                --random-order-seed <seed>

        -o, --out-dir OUT_DIR
            OUT_DIR is the parent folder of the newly created results-directory.
            If this option is not provided, the current directory is used.


EXAMPLES

    Example (takes ~1h):  ./flapy.sh run --plus-random-runs --out-dir example_results flapy_input_example.csv 5

    Example (takes ~30s): ./flapy.sh run --out-dir example_results_tiny flapy_input_example_tiny.csv 1"


    # -- PARSE ARGUMENT
    SHORT=r:,p,c:,a:,o:
    LONG=run-on:,plus-random-runs,constraint:,core-args:,out-dir:
    OPTS=$(getopt --name "flapy.sh run" --options $SHORT --longoptions $LONG -- "${@:2}") || exit
    #
    eval set -- "$OPTS"
    while :
    do
        case "$1" in
            -r | --run-on )
                RUN_ON="$2"
                shift 2
                ;;
            -p | --plus-random-runs )
                PLUS_RANDOM_RUNS=true
                shift 1
                ;;
            -c | --constraint )
                CONSTRAINT="$2"
                shift 2
                ;;
            -a | --core-args )
                CORE_ARGS="$2"
                shift 2
                ;;
            -o | --out-dir )
                OUT_DIR="$2"
                shift 2
                ;;
            --)
                INPUT_CSV="$2"
                shift;
                NUM_RUNS="$2"
                shift;
                break
                ;;
            *)
                echo "Unexpected option: $1"
                exit 1
                ;;
        esac
    done
    #
    if [ -z $INPUT_CSV ]; then
        debug_echo "ERROR: no INPUT_CSV specified -> exiting"
        debug_echo
        debug_echo "$RUN_HELP_MESSAGE"
        exit 1
    fi
    if [ -z $NUM_RUNS ]; then
        debug_echo "ERROR: NUM_RUNS not specified -> exiting"
        debug_echo
        debug_echo "$RUN_HELP_MESSAGE"
        exit 1
    fi
    if [ -z $RUN_ON ]; then
        RUN_ON="locally"
        debug_echo "--run-on not specified -> defaulting to 'locally'"
    fi

    "$SCRIPT_DIR/run_csv.sh" "$RUN_ON" "$CONSTRAINT" "$INPUT_CSV" "$PLUS_RANDOM_RUNS" "$NUM_RUNS" "$CORE_ARGS" "$OUT_DIR"

elif [ "$COMMAND" == "parse" ]; then
    export DEBUG=0;
    debug_echo "-- Prepare for docker command"
    source "$SCRIPT_DIR/prepare_for_docker_command.sh" || exit
    flapy_docker_command run --log-driver=none --rm -i --entrypoint=results_parser \
        -v "$(pwd)":/mounted_cwd --workdir /mounted_cwd \
        $FLAPY_DOCKER_IMAGE $ARGS
elif [ "$COMMAND" == "sample" ]; then
    export DEBUG=0
    debug_echo "-- Prepare for docker command"
    source "$SCRIPT_DIR/prepare_for_docker_command.sh" || exit
    flapy_docker_command run --log-driver=none --rm -i --entrypoint=sample_pypi_projects \
        -v "$(pwd)":/mounted_cwd --workdir /mounted_cwd \
        $FLAPY_DOCKER_IMAGE $ARGS
elif [ "$COMMAND" == "fetch-all-pypi-projects" ]; then
    export DEBUG=0
    debug_echo "-- Prepare for docker command"
    source "$SCRIPT_DIR/prepare_for_docker_command.sh" || exit
    flapy_docker_command run --log-driver=none --rm -i --entrypoint=fetch_all_pypi_projects \
        -v "$(pwd)":/mounted_cwd --workdir /mounted_cwd \
        $FLAPY_DOCKER_IMAGE $ARGS
else
    debug_echo "Unknown command '$COMMAND'"
    debug_echo
    debug_echo "$HELP_MESSAGE"
    exit 1
fi

