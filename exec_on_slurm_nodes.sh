#!/usr/bin/env bash

source utils.sh

NODES=$1
RUN_ME=$2
ARGS="${@:3}"

debug_echo "-- $0"
debug_echo "    NODES=$NODES"
debug_echo "    RUN_ME=$RUN_ME"
debug_echo "    ARGS=$ARGS"

DATE_TIME=$(date +%Y%m%d_%H%M%S)

LOG_DIR="output_from_exec_on_slurm_node/${DATE_TIME}_${RUN_ME}/"
mkdir -p $LOG_DIR

# rm "${script_dir}/logs/current_log.csv"
# ln -s "$log_file" "${script_dir}/logs/current_log.csv"

while read -r node; do
    sbatch \
        -o "${LOG_DIR}/%N.txt" \
        -e "${LOG_DIR}/%N.txt" \
        --parsable \
        -w "$node" -- $RUN_ME $ARGS
done < $NODES
