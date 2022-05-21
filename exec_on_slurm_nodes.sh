#!/usr/bin/env bash

NODES=$1
RUN_ME=$2
ARGS="${@:3}"

echo "-- $0"
echo "    NODES=$NODES"
echo "    RUN_ME=$RUN_ME"
echo "    ARGS=$ARGS"

DATE_TIME=$(date +%Y%m%d_%H%M%S)

LOG_DIR="output_from_exec_on_slurm_node/${DATE_TIME}_${RUN_ME}/"
mkdir -p $LOG_DIR

# rm "${script_dir}/logs/current_log.csv"
# ln -s "$log_file" "${script_dir}/logs/current_log.csv"

while read -r node; do
    sbatch \
        -o "${LOG_DIR}/%N.out" \
        -e "${LOG_DIR}/~%N.err" \
        --parsable \
        -w "$node" -- $RUN_ME $ARGS
done < $NODES
