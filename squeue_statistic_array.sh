#!/usr/bin/env bash

RUNNING=$(squeue --me --array --noheader -t R  | wc -l)
PENDING=$(squeue --me --array --noheader -t PD | wc -l)
CONFING=$(squeue --me --array --noheader -t CF | wc -l)
TOTAL=$(  squeue --me --array --noheader       | wc -l)

echo "Running:     $RUNNING"
echo "Pending:     $PENDING"
echo "Configuring: $CONFING"
echo "Total:       $TOTAL"

echo "$(date -Iseconds),$RUNNING,$PENDING,$CONFING,$TOTAL" >> squeue_statistic_array_progression.csv
