#!/bin/bash

SLURM_JOB_ID=$1

total=1
# periodically look for jobs pending/running
while [[ "${total}" -gt 0 ]]
do
  pending=$(squeue --noheader --array -j "${SLURM_JOB_ID}" -t PD | wc -l)
  running=$(squeue --noheader --array -j "${SLURM_JOB_ID}" -t R | wc -l)
  total=$(squeue --noheader --array -j "${SLURM_JOB_ID}" | wc -l)
  current_time=$(date)
  echo "${current_time}: Job ${SLURM_JOB_ID}: ${total} runs found (${pending} pending, ${running} running)"
  if [[ "${total}" -gt 0 ]]
  then
    sleep 10
  fi
done
