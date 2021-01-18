#!/bin/bash

SLURM_JOB_ID=0
PID=$$

csv_file=$1

# Adjust size of slurm array
sed "s/--array=1-.*/--array=1-$(wc -l < "${csv_file}")/g" array_job.sh > tmpfile && mv tmpfile array_job.sh

function sig_handler {
  echo "Canceling the SLURM job..."
  if [[ "$SLURM_JOB_ID" -gt 0 ]]
  then
    scancel "${SLURM_JOB_ID}"
  fi

  echo "Killing the ${0} including its children..."
  pkill -TERM -P ${PID}

  echo -e "Terminated: ${0}\n"
}
trap sig_handler INT TERM HUP QUIT

IFS=',' read SLURM_JOB_ID rest < <(sbatch \
	-o slurm-logs/slurm-%j.out \
	-e slurm-logs/slurm-%j.out \
	--parsable array_job.sh "${csv_file}"\
	)
if [[ -z "${SLURM_JOB_ID}" ]]
then
  echo "Submitting the SLURM job failed!"
  exit 1
fi

echo "SLURM job with ID ${SLURM_JOB_ID} submitted!"
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
