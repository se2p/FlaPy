#!/usr/bin/env bash

echo "Running:     $(squeue --me --array --noheader -t R  | wc -l)"
echo "Pending:     $(squeue --me --array --noheader -t PD | wc -l)"
echo "Configuring: $(squeue --me --array --noheader -t CF | wc -l)"
echo "Total:       $(squeue --me --array --noheader       | wc -l)"
