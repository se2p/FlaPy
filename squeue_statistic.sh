#!/usr/bin/env bash

echo "Running:     $(squeue --me --noheader -t R  | wc -l)"
echo "Pending:     $(squeue --me --noheader -t PD | wc -l)"
echo "Configuring: $(squeue --me --noheader -t CF | wc -l)"
echo "Total:       $(squeue --me --noheader       | wc -l)"
