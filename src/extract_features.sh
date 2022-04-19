#!/bin/bash

input="manifest.txt"

while read path; do
  basename=$(echo "${path##*/}" | cut -d "." -f1)
  cat ../reference/base_mc_fe.sh | sed "s|BASENAME|${basename}|g" | sed "s|INPUT|${path}|g" > __temp_run.sh
  sbatch __temp_run.sh
  rm __temp_run.sh
done < $input
