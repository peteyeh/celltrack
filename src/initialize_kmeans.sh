#!/bin/bash

input="manifest.txt"

cat ../reference/base_km.sh | sed "s|INPUT|${input}|g" | sed "s|GIVEN_K|${1}|g" > __temp_run.sh
sbatch __temp_run.sh
rm __temp_run.sh
