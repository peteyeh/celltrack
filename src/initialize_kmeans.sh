#!/bin/bash

input=$1

while read path; do
  python3 maskcreation.py $path
  if [ $? -eq 1 ]; then
    echo "Failed mask creation. Aborting."
    exit
  fi
  python3 ftextraction.py $path
  if [ $? -eq 1 ]; then
    echo "Failed feature extraction. Aborting."
    exit
  fi
done < $input

python3 kmeans_initialize.py $input
