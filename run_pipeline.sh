#!/bin/bash

python3 maskcreation.py $1
if [ $? -eq 1 ]; then
  echo "Aborting."
  exit
fi

python3 ftextraction.py $1
if [ $? -eq 1 ]; then
  echo "Aborting."
  exit
fi

python3 kmeans_initialize.py $1 0
if [ $? -eq 1 ]; then
  echo "Aborting."
  exit
fi

python3 kmeans_predict.py $1
if [ $? -eq 1 ]; then
  echo "Aborting."
  exit
fi
