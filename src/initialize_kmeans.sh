#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --ntasks-per-node=50
#SBATCH --time=00:40:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=image_analysis
#SBATCH --output=outlog
#SBATCH --error=errlog

module purge all                ## Unload existing modules
module load python/anaconda3.6

source activate celltrack

cd ~/celltrack/src

input="manifest.txt"

while read path; do
  echo "Running maskcreation.py for $path."
  python3 maskcreation.py $path
  if [ $? -eq 1 ]; then
    echo "Failed mask creation. Aborting."
    exit
  fi
  echo "Running ftextraction.py for $path."
  python3 ftextraction.py $path
  if [ $? -eq 1 ]; then
    echo "Failed feature extraction. Aborting."
    exit
  fi
done < $input

python3 kmeans_initialize.py $input
