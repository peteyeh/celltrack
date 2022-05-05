#!/bin/bash
#SBATCH --account=p31689
#SBATCH --partition=short
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --job-name=kmeans
#SBATCH --output=logs/kmeans.outlog
#SBATCH --error=logs/kmeans.errlog

module purge all
module load python/anaconda3.6

source activate celltrack

cd ~/celltrack/src

output="/projects/p31689/Image_Analysis/output"
input="INPUT"

python kmeans_initialize.py $input $output GIVEN_K
