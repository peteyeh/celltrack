#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --job-name=kmeans
#SBATCH --output=outlog-kmeans
#SBATCH --error=errlog-kmeans

module purge all
module load python/anaconda3.6

source activate celltrack

cd ~/celltrack/src

output="/projects/b1042/Abazeed_Lab/Pete_Priyanka/output"
input="INPUT"

python kmeans_initialize.py $input $output GIVEN_K
