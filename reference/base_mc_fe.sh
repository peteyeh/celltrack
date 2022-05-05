#!/bin/bash
#SBATCH --account=p31689
#SBATCH --partition=short
#SBATCH --ntasks-per-node=26
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --job-name=BASENAME
#SBATCH --output=logs/BASENAME-1.outlog
#SBATCH --error=logs/BASENAME-1.errlog

module purge all
module load python/anaconda3.6

source activate celltrack

cd ~/celltrack/src

output="/projects/p31689/Image_Analysis/output"
input="INPUT"

python maskcreation.py $input $output
python ftextraction.py $input $output
