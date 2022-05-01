#!/bin/bash
#SBATCH --account=p31689
#SBATCH --partition=short
#SBATCH --ntasks-per-node=51
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --job-name=BASENAME
#SBATCH --output=outlog-BASENAME
#SBATCH --error=errlog-BASENAME

module purge all
module load python/anaconda3.6

source activate celltrack

cd ~/celltrack/src

output="/projects/p31689/Image_Analysis/output"
input="INPUT"

python maskcreation.py $input $output
python ftextraction.py $input $output
