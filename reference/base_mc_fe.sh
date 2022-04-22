#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --ntasks-per-node=26
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --job-name=BASENAME
#SBATCH --output=outlog-BASENAME
#SBATCH --error=errlog-BASENAME

module purge all
module load python/anaconda3.6

source activate celltrack

cd ~/celltrack/src

output="/projects/b1042/Abazeed_Lab/Pete_Priyanka/output"
input="INPUT"

python maskcreation.py $input $output
python ftextraction.py $input $output
