#!/bin/bash
#SBATCH --account=p31689
#SBATCH --partition=short
#SBATCH --ntasks-per-node=26
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --job-name=BASENAME
#SBATCH --output=outlog-BASENAME
#SBATCH --error=errlog-BASENAME

module purge all
module load python/anaconda3.6

source activate celltrack

cd ~/celltrack/src

output="/projects/p31689/Image_Analysis/output"
input="INPUT"

python kmeans_predict.py $input $output
python cluster_analysis.py $input $output
python contour_images.py $input $output
