#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --ntasks-per-node=52
#SBATCH --time=00:40:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=image_analysis
#SBATCH --output=outlog
#SBATCH --error=errlog

module purge all                ## Unload existing modules
module load python/anaconda3.6

source activate celltrack

