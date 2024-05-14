#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 30:00
#SBATCH -p exercise-eml
#SBATCH -o slurm_output.log

# load appropriate conda paths, because we are not in a login shell

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "no cuda"
python 3_1.py --no-cuda $true

echo "with cuda"
python 3_1.py --no-cuda $false

echo "plot"
python plot.py