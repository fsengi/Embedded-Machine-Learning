#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 2:30:00
#SBATCH -p exercise-eml
#SBATCH -o slurm_output.log

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running 3_3.py"
python 3_3.py
python plot.py
