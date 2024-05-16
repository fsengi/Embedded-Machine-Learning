#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --cpus-per-task 2
#SBATCH --time 00:30:00
#SBATCH -p exercise-eml
#SBATCH -o slurm_output.log

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running 4_1.py no cuda epoch 1"
python 4_1.py --no_cuda $true --epochs $1

echo "Running 4_1.py cuda epoch 1"
python 4_1.py --no_cuda $true --epochs $1
