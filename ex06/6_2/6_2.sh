#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --cpus-per-task 2
#SBATCH --time 10:00:00
#SBATCH -p exercise-eml
#SBATCH -o slurm_output.log

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

python qat.py --epochs 50 --bit-widths 2 3 4 6

# python 6_2.py --epochs 2 --bit-widths 2 4 8 16

# echo "plot"
# python plot.py 
