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


################ create bar plot 4.2 ################### 
echo "run 4_2 exercise accuracy"
python 4_1.py --epochs 30 --L2_reg 0.0
echo "run 4_2 exercise accuracy"
python 4_1.py --epochs 30 --L2_reg 0.001
echo "run 4_2 exercise accuracy"
python 4_1.py --epochs 30 --L2_reg 0.0001
echo "run 4_2 exercise accuracy"
python 4_1.py --epochs 30 --L2_reg 0.00001
echo "run 4_2 exercise accuracy"
python 4_1.py --epochs 30 --L2_reg 0.000001

# echo "plot"
python plot.py --plot-weight-decay



############### create drop out plot ##############