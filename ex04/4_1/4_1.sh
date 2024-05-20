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


################ create bar plot ################### 
echo "Running 4_1.py no cuda epoch 1"
python 4_1.py --no-cuda --epochs 1

echo "Running 4_1.py cuda epoch 1"
python 4_1.py --epochs 1

echo "plot"
python plot.py --plot-bar


############### create drop out plot ##############

echo "Running 4_1.py"
python 4_1.py --epochs 30

echo "Running 4_1.py --dropout_p 0.1"
python 4_1.py --epochs 30 --dropout_p 0.1

echo "Running 4_1.py --dropout_p 0.1"
python 4_1.py --epochs 30 --dropout_p 0.3

echo "Running 4_1.py --dropout_p 0.1"
python 4_1.py --epochs 30 --dropout_p 0.5

echo "Running 4_1.py --dropout_p 0.1"
python 4_1.py --epochs 30 --dropout_p 0.7

echo "Running 4_1.py --dropout_p 0.1"
python 4_1.py --epochs 30 --dropout_p 0.9

echo "plot dropout"
python plot.py --plot-dropout

