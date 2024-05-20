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


################ create plot 4.3 ################### 
echo "run 4_3 exercise augmentation baseline without augmentation"
#python 4_1.py --epochs 30 --augment 0

echo "run 4_3 exercise augmentation with randomcrop"
#python 4_1.py --epochs 30 --augment 1

echo "run 4_3 exercise augmentation with normalize"
#python 4_1.py --epochs 30 --augment 2

echo "run 4_3 exercise augmentation with randomrotate"
#python 4_1.py --epochs 30 --augment 3

echo "plot"
python plot.py --plot-augment



############### create drop out plot ##############