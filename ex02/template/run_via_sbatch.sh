#!/bin/bash
#SBATCH --gres gpu:0
#SBATCH --mem 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 30:00
#SBATCH -p exercise-eml
#SBATCH -o slurm_output.log

# load appropriate conda paths, because we are not in a login shell
# eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
# conda activate eml

# python exercise02_template.py

echo "Running exercise02_template.py"
source .venv/bin/activate
python ex02/template/exercise02_template.py