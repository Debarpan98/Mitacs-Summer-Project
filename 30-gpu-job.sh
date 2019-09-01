#!/bin/bash
#SBATCH --account=def-erangauk-ab
#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-12:00            # time (DD-HH:MM)
#SBATCH --mail-user=fabiisele@gmail.com  # adjust this to match your email address
#SBATCH --mail-type=ALL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
. venv/bin/activate
python main.py 
