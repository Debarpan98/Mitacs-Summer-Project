#!/bin/bash
#SBATCH --job-name=mask-generation
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --mail-user=dasdebarpan10@gmail.com # adjust this to match your email address
#SBATCH --mail-type=ALL
. venv/bin/activate
python generate_mask_patches.py
