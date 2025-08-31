#!/bin/bash
#SBATCH --job-name=image-generation
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --mail-user=dasdebarpan10@gmail.com # adjust this to match your email address
#SBATCH --mail-type=ALL
. venv/bin/activate
python generate_slices.py
