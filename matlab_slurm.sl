#!/bin/bash -l
#SBATCH --job-name=matlab_test
#SBATCH --account=def-erangauk-ab # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-00:05         # adjust this to match the walltime of your job
#SBATCH --nodes=1      
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1      # adjust this if you are using parallel commands
#SBATCH --mem=4000             # adjust this according to your the memory requirement per node you need
#SBATCH --mail-user=you@youruniversity.ca # adjust this to match your email address
#SBATCH --mail-type=ALL

# Choose a version of MATLAB by loading a module:
module load matlab/2018a
# Remove -singleCompThread below if you are using parallel commands:
srun matlab -nodisplay -singleCompThread -r "cosplot"
