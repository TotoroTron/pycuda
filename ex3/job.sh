#!/bin/bash

#SBATCH --job-name=tutorial
#SBATCH --output=logs/slurm.%j.%N.out   # Output file name
#SBATCH --error=logs/slurm.%j.%N.err     # STDERR output file (optional)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=12000                 # Real memory (RAM) required (MB)
#SBATCH --time=01:00:00             # Total run time limit (HH:MM:SS)

# CREATE A NEW LOG DIRECTORY BASED ON PYTHON FILE EXECUTED

# CHECK IF FILENAME PROVIDED
if [[ -z $1 ]]; 		# if param is NULL or uninit (no arg passed)
then
    echo "No Python filename passed. Exiting."
    exit 1
fi

# EXTRACT THE FILENAME WITHOUT EXTENSION
filename=$(basename -- "$1")	# extracts "file.txt" from "path/to/file.txt"
filename="${filename%.*}" 	# extracts "file" from "file.txt"

# CREATE DIRECTORY IF DOESNT EXIST
log_dir="logs_$filename"
mkdir -p "$log_dir"

# SET OUTPUT/ERROR FILE PATHS
#SBATCH --output="$log_dir/slurm.%j.out"
#SBATCH --error="$log_dir/slurm.%j.err"

module purge
module load cuda/10.0

source /home/bbc33/anaconda3/bin/activate
conda activate /home/bbc33/anaconda3/envs/pycuda

python3 $1
