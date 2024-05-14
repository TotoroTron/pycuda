#!/bin/bash
#SBATCH --job-name=tutorial
#SBATCH --output=logs/slurm.%j.%N.out   # Output file name
#SBATCH --error=logs/slurm.%j.%N.err     # STDERR output file (optional)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64000                 # Real memory (RAM) required (MB)
#SBATCH --time=01:00:00             # Total run time limit (HH:MM:SS)

module purge

module load cuda/10.0

source /home/bbc33/anaconda3/bin/activate
conda activate /home/bbc33/anaconda3/envs/pycuda
python3 main.py
