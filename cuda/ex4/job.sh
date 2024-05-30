#!/bin/bash

# https://slurm.schedmd.com/sbatch.html 

# CREATE A NEW LOG DIRECTORY BASED ON PYTHON FILE EXECUTED
# USAGE: ./job.sh filename.py
# OR: bash job.sh filename.py

# CHECK IF FILENAME PROVIDED
if [ -z "$1" ]; then
    echo "Error: No Python filename provided. Exiting."
    exit 1
fi

# EXTRACT THE FILENAME WITHOUT EXTENSION
filename=$(basename -- "$1")
filename="${filename%.*}"

# CREATE DIRECTORY IF DOESNT EXIST
log_dir="logs/logs_$filename"
mkdir -p "$log_dir"

log_old_dir="$log_dir/old"
mkdir -p "$log_old_dir"

# FIND AND COUNT THE NUMBER OF FILES
file_count=$(ls -1t "$log_dir" | wc -l)
# -1 : one per line, -t : sort by modification time
# wc : word count, -l : line count instead

# IF MORE THAN 6 FILES, MOVE ALL EXCEPT NEWEST 6 TO 'old'
if [ "$file_count" -gt 6 ]; then # -gt 6 : greater than 6
    files_to_move=$(($file_count - 6))

    # List the oldest files and move them to the 'old' subdirectory
    ls -1t "$log_dir" | tail -n "$files_to_move" | while read file; do
        # ls -1 lists one file per line, -t sort by modification time
        # tail (opposite of head) -n <number of lines>
        # loop: read a line from tail into $file
        mv "${log_dir}/${file}" "$log_old_dir"
    done
fi

# SET OUTPUT/ERROR FILE PATHS
output_file="$log_dir/slurm.%j.%N.out"
error_file="$log_dir/slurm.%j.%N.err"

# CALL SBATCH WITH OUTPUT/ERROR FILE PATHS
# PASS IN SLURM SCRIPT AS HERE-DOC
# command --args --args <<DELIMITER (text text) DELIMITER
sbatch --output="$output_file" --error="$error_file" <<EOF
#!/bin/bash
#SBATCH --job-name=tutorial
#SBATCH --partition=main
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64000                 # Real memory (RAM) required (MB)
#SBATCH --time=01:00:00             # Total run time limit (HH:MM:SS)

module purge
module load cuda/10.0
source /home/bbc33/anaconda3/bin/activate
conda activate /home/bbc33/anaconda3/envs/pycuda    

python3 $1
EOF
