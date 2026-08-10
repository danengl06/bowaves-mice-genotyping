#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=mice
#SBATCH --partition=ececis_research
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH -D /work/cniel/mice
#UD_QUIET_JOB_SETUP=YES
#export UD_JOB_EXIT_FN_SIGNALS="SIGTERM EXIT"
#
# Add a container to the environment:
#
vpkg_devrequire intel-python/2022u1:python3
conda activate /work/cniel/mice/venv

# Specify the path to the config file
# for j in 4 3 2 1 0; do for i in 0 1; do for k in test train; do for s in 0 1 2; do for g in 0 1; do echo $i,$j,$k,$s,$g; done; done; done; done; done > new_job_array.csv
config=all_count_job_array.csv

line_N=$( awk "NR==$SLURM_ARRAY_TASK_ID" $config )  # NR means row-# in Awk
fold=$( echo "$line_N" | cut -d "," -f 1 )  # grab comma-delim'd field #2
split=$( echo "$line_N" | cut -d "," -f 2 )  # grab comma-delim'd field #3
learn=$( echo "$line_N" | cut -d "," -f 3 )  # grab comma-delim'd field #4
class1=$( echo "$line_N" | cut -d "," -f 4 )  # grab comma-delim'd field #4
class2=$( echo "$line_N" | cut -d "," -f 5 )  # grab comma-delim'd field #4

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID} >> output.txt
echo "This is array task ${SLURM_ARRAY_TASK_ID}, fold $fold, split $split, learn $learn, class1 $class1, class2 $class2." >> output.txt

# Run command
# python get_counts_by_split.py --fold $fold --split $split  --learn $learn --segments 480 --hours_in_segment 1
python get_counts_by_split.py --fold $fold --split $split --learn $learn --class1 $class1  --class2 $class2  --segments 480 --windows 100000 --hours_in_segment 1 --sphere --svd
