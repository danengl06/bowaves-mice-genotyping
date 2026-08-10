#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=mice
#SBATCH --partition=ececis_research
#SBATCH --time=10:00:00
#SBATCH --mail-user='ajbrock@udel.edu'
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
config=dict_job_array.csv

line_N=$( awk "NR==$SLURM_ARRAY_TASK_ID" $config )  # NR means row-# in Awk
fold=$( echo "$line_N" | cut -d "," -f 1 )  # grab comma-delim'd field #2
split=$( echo "$line_N" | cut -d "," -f 2 )  # grab comma-delim'd field #3
class1=$( echo "$line_N" | cut -d "," -f 3 )  # grab comma-delim'd field #4
class2=$( echo "$line_N" | cut -d "," -f 4 )  # grab comma-delim'd field #4

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID} >> output.txt
echo "This is array task ${SLURM_ARRAY_TASK_ID}, fold $fold, split $split, task $task." >> output.txt

# Run command
python get_dict_by_split.py --fold $fold --svd --sphere --split $split --class1 $class1 --class2 $class2  --windows 100000 

