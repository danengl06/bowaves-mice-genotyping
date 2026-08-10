#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=mice_sc
#SBATCH --partition=ececis_research
#SBATCH --time=1-10:00:00
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
config=classifier_job_args.csv

line_N=$( awk "NR==$SLURM_ARRAY_TASK_ID" $config )  # NR means row-# in Awk
fold=$( echo "$line_N" | cut -d "," -f 1 )  # grab comma-delim'd field #2
split=$( echo "$line_N" | cut -d "," -f 2 )  # grab comma-delim'd field #3
task=$( echo "$line_N" | cut -d "," -f 3 )  # grab comma-delim'd field 

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID} >> output.txt
echo "This is array task ${SLURM_ARRAY_TASK_ID}, fold $fold, split $split, task $task." >> output.txt

# Run command
python get_soft_classifiers_by_split.py --fold $fold --split $split --task $task --windows 100000 --loo --sphere --svd
