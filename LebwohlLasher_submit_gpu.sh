#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --job-name=gpujob
#SBATCH --account=PHYS030544

# Load environments
module load languages/anaconda3/2022.11-3.9.13

# Change to working directory, where job was submitted from
cd "${SLURM_SUBMIT_DIR}"

# Record some potentially useful details about the job: 
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
echo "CPUs per task = ${SLURM_CPUS_PER_TASK}"
echo "N p Seed Steps ReachedEnd Time(s)"
printf "\n\n"

# Submitting and timing code runs
# Recording start time
start_time=$(date +%s)

# File run
python -m cProfile code_files/LebwohlLasher_cupy.py 50 100 0.5 0
python -m cProfile code_files/LebwohlLasher_cupy.py 50 100 0.5 0
python -m cProfile code_files/LebwohlLasher_cupy.py 50 100 0.5 0

# End recording the end time
end_time=$(date +%s)

# Calculate and print the runtime
runtime=$((end_time - start_time))
echo "Total runtime: $runtime seconds"