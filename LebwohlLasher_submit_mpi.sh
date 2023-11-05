#!/bin/bash

#SBATCH --job-name=Project1
#SBATCH --partition=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --time=6:0:0
#SBATCH --mem-per-cpu=100M
#SBATCH --account=PHYS030544

# Load correct anaconda environment
module load languages/anaconda3/2020-3.8.5

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
python -m cProfile code_files/LebwohlLasher_mpi4py.py 50 250 0.5
python -m cProfile code_files/LebwohlLasher_mpi4py.py 50 250 0.5
python -m cProfile code_files/LebwohlLasher_mpi4py.py 50 250 0.5

# End recording the end time
end_time=$(date +%s)

# Calculate and print the runtime
runtime=$((end_time - start_time))
echo "Total runtime: $runtime seconds"