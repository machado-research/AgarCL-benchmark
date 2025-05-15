#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --account=XX
#SBATCH --mail-user=XX
#SBATCH --job-name=XX
#SBATCH --output=XX.out
#SBATCH --error=XX.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=15:00:00
#SBATCH --cpu-freq=Performance
#SBATCH --array=1-10

# module load apptainer
# module --force purge 
module load clang/17.0.6
module load cmake
# module load python/3.10.13
module load python/3.11.5
module load scipy-stack
module load glfw
module load cuda/12.2
module load opencv/4.11.0
export CC=clang
export CXX=clang++
export EGL_PLATFORM=surfaceless


seed=$SLURM_ARRAY_TASK_ID


python random_walk.py --seed $seed
