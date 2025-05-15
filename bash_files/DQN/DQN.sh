#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --account=XX
#SBATCH --mail-user=XX
#SBATCH --job-name=XX
#SBATCH --output=XX.out
#SBATCH --error=XX.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=23:00:00
#SBATCH --cpu-freq=Performance
# module load apptainer
# module --force purge 
module load clang/17.0.6
module load cmake
module load python/3.11.5 
module load scipy-stack
module load glfw
module load cuda/12.2
module load opencv/4.11.0

export CC=clang
export CXX=clang++
export EGL_PLATFORM=surfaceless

cd CODE_DIR

python DQN_full_action_set.py --outdir "OUTPUT_DIR"
