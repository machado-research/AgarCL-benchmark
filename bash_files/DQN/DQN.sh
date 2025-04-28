#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --account=aip-machado
#SBATCH --mail-user=mamoham3@ualberta.ca
#SBATCH --job-name=DQN_Exp_3
#SBATCH --output=DQN_Exp_3.out
#SBATCH --error=DQN_Exp_3.err
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

cd /home/mayman/projects/aip-machado/mayman/AgarLE-benchmark

python DQN_full_action_set.py --outdir "/home/mayman/Results/DQN_mode_2"
