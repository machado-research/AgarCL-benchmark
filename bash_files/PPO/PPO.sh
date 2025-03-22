#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-machado
#SBATCH --mail-user=mamoham3@ualberta.ca
#SBATCH --job-name=ppo_6
#SBATCH --output=ppo_6.out
#SBATCH --error=ppo_6.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=5:00:00
#SBATCH --cpu-freq=Performance

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
# Define the Singularity image



cd /home/mayman/projects/def-machado/mayman/AgarLE-benchmark


python PPO_multi_heads_full_action.py --reward "min_max" --lr 1e-5 --entropy-coef 0.05 --clip-eps 0.4 --value-func-coef 0.9 --max-grad-norm 0.5  --outdir "/home/mayman/Results/PPO_mode_2"