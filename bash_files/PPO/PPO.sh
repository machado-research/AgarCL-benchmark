#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --account=XX
#SBATCH --job-name=XX
#SBATCH --output=XX.out
#SBATCH --error=XX.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=15:00:00
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



cd CODE_DIR


python PPO_multi_heads_full_action.py --reward "min_max" --lr 1e-5 --entropy-coef 0.05 --clip-eps 0.4 --value-func-coef 0.9 --max-grad-norm 0.5  --outdir "OUTPUT_DIR"
