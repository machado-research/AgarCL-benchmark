#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-machado
#SBATCH --mail-user=mamoham3@ualberta.ca
#SBATCH --job-name=rr_exp4_noise
#SBATCH --output=rr_exp_4_noise.out
#SBATCH --error=rr_exp_4_noise.err
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


# cd /home/mayman/projects/def-machado/mayman/AgarLE

# python3 setup.py install --user 

python /home/mayman/projects/def-machado/mayman/AgarLE/bench/agarle_bench.py --seed $seed
