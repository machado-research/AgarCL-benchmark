#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-machado
#SBATCH --mail-user=mamoham3@ualberta.ca
#SBATCH --job-name=SAC_mode_2
#SBATCH --output=SAC_mode_2.out
#SBATCH --error=SAC_mode_2.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=23:00:00
#SBATCH --cpu-freq=Performance
#SBATCH --array=1-10

module load clang/17.0.6
module load cmake
# module load python/3.10.13 #Noise version
module load python/3.11.5
module load scipy-stack
module load glfw
module load cuda/12.2
module load opencv/4.11.0

export CC=clang
export CXX=clang++
export EGL_PLATFORM=surfaceless
# Define the Singularity image

seed=$SLURM_ARRAY_TASK_ID


# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# pip install imageio
# # pip install opencv-python
# pip3 install tenacity
# cd /home/mayman/projects/def-machado/mayman/AgarLE

# python setup.py install --user 

# python3 install -r /home/mayman/projects/def-machado/mayman/AgarLE-benchmark/requirements.txt


cd /home/mayman/projects/def-machado/mayman/agarle_bench/AgarLE-benchmark

# python train_SAC.py --seed $seed
python SAC_full_action_set.py --reward "reward_gym" --lr 0.00001 --seed $seed --outdir '/home/mayman/Results/SAC_mode_2' --soft-update-tau 0.001 --max-grad-norm 0.7