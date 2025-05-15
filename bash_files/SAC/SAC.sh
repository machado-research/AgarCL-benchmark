#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --account=XX
#SBATCH --mail-user=XX
#SBATCH --job-name=XX
#SBATCH --output=XX.out
#SBATCH --error=XX.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=23:00:00
#SBATCH --cpu-freq=Performance

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
# Define the Singularity image

# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# pip install imageio
# # pip install opencv-python
# pip3 install tenacity
# cd /home/mayman/projects/def-machado/mayman/AgarLE

# python setup.py install --user 

# python3 install -r /home/mayman/projects/def-machado/mayman/AgarLE-benchmark/requirements.txt


cd /home/mayman/projects/aip-machado/mayman/AgarLE-benchmark

# python train_SAC.py --seed $seed
python SAC_full_action_set.py --reward "reward_gym" --lr 0.00001 --seed 0 --outdir '/home/mayman/Results/SAC_mode_6' --soft-update-tau 0.001 --max-grad-norm 0.7