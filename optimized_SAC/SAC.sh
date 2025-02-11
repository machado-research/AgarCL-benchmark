#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-machado
#SBATCH --mail-user=mamoham3@ualberta.ca
#SBATCH --job-name=SAC_exp1
#SBATCH --output=SAC_exp1.out
#SBATCH --error=SAC_exp1.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=23:00:00
#SBATCH --cpu-freq=Performance
#SBATCH --array=1-10
# module load apptainer
# module --force purge 
module load StdEnv/202
module load clang/17.0.6
module load cmake
module load python/3.11.5
module load scipy-stack
module load glfw
module load cuda/12.2

export CC=clang
export CXX=clang++
export EGL_PLATFORM=surfaceless
# Define the Singularity image

seed=$SLURM_ARRAY_TASK_ID



pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/prabhatnagarajan/pfrl.git@gymnasium_support
pip install imageio
# pip install opencv-python
pip3 install tenacity
cd /home/mayman/projects/def-machado/mayman/AgarLE

python3 setup.py install --user 

# python3 install -r /home/mayman/projects/def-machado/mayman/AgarLE-benchmark/requirements.txt

cd /home/mayman/projects/def-machado/mayman/AgarLE-benchmark

python /home/mayman/projects/def-machado/mayman/AgarLE-benchmark/optimized_SAC/optimized_SAC.py --seed $seed


