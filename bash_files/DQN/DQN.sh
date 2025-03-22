#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-machado
#SBATCH --mail-user=mamoham3@ualberta.ca
#SBATCH --job-name=DQN_Exp_6
#SBATCH --output=DQN_Exp_6.out
#SBATCH --error=DQN_Exp_6.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=5:00:00
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
# Define the Singularity image



# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip3 install git+https://github.com/prabhatnagarajan/pfrl.git@gymnasium_support
# pip3 install imageio

# cd YOUR_DIR/AgarLE

# python3 setup.py install --user 

# python3 install -r /home/mayman/projects/def-machado/mayman/AgarLE-benchmark/requirements.txt

cd /home/mayman/projects/def-machado/mayman/AgarLE-benchmark

python DQN_full_action_set.py --outdir "/home/mayman/Results/DQN_mode_2"
