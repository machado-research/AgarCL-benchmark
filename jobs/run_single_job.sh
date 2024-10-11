#!/bin/bash 
#SBATCH --account=rrg-whitem 
#SBATCH --time=3:00:00 
#SBATCH --cpus-per-task=8 
#SBATCH --mem-per-cpu=10000

cd ~/projects/def-whitem/annahakh/AgarLE-benchmark/ 
source /venv/bin/activate 
wandb offline
parallel  < $1 