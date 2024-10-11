#!/bin/bash 
#SBATCH --account=rrg-whitem 
#SBATCH --time=01:00:00 
for FILE in /home/annahakh/projects/def-whitem/annahakh/AgarLE-benchmark/jobs/*.txt 
     do
         sbatch ~/projects/def-whitem/annahakh/AgarLE-benchmark/jobs/run_single_job.sh $FILE; 
     done