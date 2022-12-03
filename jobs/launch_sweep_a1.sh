#!/bin/bash -l
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks/node
#SBATCH --partition=cpu           # The partion name: gpu or cpu
#SBATCH --job-name=train_wnn_a1   # job name
#SBATCH --exclusive               # Reserve this node only for you
#SBATCH --account=vinicius.mello  # account name


cd $HOME/Workspace/Projects/weightless-neural-networks

singularity exec \
    --env-file $PWD/.env \
    -B $PWD:/app \
    weightless-neural-networks_latest.sif \
    python assignments/a1/sweep.py --sweep 