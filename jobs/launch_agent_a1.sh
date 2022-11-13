#!/bin/bash -l
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks/node
#SBATCH --cpus-per-task=10        # Number of threads/task
#SBATCH --partition=cpu           # The partion name: gpu or cpu
#SBATCH --job-name=train_wnn_a1   # job name
#SBATCH --exclusive               # Reserve this node only for you
#SBATCH --account=vinicius.mello  # account name


cd $HOME/Workspace/Projects/weightless-neural-networks

singularity exec \
    --nv \
    --env-file $PWD/.env \
    -B $PWD/.env:/home/sonar/code/.env \
    -B $PWD/src:/home/sonar/code/src \
    -B $PWD/data:/home/sonar/code/data \
    -B $PWD/training:/home/sonar/code/training \
    -B $PWD/notebooks:/home/sonar/code/notebooks \
    -B $PWD/scripts:/home/sonar/code/scripts \
    weightless_neural_networks.sif \
    python assignments/a1/train.py \
    --agent \
    --sweep_id $WANDB_SWEEP_ID