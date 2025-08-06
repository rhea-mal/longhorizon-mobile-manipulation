#!/bin/bash

#SBATCH --partition=iris-hi
#SBATCH --chdir=/iris/u/rheamal/homer
#SBATCH --output=slurm/drawer_1dense.out
#SBATCH --error=slurm/drawer_1dense.err
#SBATCH --job-name=1dense
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=20G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=iris
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1

echo "Starting job on $(hostname) at $(date)"
echo "Activating environment..."

# Load environment
source ~/.bashrc
conda activate tidybot2
pip install rich
python scripts/train_dense.py --config_path cfgs/dense/drawer_longhorizon.yaml
echo "drawer 1 single dense policy"
echo "Job completed at $(date)"
