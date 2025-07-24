#!/bin/bash

#SBATCH --partition=iris-hi
#SBATCH --chdir=/iris/u/rheamal/homer
#SBATCH --output=slurm/placeonly-%j.out
#SBATCH --error=slurm/placeonly-%j.err
#SBATCH --job-name=place
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=iris
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1

echo "Starting job on $(hostname) at $(date)"
echo "Activating environment..."

# Load environment
source ~/.bashrc
conda activate tidybot2

python scripts/train_waypoint.py --config_path cfgs/waypoint/cube_lh_placeonly_wbc.yaml

echo "Job completed at $(date)"
