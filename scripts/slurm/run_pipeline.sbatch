#!/bin/bash
#SBATCH --job-name=evacmob
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge || true
# If using conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate evacmob

mkdir -p logs
python scripts/run_pipeline.py
