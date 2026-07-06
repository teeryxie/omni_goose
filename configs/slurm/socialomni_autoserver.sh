#!/bin/bash
#SBATCH --job-name=socialomni
#SBATCH --partition=gpu300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail
export PYTHONPATH=/mnt/c/Users/HP/Documents/Codex/2026-06-29/ni/src:/mnt/c/Users/HP/Documents/Codex/2026-06-29/ni:

cd "$SLURM_SUBMIT_DIR"
exec bash "configs/slurm/socialomni_autoserver.slurm"
