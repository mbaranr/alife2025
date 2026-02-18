#!/bin/bash
#SBATCH --job-name=dgca_run_%A_%a  # Job name
#SBATCH --output=logs/run_%A_%a.out
#SBATCH --error=logs/run_%A_%a.err   
#SBATCH --array=0-149%10  # 150 runs, max 10 at a time
#SBATCH --ntasks=1
#SBATCH --constraint=avx2
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2G

source /path/to/conda/activate
conda activate dgca_tasks

python3 /path/to/run.py --run_id $SLURM_ARRAY_TASK_ID