#!/bin/bash
#SBATCH --time=00-23:59:00
#SBATCH --mem=1000G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=vamp_sim_1
#SBATCH --reservation=mondegrp_99
#SBATCH --nodelist=zeta253
#SBATCH --output=/nfs/scistore13/mondegrp/asharipo/vampW/outputs/%A-_run_%a.log

loc=/nfs/scistore13/mondegrp/asharipo/vampW

module load python/3.9
source ${loc}/venv/bin/activate


srun python3 $loc/run_vamp_bed.py
