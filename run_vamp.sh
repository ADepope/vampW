#!/bin/bash
#SBATCH --time=00-0:15:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=vamp_sim_1
#SBATCH --reservation=mondegrp_99
#SBATCH --nodelist=zeta253
#SBATCH --output=/nfs/scistore13/mondegrp/asharipo/vampW/outputs/%A-_run_%a.log

loc=/nfs/scistore13/mondegrp/asharipo/vampW

ml python3
source ${loc}/venv/bin/activate


srun python3 $loc/run_vamp.py
