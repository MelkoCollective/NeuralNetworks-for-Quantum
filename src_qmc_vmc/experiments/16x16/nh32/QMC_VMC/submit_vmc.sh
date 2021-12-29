#!/bin/bash
#SBATCH -t 7-00:00
#SBATCH --mem=5000
#SBATCH --account=rrg-rgmelko-ab

#SBATCH --mail-user=msmoss@uwaterloo.ca
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements_vmc.txt

python run_vmc.py
