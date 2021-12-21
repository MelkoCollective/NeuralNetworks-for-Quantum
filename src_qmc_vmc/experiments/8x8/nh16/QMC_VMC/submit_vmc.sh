#!/bin/bash
#SBATCH -t 1-05:00
#SBATCH --mem=5000
#SBATCH --account=def-rgmelko

#SBATCH --mail-user=sczischek@uwaterloo.ca
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.8.2
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt

python run_vmc.py
