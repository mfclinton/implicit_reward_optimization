#!/bin/bash
#
#SBATCH --job-name=run1
#SBATCH --output=outputs/stdoutput/res_%j.txt  # output file
#SBATCH -e outputs/stdoutput/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to
#
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-5:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=250    # Memory in MB per cpu allocated
#SBATCH --mail-type=END
#SBATCH --mail-user=mfclinton@umass.edu
#
#SBATCH --array=0-30

export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2


python -m Src.run -cn config_GW seed=$SLURM_ARRAY_TASK_ID num_runs=1 config.name=regular_method config.env.aux_r_id=0

sleep 1
