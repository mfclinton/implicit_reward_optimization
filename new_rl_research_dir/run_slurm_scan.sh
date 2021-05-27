#!/bin/bash
#
#SBATCH --job-name=sweep
#SBATCH --output=outputs/stdoutput/res_%j.txt  # output file
#SBATCH -e outputs/stdoutput/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=12
#SBATCH --time=0-5:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=100    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mfclinton@umass.edu

export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2


python -m Src.run -cn config_GW -m

sleep 1