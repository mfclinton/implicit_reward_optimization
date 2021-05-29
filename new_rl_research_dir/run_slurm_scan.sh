#!/bin/bash
#
#SBATCH --job-name=sweep
#SBATCH --output=outputs/stdoutput/res_%j.txt  # output file
#SBATCH -e outputs/stdoutput/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=0-5:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=800    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mfclinton@umass.edu
#
#SBATCH --array=0-4

export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=32


python -m Src.run -cn config_GW -m

sleep 1
