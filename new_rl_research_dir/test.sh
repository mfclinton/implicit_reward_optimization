#!/bin/bash
#
#SBATCH --job-name=run
#SBATCH --output=outputs/stdoutput/res_%j.txt  # output file
#SBATCH -e outputs/stdoutput/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to
#
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-5:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=250    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ateeter@umass.edu
#
#SBATCH --array=0-30

export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

echo test

sleep 1
