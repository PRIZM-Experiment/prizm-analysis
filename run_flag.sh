#!/bin/bash
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00

cd $SLURM_SUBMIT_DIR

module load gcc/9.2.0 intel/2019u5 intelmpi/2019u5

source $HOME/.virtualenvs/prizmenv/bin/activate

arg1='70MHz'
arg2='NS'
arg3='2021'

mpirun -n 10 python run_flag.py "$arg1" "$arg2" "$arg3"

deactivate
