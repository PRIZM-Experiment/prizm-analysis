#!/bin/bash
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00

cd $SLURM_SUBMIT_DIR

module load gcc/9.2.0 intel/2019u5 intelmpi/2019u5

source $HOME/.virtualenvs/prizmenv/bin/activate

arg1='70MHz'
arg2='NS'
arg3='2021'
lam='0.01'

calib='shorts'

mpirun -n 10 python run_calib_filter.py "$arg1" "$arg2" "$arg3" "$lam" "$calib"

calib='res100'

mpirun -n 10 python run_calib_filter.py "$arg1" "$arg2" "$arg3" "$lam" "$calib"

calib='res50'

mpirun -n 10 python run_calib_filter.py "$arg1" "$arg2" "$arg3" "$lam" "$calib"

deactivate
