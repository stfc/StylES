#!/bin/bash
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=1

##SBATCH --mem=<mem_per_node>
#SBATCH --partition=boost_fua_prod
#SBATCH --qos=normal
#SBATCH --job-name=test
#SBATCH --mem=123000 
#SBATCH --err=test.err 
#SBATCH --out=test.out 
#SBATCH --account=FUAL8_UKAEA_ML
#SBATCH --gres=gpu:1

python convert_netCDF2png.py
#python create_restart.py
