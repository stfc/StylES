#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=test
#SBATCH --error=test.err
#SBATCH --output=test.out
#SBATCH --time=24:00:00
#SBATCH --partition=small
#SBATCH --gres=gpu:1

python main.py > output.txt

