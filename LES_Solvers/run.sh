#BSUB -J test
#BSUB -e test.err
#BSUB -o test.out
#BSUB -q scafellpikeGPUdev
#BSUB -n 1
#BSUB -R "span[ptile=32]"
#BSUB -gpu "num=1"
#BSUB -W 144:00

python LES_solver_staggered.py > output.txt
