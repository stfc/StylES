#BSUB -J test
#BSUB -e test.err
#BSUB -o test.out
#BSUB -q scafellpikeSKL
#BSUB -n 128 
#BSUB -R "span[ptile=32]"
#BSUB -W 48:00

mpirun -np 128 ./hasegawa-wakatani
