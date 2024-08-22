#BSUB -J test
#BSUB -e test.err
#BSUB -o test.out
#BSUB -q scafellpikeGPU
#BSUB -n 1
#BSUB -R "span[ptile=32]"
#BSUB -gpu "num=1"
#BSUB -W 12:00

python create_restart.py > output.txt
#python convert_netCDF2png.py > output.txt
#python plot_differences.py > output.txt

