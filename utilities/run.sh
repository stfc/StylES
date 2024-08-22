#BSUB -J test
#BSUB -e test.err
#BSUB -o test.out
#BSUB -q scafellpikeGPU
#BSUB -n 1
#BSUB -R "span[ptile=32]"
#BSUB -gpu "num=1"
#BSUB -W 12:00

python convert_to_SingleChannel.py  > output.txt
#python check_reconstruction.py > output.txt
