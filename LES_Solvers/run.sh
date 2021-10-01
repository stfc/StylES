#BSUB -J test
#BSUB -e test.err
#BSUB -o test.out
#BSUB -q scafellpikeGPU
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -gpu "num=1"
#BSUB -W 24:00

module load python3/anaconda
module load cuda/10.2
source ~/.bashrc
export CONDA_PKGS_DIRS=/lustre/scafellpike/local/HT03807/jxc05/jxc74-jxc05/downloads/
export CONDA_ENVS_PATH=/lustre/scafellpike/local/HT03807/jxc05/jxc74-jxc05/.condaenv/tf_gpu_2.2
export PYTHONPATH=/lustre/scafellpike/local/apps/gcc/xalt/1.1.2/site:/lustre/scafellpike/local/apps/gcc/xalt/1.1.2/libexec
conda activate /lustre/scafellpike/local/HT03807/jxc05/jxc74-jxc05/.condaenv/tf_gpu_2.2
export CUDA_VISIBLE_DEVICES=0

python LES_solver.py > output.txt
