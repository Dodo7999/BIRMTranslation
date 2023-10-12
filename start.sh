#!/bin/sh

#SBATCH --job-name=w2v_birm_open
#SBATCH --error=/storage0/bi/w2v_birm_open.err
#SBATCH --output=/storage0/bi/w2v_birm_open.log
#SBATCH --partition=a100
#SBATCH --nodelist=ngpu06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-socket=1

# conda
. "/userspace/bma/miniconda3/etc/profile.d/conda.sh"
conda activate pycuda
export TRANSFORMERS_CACHE=/userspace/bma/.transformersCache
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
nvidia-smi -L
nvcc -V
python -V
#conda install --file requirements.txt

python main.py

