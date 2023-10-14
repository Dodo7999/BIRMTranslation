#!/bin/sh

#SBATCH --job-name=bsl_shift
#SBATCH --error=/userspace/bma/bsl_shift_err.log
#SBATCH --output=/userspace/bma/bsl_shift.log
#SBATCH --partition=a100
#SBATCH --nodelist=ngpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-socket=1
#SBATCH --no-requeue
#SBATCH -o bsl_shift.log

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

