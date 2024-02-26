#!/bin/sh

#SBATCH --job-name=max_birm
#SBATCH --error=/userspace/bma/gpt2_birm_max_err.log
#SBATCH --output=/userspace/bma/gpt2_birm_max.log
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-socket=1
#SBATCH --no-requeue
#SBATCH -o gpt2_birm_max.log

# General SLURM Parameters
echo "# SLURM_JOBID  = ${SLURM_JOBID}"
echo "# SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "# SLURM_NNODES = ${SLURM_NNODES}"
echo "# SLURM_NTASKS = ${SLURM_NTASKS}"
echo "# SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "# SLURMTMPDIR = ${SLURMTMPDIR}"
echo "# Submission directory = ${SLURM_SUBMIT_DIR}"

# conda
. "/userspace/bma/miniconda3/etc/profile.d/conda.sh"
conda activate pycuda
export HF_HOME=/userspace/bma/huggingface_cached
export TRANSFORMERS_CACHE=/userspace/bma/.transformersCache
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
nvidia-smi -L
nvcc -V
python -V
python -c "import torch, transformers, datasets, tokenizers; print(f'torch.version = {torch.__version__}, CUDA = {torch.cuda.is_available()}, transformers.version = {transformers.__version__}, datasets.version = {datasets.__version__}, tokenizers.version = {tokenizers.__version__}')"

python -u pretraining/something_max.py

