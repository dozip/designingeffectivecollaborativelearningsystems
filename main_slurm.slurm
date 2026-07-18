#!/bin/bash -l

#SBATCH --job-name=without_zero_shot_multi_product_runs_5_part_2_lambda_075_tau_0
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/tsfm-%j.out
#SBATCH --error=logs/tsfm-%j.err
#SBATCH --export=NONE

set -euo pipefail

unset SLURM_EXPORT_ENV

module purge
module load python

PROJECT="/home/hpc/btr0/btr0106h/projects/designingeffectivecollaborativelearningsystems"
ENV_PATH="$WORK/venvs/numpy2-flax-test"

EXPECTED_GPUS=8
THREADS_PER_GPU=8

cd "$PROJECT"

if [[ ! -x "$ENV_PATH/bin/python" ]]; then
    echo "Python environment not found: $ENV_PATH" >&2
    exit 1
fi

mkdir -p logs

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

export HF_HOME="$WORK/huggingface"
mkdir -p "$HF_HOME"

export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

# JAX auf den zugewiesenen CUDA-GPUs ausführen.
export JAX_PLATFORMS=cuda

# Verhindert, dass jeder JAX-Prozess sofort fast den gesamten
# GPU-Speicher der ihm zugewiesenen RTX 3080 reserviert.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Kann Speicherfragmentierung bei mehreren JAX-Prozessen reduzieren.
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Es laufen bis zu acht Experimentprozesse gleichzeitig:
# ein Prozess pro RTX 3080. Daher erhält jeder Prozess rechnerisch
# acht der insgesamt 64 zugewiesenen physischen CPU-Kerne.
export OMP_NUM_THREADS="$THREADS_PER_GPU"
export MKL_NUM_THREADS="$THREADS_PER_GPU"
export OPENBLAS_NUM_THREADS="$THREADS_PER_GPU"
export NUMEXPR_NUM_THREADS="$THREADS_PER_GPU"

# srun soll die vollständige CPU-Zuweisung des Slurm-Tasks erhalten.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

echo "============================================================"
echo "Node:                  $(hostname)"
echo "Working directory:     $(pwd)"
echo "Python:                 $ENV_PATH/bin/python"
echo "SLURM_JOB_ID:           ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_GPUS:         ${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS:        ${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES:   ${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_CPUS_PER_TASK:    ${SLURM_CPUS_PER_TASK:-}"
echo "OMP_NUM_THREADS:        ${OMP_NUM_THREADS:-}"
echo "Expected GPUs:          $EXPECTED_GPUS"
echo "============================================================"

nvidia-smi

# Prüfen, ob JAX wirklich alle acht zugewiesenen RTX-3080-GPUs sieht.
EXPECTED_GPUS="$EXPECTED_GPUS" "$ENV_PATH/bin/python" -c '
import os
import jax

expected = int(os.environ["EXPECTED_GPUS"])
devices = jax.devices()

print("JAX version:", jax.__version__)
print("JAX devices:", devices)
print("Number of JAX devices:", len(devices))

if len(devices) != expected:
    raise RuntimeError(
        f"Expected {expected} JAX device(s), but found {len(devices)}"
    )

if any(device.platform != "gpu" for device in devices):
    raise RuntimeError(
        f"Expected only GPU devices, but found: {devices}"
    )
'

srun --ntasks=1 \
    "$ENV_PATH/bin/python" -u main_slurm.py \
    --gpu-allocation slurm \
    --name without_zero_shot_multi_product_runs_5_part_2_lambda_075_tau_0
