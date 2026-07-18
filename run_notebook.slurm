#!/bin/bash -l

#SBATCH --job-name=eval-synthetic-data
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/hpc/btr0/btr0106h/projects/designingeffectivecollaborativelearningsystems
#SBATCH --output=/home/hpc/btr0/btr0106h/projects/designingeffectivecollaborativelearningsystems/Results/evaluation_synthetic_data-%j.out
#SBATCH --error=/home/hpc/btr0/btr0106h/projects/designingeffectivecollaborativelearningsystems/Results/evaluation_synthetic_data-%j.err
#SBATCH --export=NONE

set -euo pipefail

unset SLURM_EXPORT_ENV

PROJECT_ROOT="/home/hpc/btr0/btr0106h/projects/designingeffectivecollaborativelearningsystems"
# NOTEBOOK="$PROJECT_ROOT/evaluation_synthetic_data.ipynb"
NOTEBOOK="$PROJECT_ROOT/market_configuration_paired_evaluation_all_metrics_repaired_export_lamda_075_tau_0.ipynb"
NOTEBOOK="$PROJECT_ROOT/market_configuration_paired_evaluation_all_metrics_repaired_export_lamda_1_tau_0.ipynb"
EXECUTED_NOTEBOOK_DIR="$PROJECT_ROOT/Results/Executed_Notebooks"

module purge
module load python

source "$WORK/venvs/data-analysis/bin/activate"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$EXECUTED_NOTEBOOK_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname -f)"
echo "Python: $(which python)"
echo "Notebook: $NOTEBOOK"

python -m jupyter nbconvert \
    --to notebook \
    --execute "$NOTEBOOK" \
    --output "evaluation_synthetic_data_executed_${SLURM_JOB_ID}.ipynb" \
    --output-dir "$EXECUTED_NOTEBOOK_DIR" \
    --ExecutePreprocessor.kernel_name=data-analysis \
    --ExecutePreprocessor.timeout=-1

echo "Notebook completed successfully."
echo "Executed notebook:"
echo "$EXECUTED_NOTEBOOK_DIR/evaluation_synthetic_data_executed_${SLURM_JOB_ID}.ipynb"

echo "Analysis results should be under:"
echo "$PROJECT_ROOT/Results/Evaluation_Leadtime/"