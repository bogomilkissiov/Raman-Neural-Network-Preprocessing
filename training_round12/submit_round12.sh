#!/bin/bash
#SBATCH -J round12_v3_4_8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --open-mode=append
#SBATCH -o training_round12/slurm-%x-%j.out
#SBATCH -e training_round12/slurm-%x-%j.err

# Optional OSCAR flags (uncomment/edit if needed):
##SBATCH -p gpu-he                 # For A100/H100/L40s high-end GPUs (if allocated)
##SBATCH --constraint=ampere       # Target RTX3090 / A40 / A5000 / A6000 / A100 class GPUs
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=<USERID>@brown.edu

set -euo pipefail

# ---------------------------------------------------------------------
# 1. Environment Setup (per Brown CCV OSCAR documentation)
# ---------------------------------------------------------------------
module purge
unset LD_LIBRARY_PATH
module load cudnn cuda 2>/dev/null || module load cuda

# Submit from project root (`raman preprocessing`), or override REPO_DIR
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"

VENV_DIR="${VENV_DIR:-$HOME/pytorch.venv}"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found at '$VENV_DIR'." >&2
    echo "Create it on an interactive GPU node first (see training_round12/OSCAR_RUNBOOK.md)." >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Optional overrides if datasets live in ~/scratch or ~/data
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-training_round6/training_data6}"
TEST_DATA_DIR="${TEST_DATA_DIR:-training_round6/testing_data6}"
OUT_DIR="${OUT_DIR:-training_round12}"

echo "====================================================================="
echo "OSCAR Slurm Job: ${SLURM_JOB_ID:-local} | Node: $(hostname)"
echo "Repo Dir:        $REPO_DIR"
echo "Train Data Dir:  $TRAIN_DATA_DIR"
echo "Test Data Dir:   $TEST_DATA_DIR"
nvidia-smi || true
echo "====================================================================="

# ---------------------------------------------------------------------
# 2. Train PolyGaussNet V3.4 (8-Section) for 97 Epochs (Exact Round 6 Regime)
#    Automatically resumes from training_checkpoint12.pth if present.
# ---------------------------------------------------------------------
srun python -u training_round12/train_round12.py \
    --epochs 97 \
    --device cuda \
    --data-cache auto \
    --data-dir "$TRAIN_DATA_DIR" \
    --output-dir "$OUT_DIR"

# ---------------------------------------------------------------------
# 3. Head-to-Head Testing (Conventional vs V3 Round 6 vs V3.4-8sec R12)
#    `--require-epoch 97` safely skips evaluation if job was preempted early.
# ---------------------------------------------------------------------
srun python -u training_round12/testing12.py \
    --require-epoch 97 \
    --device cuda \
    --data-dir "$TEST_DATA_DIR" \
    --v3-model training_round6/polygaussnet6.pth \
    --v3-ckpt training_round6/training_checkpoint6.pth \
    --output-dir "$OUT_DIR" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-8}"

echo "Round 12 Slurm job finished at $(date)."
