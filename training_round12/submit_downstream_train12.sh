#!/bin/bash
#SBATCH -J round12_downstream
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
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
BACTERIA_DIR="${BACTERIA_DIR:-datasets/bacteria-ID}"
MLROD_DIR="${MLROD_DIR:-datasets/MLROD}"
PGNET_MODEL="${PGNET_MODEL:-training_round12/polygaussnet12.pth}"
PGNET_CKPT="${PGNET_CKPT:-training_round12/training_checkpoint12.pth}"
OUT_ROOT="${OUT_ROOT:-training_round12}"
N_JOBS="${SLURM_CPUS_PER_TASK:-8}"

echo "====================================================================="
echo "OSCAR Slurm Job: ${SLURM_JOB_ID:-local} | Node: $(hostname)"
echo "Repo Dir:        $REPO_DIR"
echo "Bacteria-ID Dir: $BACTERIA_DIR"
echo "MLROD Dir:       $MLROD_DIR"
echo "PGNet Model:     $PGNET_MODEL"
echo "Output Root:     $OUT_ROOT"
echo "CPU Workers:     $N_JOBS"
nvidia-smi || true
echo "====================================================================="

# ---------------------------------------------------------------------
# 2. Verify Datasets & Python Dependencies Before Starting
# ---------------------------------------------------------------------
for req_file in \
    "$BACTERIA_DIR/X_reference.npy" \
    "$BACTERIA_DIR/X_finetune.npy" \
    "$BACTERIA_DIR/X_test.npy" \
    "$MLROD_DIR/X.npy" \
    "$MLROD_DIR/y.npy" \
    "$MLROD_DIR/wavenumbers.npy" \
    "$MLROD_DIR/class_names.npy"; do
    if [ ! -f "$req_file" ]; then
        echo "[ERROR] Required dataset file missing: '$req_file'" >&2
        echo "Sync datasets/bacteria-ID and datasets/MLROD to OSCAR before submitting." >&2
        exit 1
    fi
done

python -c "import torch, sklearn, xgboost, joblib; print('Dependencies verified: torch, sklearn, xgboost, joblib')" || {
    echo "[ERROR] Missing Python dependencies in '$VENV_DIR'. Run: pip install scikit-learn xgboost joblib" >&2
    exit 1
}

# ---------------------------------------------------------------------
# 3. Program 1: Downstream Classifier Training (18 Runs)
#    - Normalizes all spectra to [0, 1] per spectrum before preprocessing
#    - Runs raw, conventional, and pgnet_v34 preprocessing & caches val/test
#    - Trains raw to convergence, then matches exact epochs/rounds for
#      conventional and pgnet_v34 across Linear, XGBoost, and ResNet1D
# ---------------------------------------------------------------------
srun python -u training_round12/train_downstream12.py \
    --bacteria-dir "$BACTERIA_DIR" \
    --mlrod-dir "$MLROD_DIR" \
    --mlrod-subset pure12 \
    --pgnet-model "$PGNET_MODEL" \
    --pgnet-ckpt "$PGNET_CKPT" \
    --output-root "$OUT_ROOT" \
    --device cuda \
    --xgb-device auto \
    --n-jobs "$N_JOBS" \
    --cache-train \
    --use-cache

# ---------------------------------------------------------------------
# 4. Program 2: Downstream Test Evaluation & Report Generation
#    - `--require-all-runs` safely defers evaluation if job was preempted early
#    - Loads saved model weights + cached preprocessed test sets
#    - Generates per-run .txt reports, head-to-head .txt comparisons,
#      McNemar significance tables, .csv/.json/.npz archives, and plots
# ---------------------------------------------------------------------
srun python -u training_round12/test_downstream12.py \
    --require-all-runs \
    --output-root "$OUT_ROOT" \
    --bacteria-dir "$BACTERIA_DIR" \
    --mlrod-dir "$MLROD_DIR" \
    --pgnet-model "$PGNET_MODEL" \
    --pgnet-ckpt "$PGNET_CKPT" \
    --device cuda \
    --checkpoint-mode best \
    --n-jobs "$N_JOBS"

echo "Round 12 downstream training & evaluation finished at $(date)."
