# Round 12 OSCAR Cluster Runbook

This directory (`training_round12`) trains **PolyGaussNet V3.4 (8-section)** (`num_quadrants=8`, 108,394 parameters) for **97 epochs** under the **exact Round 6 training regime and dataset**, and then runs a 3-way head-to-head benchmark against **Round 6 V3 (`polygaussnet6.pth`)** and the **Conventional Pipeline (`test_files/conventional/pre.py`)** on `training_round6/testing_data6`.

## What Was Optimized for Brown's OSCAR

Based on `/Users/bogiekissiov/Desktop/OSCAR/oscar_docs_complete.md`:

1. **Eliminated Repeated Storage I/O (`--data-cache auto`):**
   - `training_round6/training_data6` is 40 `.npz` chunks (~8.0 GiB compressed, ~9.5 GiB uncompressed float32).
   - Instead of re-reading and unzipping 40 files from shared VAST storage on every epoch (3,880 decompressions over 97 epochs), `train_round12.py` preloads all chunks **once** in parallel at startup:
     - On GPUs with sufficient VRAM headroom (total dataset + 8 GiB free VRAM, e.g., V100 32GB, A40 45GB, A6000/L40s 48GB, A100/H100 80GB), it caches tensors directly on **GPU VRAM** (zero host-to-device copies per epoch).
     - On 24 GiB GPUs (e.g., RTX 3090, A5000, Titan RTX), it automatically caches in **pinned CPU RAM** and streams via async `non_blocking=True` transfers.
2. **Removed Python `DataLoader` & Per-Chunk GC Stalls:**
   - Replaced per-sample Python `DataLoader` iteration with direct `torch.randperm` tensor indexing (mathematically identical per-chunk shuffling at `batch_size=512`).
   - Removed the 40×/epoch `gc.collect()` + `empty_cache()` synchronization stalls.
3. **Ampere+ CUDA Math Throughput:**
   - Enables `TF32` (`float32_matmul_precision("high")`, `cudnn.allow_tf32=True`) and `cudnn.benchmark=True`.
   - Optional flags `--amp` (BF16 autocast with FP32 loss) and `--compile` (`torch.compile(dynamic=True)`) are available if desired, kept off by default to match Round 6 FP32 math exactly.
4. **Slurm Preemption Safety & Atomic Checkpointing:**
   - `submit_round12.sh` uses `#SBATCH --signal=B:TERM@60` and `#SBATCH --open-mode=append`.
   - `train_round12.py` handles `SIGTERM`, finishes the active epoch cleanly, writes checkpoints atomically (`.tmp` → `os.replace`), and resumes deterministically from `training_checkpoint12.pth`.
   - `testing12.py` uses `--require-epoch 97` in Slurm so testing only runs once all 97 epochs finish, sets `matplotlib.use("Agg")` for headless nodes, and sets `OMP_NUM_THREADS=1` inside multiprocessing workers.

---

## Step-by-Step Deployment on OSCAR

### 1. Transfer Workspace to OSCAR (`~/scratch`)

Remember OSCAR quotas from the docs (`~/home` = 20 GB, `~/scratch` = 512 GB short-term, `~/data` = 256 GB persistent): **put `training_data6` (~8 GB) in `~/scratch` or `~/data`, NOT `~/home`**.

From your Mac terminal (skipping huge historical round datasets except Round 6):

```bash
rsync -azvp --progress \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude 'training_round[1-5]/' \
  --exclude 'training_round[7-9]/' \
  --exclude 'training_round10/' \
  --exclude 'training_round11/' \
  "/Users/bogiekissiov/Desktop/raman preprocessing/" \
  <USERNAME>@ssh.ccv.brown.edu:~/scratch/raman_preprocessing/
```

### 2. Build the PyTorch Environment Once (Interactive GPU Node)

Per OSCAR docs, build on an Ampere GPU node with `module purge && unset LD_LIBRARY_PATH`:

```bash
ssh <USERNAME>@ssh.ccv.brown.edu
interact -q gpu -g 1 -f ampere -m 20g -n 4

nvidia-smi
module purge
unset LD_LIBRARY_PATH
module load cudnn cuda

python -m venv ~/pytorch.venv
source ~/pytorch.venv/bin/activate
pip install --upgrade pip
pip install -r ~/scratch/raman_preprocessing/requirements.txt

# Verify CUDA + conventional dependencies
python -c "import torch, spectra_class; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit
```

*(Note: If compute/login nodes cannot reach GitHub for `Raman-Plotting-Pipeline`, you can `rsync` `/Users/bogiekissiov/Desktop/raman plotting` to `~/scratch/raman plotting` alongside `raman_preprocessing`—`testing12.py` automatically checks `../raman plotting` too.)*

### 3. Submit Training + Testing Job

From `~/scratch/raman_preprocessing`:

```bash
cd ~/scratch/raman_preprocessing
chmod +x training_round12/submit_round12.sh
sbatch training_round12/submit_round12.sh
```

Monitor progress:

```bash
myq
tail -f training_round12/training_history12.txt
```

If the job ever hits walltime or is preempted before epoch 97, simply re-run `sbatch training_round12/submit_round12.sh`—it automatically resumes from `training_round12/training_checkpoint12.pth` and runs `testing12.py` once Epoch 97 finishes.

### 4. Archive Long-Term Results & Copy Back to Mac

OSCAR purges `~/scratch` files unaccessed for 30 days. Copy finished weights and reports to `~/data` and your Mac:

```bash
# On OSCAR:
mkdir -p ~/data/raman_round12
cp training_round12/{polygaussnet12.pth,training_checkpoint12.pth,training_history12.txt,validation_results_round12.txt,*.png} ~/data/raman_round12/

# On Mac:
rsync -azvp "<USERNAME>@ssh.ccv.brown.edu:~/scratch/raman_preprocessing/training_round12/" \
  "/Users/bogiekissiov/Desktop/raman preprocessing/training_round12/"
```
