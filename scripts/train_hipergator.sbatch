#!/bin/bash
# ==============================================================================
# EchoVerify: HiPerGator Slurm GPU Training & Evaluation Job Template
# ==============================================================================
#SBATCH --job-name=echoverify_train
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL@ufl.edu>
#SBATCH --account=<YOUR_HIPERGATOR_ACCOUNT>     # e.g., cis6930, ufrc, etc.
#SBATCH --qos=<YOUR_QOS>                       # e.g., cis6930 or standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=gpu
#SBATCH --gpus=1                               # Requests 1 GPU (e.g. A100 or RTX)
#SBATCH --time=03:00:00                        # Wall-clock runtime limit
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

echo "=== Job Started: $(date) ==="
echo "Node: $(hostname)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------------------------
# 1. User Configuration & Environment Setup
# ------------------------------------------------------------------------------
# Set path to the ASVspoof 2019 LA directory on HiPerGator storage (/blue or /orange)
export ASVSPOOF2019_LA_ROOT="/blue/<YOUR_GROUP>/<YOUR_USER>/data/ASVspoof2019_LA"

# Set project working directory
PROJECT_DIR="/blue/<YOUR_GROUP>/<YOUR_USER>/EchoVerify"
cd "$PROJECT_DIR" || exit 1

mkdir -p logs results checkpoints

# Load HiPerGator modules
module purge
module load conda
module load ffmpeg

# Activate your custom environment (or virtualenv)
# e.g.: conda activate echoverify
# or:   source /blue/<YOUR_GROUP>/<YOUR_USER>/envs/echoverify/bin/activate
source activate echoverify

python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# ------------------------------------------------------------------------------
# 2. Experiment A: Train Raw Baseline CNN
# ------------------------------------------------------------------------------
echo "--- Starting Raw Baseline Training ---"
python src/train.py \
    --data-root "$ASVSPOOF2019_LA_ROOT" \
    --mode raw \
    --epochs 5 \
    --batch-size 32 \
    --lr 0.001 \
    --max-train-samples 2000 \
    --max-dev-samples 1000 \
    --seed 42 \
    --device cuda \
    --output-dir results/raw \
    --checkpoint-dir checkpoints/raw

# ------------------------------------------------------------------------------
# 3. Experiment B: Train Silence-Trimmed CNN
# ------------------------------------------------------------------------------
echo "--- Starting Silence-Trimmed CNN Training ---"
python src/train.py \
    --data-root "$ASVSPOOF2019_LA_ROOT" \
    --mode trimmed \
    --trim-top-db 30.0 \
    --epochs 5 \
    --batch-size 32 \
    --lr 0.001 \
    --max-train-samples 2000 \
    --max-dev-samples 1000 \
    --seed 42 \
    --device cuda \
    --output-dir results/trimmed \
    --checkpoint-dir checkpoints/trimmed

# ------------------------------------------------------------------------------
# 4. Evaluation on Held-Out Eval Partition (if available)
# ------------------------------------------------------------------------------
echo "--- Evaluating on Held-Out Partition ---"
python src/evaluate.py \
    --checkpoint checkpoints/raw/best_model.pt \
    --split eval \
    --data-root "$ASVSPOOF2019_LA_ROOT" \
    --device cuda \
    --output-dir results/raw

python src/evaluate.py \
    --checkpoint checkpoints/trimmed/best_model.pt \
    --split eval \
    --data-root "$ASVSPOOF2019_LA_ROOT" \
    --device cuda \
    --output-dir results/trimmed

echo "=== Job Finished: $(date) ==="

