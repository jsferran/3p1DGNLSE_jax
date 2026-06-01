#!/bin/bash
#SBATCH --job-name=grin3d_nt_conv
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/grin3d_nt_conv_%j.out
#SBATCH --error=logs/grin3d_nt_conv_%j.err

# Runs forward passes at Nt in {128, 256, 512, 1024, 2048} sequentially.
# Results written to $OUTDIR; convergence figure saved as nt_convergence.png.

CODE_DIR="$HOME/project_pi_lgw23/jf2447/3p1DGNLSE_jax"
OUTDIR="$HOME/scratch_pi_lgw23/jf2447/grin3d_nt_convergence"

mkdir -p logs "$OUTDIR"

module purge
module load miniconda
conda activate grin_env

python "$CODE_DIR/grin3d_nt_convergence.py" \
    --code-dir    "$CODE_DIR" \
    --mode-folder "$CODE_DIR/grin_modes_300_50um" \
    --n-modes     100 \
    --n-windows   220 \
    --dz-frac     0.05 \
    --lz-cm       30.0 \
    --p-mult      2.0 \
    --outdir      "$OUTDIR"
