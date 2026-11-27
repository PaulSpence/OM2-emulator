#!/bin/bash
# Run a notebook from notebooks/ headless on one MI250X GCD on Setonix (Pawsey). The Slurm
# counterpart of run_notebook.pbs (Gadi).
#
#   sbatch --export=ALL,NOTEBOOK=Ocean_Emulator_UNet,SEED=42,NOTEBOOK_COMMIT=$(git rev-parse HEAD) jobs/run_notebook.slurm
#   sbatch -J om2-crps43 --export=ALL,NOTEBOOK=Ocean_Emulator_UNet_CRPS,SEED=43,NOTEBOOK_COMMIT=$(git rev-parse HEAD) jobs/run_notebook.slurm
#
# The notebook is taken from git at NOTEBOOK_COMMIT (default: HEAD when the job starts), so editing
# or saving the working copy cannot change a queued run. The executed notebook, with all outputs,
# and this log go to runs/.
#
# gpu-highmem, not gpu: Slurm gives each GCD 1/8 of the node's memory, ~61 GB here against ~30 GB on
# gpu. The notebooks memory-map the 102 GB weather cache and keep the fields cache in host memory.
#
# The weather cache lives on /scratch (purged after 21 days untouched); the copy that lasts is in
# Acacia (rclone remote "acacia", bucket om2-emulator). The data folder's emulator_cache/ links to the
# scratch copy. This job restores it from Acacia if the purge took it, and touches it so it is kept.
#SBATCH --job-name=om2-notebook
#SBATCH --account=pawsey0364-gpu
#SBATCH --partition=gpu-highmem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/software/projects/pawsey0364/txs156/OM2-emulator/runs/%x_%j.log

set -euo pipefail

REPO=/software/projects/pawsey0364/txs156/OM2-emulator
VENV=/software/projects/pawsey0364/txs156/venvs/om2-emulator
WEATHER=/scratch/pawsey0364/txs156/OM2-emulator/cache/coupler_weather_all_v1.pt
WEATHER_BYTES=102126679702

module load rclone/1.68.1
if [ "$(stat -c %s "$WEATHER" 2>/dev/null || echo 0)" != "$WEATHER_BYTES" ]; then
    echo "Weather cache missing or incomplete on scratch: restoring from Acacia"
    mkdir -p "$(dirname "$WEATHER")"
    rclone copy acacia:om2-emulator/emulator_cache/coupler_weather_all_v1.pt "$(dirname "$WEATHER")"/
fi
touch "$WEATHER"

cd "$REPO"
mkdir -p runs
COMMIT=${NOTEBOOK_COMMIT:-$(git rev-parse HEAD)}
NOTEBOOK=${NOTEBOOK:-Ocean_Emulator_UNet}
export SEED=${SEED:-42}
INPUT=runs/${NOTEBOOK}_seed${SEED}_${SLURM_JOB_ID}.ipynb
git show "${COMMIT}:notebooks/${NOTEBOOK}.ipynb" > "$INPUT"
echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date); ${NOTEBOOK} seed ${SEED} from commit ${COMMIT}"

# torch.compile's caches on scratch, not in $HOME (1 GB quota).
export TRITON_CACHE_DIR=/scratch/pawsey0364/txs156/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/pawsey0364/txs156/.inductor_cache
# The notebooks set PYTORCH_CUDA_ALLOC_CONF; this is the same setting under its device-generic name,
# which ROCm builds read (checked on an MI250X).
export PYTORCH_ALLOC_CONF=expandable_segments:True

module load pytorch/2.13.0-rocm7.2.4
# Inside the container, with the venv (lightning, xarray, netCDF4, ...). The kernel is the venv's
# "om2-emulator" kernelspec: the default python3 kernel would run the container's python, without them.
srun -c "$SLURM_CPUS_PER_TASK" bash -c "
    source $VENV/bin/activate
    export JUPYTER_PATH=$VENV/share/jupyter
    rocm-smi --showproductname --showmeminfo vram 2>/dev/null | grep -E 'Card series|Total Memory' || true
    python -m jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=om2-emulator \
        --output-dir runs \
        --output '${NOTEBOOK}_seed${SEED}_${SLURM_JOB_ID}_executed' \
        '$INPUT'
"

echo "Finished at $(date)"
