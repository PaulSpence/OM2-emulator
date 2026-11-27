#!/bin/bash
# Run a notebook from notebooks/ headless on one whole Setonix GPU node (Pawsey): 8 MI250X GCDs, data-parallel
# training. The Slurm counterpart of run_notebook.pbs (Gadi).
#
#   sbatch --export=ALL,NOTEBOOK=Ocean_Emulator_UNet,SEED=42,NOTEBOOK_COMMIT=$(git rev-parse HEAD) jobs/run_notebook.slurm
#   sbatch -J om2-crps43 --export=ALL,NOTEBOOK=Ocean_Emulator_UNet_CRPS,SEED=43,NOTEBOOK_COMMIT=$(git rev-parse HEAD) jobs/run_notebook.slurm
#
# srun starts the notebook once per GCD (one Slurm task each). Every copy runs the same cells on its own GCD
# (cell 2 picks it from the task number) and Lightning trains them as one data-parallel run; after training the
# other copies stop, and copy 0 alone runs the tests. Copy 0's executed notebook, with all outputs, and this log
# go to runs/.
#
# The notebook is taken from git at NOTEBOOK_COMMIT (default: HEAD when the job starts), so editing or saving the
# working copy cannot change a queued run.
#
# gpu-highmem (490 GB a node, against 245 GB on gpu): each copy holds the fields cache in host memory, and all
# share the page cache of the memory-mapped 102 GB weather cache.
#
# The weather cache lives on /scratch (purged after 21 days untouched); the copy that lasts is in Acacia (rclone
# remote "acacia", bucket om2-emulator). This job restores it from Acacia if the purge took it, touches it so it is
# kept, and copies it to the node's own NVMe disk (/tmp): the data folder's emulator_cache/ links to that node-local
# copy. Memory-mapped from /scratch, every copy of the notebook faulted it in at ~100 MB/s (17 minutes per job).
#SBATCH --job-name=om2-notebook
#SBATCH --account=pawsey0364-gpu
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --time=1-00:00:00
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

# To the node's NVMe, in 16 parallel 64 MiB-block streams (one stream reads /scratch at ~260 MB/s).
LOCAL_WEATHER=/tmp/om2-emulator/coupler_weather_all_v1.pt
mkdir -p "$(dirname "$LOCAL_WEATHER")"
START=$(date +%s)
BLOCKS=$(( (WEATHER_BYTES + 67108863) / 67108864 ))
PER=$(( (BLOCKS + 15) / 16 ))
for i in $(seq 0 15); do
    dd if="$WEATHER" of="$LOCAL_WEATHER" bs=64M skip=$((i * PER)) seek=$((i * PER)) count=$PER \
       conv=notrunc,sparse status=none &
done
wait
[ "$(stat -c %s "$LOCAL_WEATHER")" = "$WEATHER_BYTES" ] || { echo "Staging the weather cache failed"; exit 1; }
echo "Weather cache on the node's NVMe in $(( $(date +%s) - START )) s"

cd "$REPO"
mkdir -p runs
COMMIT=${NOTEBOOK_COMMIT:-$(git rev-parse HEAD)}
NOTEBOOK=${NOTEBOOK:-Ocean_Emulator_UNet}
export SEED=${SEED:-42}
NAME=${NOTEBOOK}_seed${SEED}_${SLURM_JOB_ID}
INPUT=runs/${NAME}.ipynb
# NOTEBOOK_FILE: run that file instead (an uncommitted test copy, e.g. with fewer epochs).
if [ -n "${NOTEBOOK_FILE:-}" ]; then
    cp "$NOTEBOOK_FILE" "$INPUT"
    COMMIT="(file $NOTEBOOK_FILE)"
else
    git show "${COMMIT}:notebooks/${NOTEBOOK}.ipynb" > "$INPUT"
fi
echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date); ${NOTEBOOK} seed ${SEED} from commit ${COMMIT}"

# torch.compile's caches on scratch, not in $HOME (1 GB quota).
export TRITON_CACHE_DIR=/scratch/pawsey0364/txs156/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/pawsey0364/txs156/.inductor_cache
# The notebooks set PYTORCH_CUDA_ALLOC_CONF; this is the same setting under its device-generic name, which ROCm
# builds read (checked on an MI250X).
export PYTORCH_ALLOC_CONF=expandable_segments:True

# AMD's convolution library (MIOpen) and its compiler. With benchmark=True MIOpen compiled and timed every
# candidate kernel of every convolution (~40 kernels/s, about an hour before the first steps finished). FAST: pick
# kernels from MIOpen's tuned database and heuristics instead. Its databases and the compiled kernels live on the
# node's /tmp during the job (8 processes writing them at once; /scratch's locking is not made for that), seeded
# from and saved back to /scratch, so each job reuses what the earlier ones compiled.
export MIOPEN_FIND_MODE=FAST
KERNEL_CACHE=/scratch/pawsey0364/txs156/.miopen_cache
export MIOPEN_USER_DB_PATH=/tmp/miopen/db MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen/cache AMD_COMGR_CACHE_DIR=/tmp/miopen/comgr
mkdir -p "$KERNEL_CACHE" /tmp/miopen/db /tmp/miopen/cache /tmp/miopen/comgr
cp -r "$KERNEL_CACHE"/. /tmp/miopen/ 2>/dev/null || true
trap 'rsync -a /tmp/miopen/ "$KERNEL_CACHE"/ || true' EXIT

module load pytorch/2.13.0-rocm7.2.4
# Inside the container, with the venv (lightning, xarray, netCDF4, ...); the kernel is the venv's "om2-emulator"
# kernelspec (the default python3 kernel would run the container's python, without them). Every task sees all 8
# GCDs and uses the one its task number names. Copies 1-7 end their kernel after training on purpose (os._exit),
# which nbconvert reports as an error: only copy 0's exit status counts. The kernels talk to nbconvert over
# Unix sockets in a folder of their own (ipc): 8 kernels picking random TCP ports at once collided.
srun -N 1 -n 8 -c 8 --gres=gpu:8 bash -c "
    source $VENV/bin/activate
    export JUPYTER_PATH=$VENV/share/jupyter
    export JUPYTER_RUNTIME_DIR=/tmp/jupyter_\${SLURM_JOB_ID}_\$SLURM_PROCID
    mkdir -p \$JUPYTER_RUNTIME_DIR
    if [ \$SLURM_PROCID = 0 ]; then
        rocm-smi --showproductname 2>/dev/null | grep -m1 'Card series' || true
        OUTPUT='${NAME}_executed'
    else
        OUTPUT=${NAME}_copy\$SLURM_PROCID
    fi
    python -m jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=om2-emulator \
        --KernelManager.transport=ipc --KernelManager.ip=\$JUPYTER_RUNTIME_DIR/kernel \
        --output-dir runs --output \"\$OUTPUT\" \
        '$INPUT' > runs/${NAME}_copy\$SLURM_PROCID.out 2>&1 \
    || [ \$SLURM_PROCID != 0 ]
"
rm -f runs/${NAME}_copy[1-7].ipynb runs/${NAME}_copy[1-7].out

echo "Finished at $(date)"
