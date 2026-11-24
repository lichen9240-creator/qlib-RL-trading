export QLIB_WORK=/root/autodl-tmp/qlib-work
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$QLIB_WORK/envs/qlib"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$QLIB_WORK/cache/pip"
export MPLCONFIGDIR="$QLIB_WORK/cache/matplotlib"
export TMPDIR="$QLIB_WORK/cache/tmp"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
cd "$QLIB_WORK/qlib/examples/rl_order_execution"

export XDG_CACHE_HOME="$QLIB_WORK/cache"
export TORCH_HOME="$QLIB_WORK/cache/torch"
export CUDA_CACHE_PATH="$QLIB_WORK/cache/cuda"
