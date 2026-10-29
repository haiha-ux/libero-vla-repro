# source this: shared env for all heavy steps (shared machine, see README)
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export LIBERO_CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.libero_cfg"
export TOKENIZERS_PARALLELISM=false
heavy() { nice -n 19 ionice -c3 "$@"; }
