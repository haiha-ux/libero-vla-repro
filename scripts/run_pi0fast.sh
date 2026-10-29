#!/bin/bash
# usage: scripts/run_pi0fast.sh <suite> <n_episodes_per_task> [tag] [extra args...]
# Needs ckpt/pi0fast-libero-v044 (see README: tokenizer_name patched to local PaliGemma tokenizer copy).
cd "$(dirname "$0")/.." && source scripts/run_env.sh
S=${1:-libero_spatial}; N=${2:-10}; TAG=${3:-pi0fast}; shift 3
timeout 14400 nice -n 19 ionice -c3 ${VENV:-.venv}/bin/python scripts/eval_timed.py results/timing_${TAG}_$S.json \
 --policy.path=ckpt/pi0fast-libero-v044 --env.type=libero --env.task=$S --eval.batch_size=1 --eval.n_episodes=$N \
 --env.max_parallel_tasks=1 --seed=1000 --output_dir=logs/${TAG}_$S \
 --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}' "$@"
