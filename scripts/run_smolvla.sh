#!/bin/bash
# usage: scripts/run_smolvla.sh <suite> <n_episodes_per_task> [tag] [extra lerobot-eval args...]
# default checkpoint: HuggingFaceVLA/smolvla_libero (native image/image2 keys, n_action_steps=1)
cd "$(dirname "$0")/.." && source scripts/run_env.sh
S=${1:-libero_spatial}; N=${2:-10}; TAG=${3:-smolvla}; shift 3
timeout 14400 nice -n 19 ionice -c3 ${VENV:-.venv}/bin/python scripts/eval_timed.py results/timing_${TAG}_$S.json \
 --policy.path=${POLICY:-HuggingFaceVLA/smolvla_libero} --env.type=libero --env.task=$S --eval.batch_size=1 --eval.n_episodes=$N \
 --env.max_parallel_tasks=1 --seed=1000 --output_dir=logs/${TAG}_$S "$@"
