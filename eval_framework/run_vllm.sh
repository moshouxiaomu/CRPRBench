#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICES=${1:-"2"}
MODEL_PATH=${2:-"/home/data2/models/Mol-Reasoner/grpo/grpo_text_based_de_novo_molecule_generation"}
SERVED_MODEL_NAME=${3:-"local_model"}
HOST=${4:-"0.0.0.0"}
PORT=${5:-"8000"}
TENSOR_PARALLEL_SIZE=${6:-"1"}
GPU_MEMORY_UTILIZATION=${7:-"0.8"}
MAX_MODEL_LEN=${8:-"8192"}
MAX_NUM_SEQS=${9:-"8"}
REASONING_PARSER=${10:-"none"}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# 仅用于提示，不参与评估输出
MODEL_PATH_CLEAN="${MODEL_PATH%/}"
MODEL_SUFFIX="$(basename "${MODEL_PATH_CLEAN}")"
MODEL_SUFFIX_SAFE="$(echo "${MODEL_SUFFIX}" | sed 's#[/[:space:]]#_#g')"

cmd=(
  python -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
)

if [[ -n "${REASONING_PARSER}" && "${REASONING_PARSER}" != "none" ]]; then
  cmd+=(--reasoning-parser "${REASONING_PARSER}")
fi

echo "============================================="
echo " vLLM Server"
echo "============================================="
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] MODEL_SUFFIX=${MODEL_SUFFIX_SAFE}"
echo "[INFO] SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "[INFO] HOST=${HOST}"
echo "[INFO] PORT=${PORT}"
echo "[INFO] BASE_URL=http://localhost:${PORT}/v1"
echo ""
echo "[INFO] 评估脚本建议使用后缀参数:"
echo "       ${MODEL_SUFFIX_SAFE}"
echo "============================================="

printf '[INFO] Command: %q ' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
