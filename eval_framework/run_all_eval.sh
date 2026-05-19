#!/usr/bin/env bash
# =============================================================================
# run_all_eval.sh
#
# 一键评估已经启动的 vLLM Server 在 5 个测试集上的表现。
#
# 注意：
#   本脚本不启动 vLLM Server。
#   请先在 vLLM conda 环境中启动 server，
#   再在 eval conda 环境中运行本脚本。
#
# 用法:
#   bash run_all_eval.sh [BASE_URL] [MODEL_NAME] [DATA_DIR] [OUTPUT_DIR] [FILE_SUFFIX]
#
# 示例:
#   bash run_all_eval.sh \
#     "http://localhost:8000/v1" \
#     "local_model" \
#     "/home/hexuesong/Kg2SciQA/eval/data" \
#     "./eval_outputs" \
#     "ChemDFM-R"
#
# 参数说明:
#   1 BASE_URL      vLLM OpenAI-compatible API 地址
#   2 MODEL_NAME    vLLM served-model-name
#   3 DATA_DIR      数据目录
#   4 OUTPUT_DIR    输出目录
#   5 FILE_SUFFIX   输出文件后缀，例如 ChemDFM-R；可为空
#
# 输出文件:
#   如果 FILE_SUFFIX=ChemDFM-R:
#     task_1_result_ChemDFM-R.json
#     summary_all_ChemDFM-R.json
#     run_all_eval_ChemDFM-R.log
#
#   如果 FILE_SUFFIX 为空:
#     task_1_result.json
#     summary_all.json
#     run_all_eval.log
# =============================================================================

set -euo pipefail

# ---------- 可配置参数，支持命令行覆盖 ----------
BASE_URL=${1:-"http://localhost:8000/v1"}
MODEL_NAME=${2:-"local_model"}
DATA_DIR=${3:-"/home/hexuesong/Kg2SciQA/eval/data"}
OUTPUT_DIR=${4:-"./eval_outputs"}

# 手动指定输出文件后缀
# 例如 ChemDFM-R，则输出 task_1_result_ChemDFM-R.json
FILE_SUFFIX=${5:-""}

# 评估配置
VARIANT="std"
NUM_RUNS=1
TEMPERATURE=0.01
MAX_TOKENS=4096
TIMEOUT=120.0

TASK_IDS=(1 2 3 4 5)

# ---------- 处理文件后缀 ----------
# 防止后缀里包含空格、斜杠等不适合作为文件名的字符
FILE_SUFFIX_SAFE="$(echo "${FILE_SUFFIX}" | sed 's#[/[:space:]]#_#g')"

if [[ -n "${FILE_SUFFIX_SAFE}" ]]; then
    SUFFIX_PART="_${FILE_SUFFIX_SAFE}"
else
    SUFFIX_PART=""
fi

# ---------- 初始化输出目录 ----------
mkdir -p "${OUTPUT_DIR}"

SUMMARY_FILE="${OUTPUT_DIR}/summary_all${SUFFIX_PART}.json"
LOG_FILE="${OUTPUT_DIR}/run_all_eval${SUFFIX_PART}.log"

echo "=============================================" | tee -a "${LOG_FILE}"
echo " vLLM Benchmark Evaluation"                   | tee -a "${LOG_FILE}"
echo " BASE_URL        : ${BASE_URL}"               | tee -a "${LOG_FILE}"
echo " MODEL_NAME      : ${MODEL_NAME}"             | tee -a "${LOG_FILE}"
echo " DATA_DIR        : ${DATA_DIR}"               | tee -a "${LOG_FILE}"
echo " OUTPUT_DIR      : ${OUTPUT_DIR}"             | tee -a "${LOG_FILE}"
echo " FILE_SUFFIX     : ${FILE_SUFFIX_SAFE}"       | tee -a "${LOG_FILE}"
echo " SUMMARY_FILE    : ${SUMMARY_FILE}"           | tee -a "${LOG_FILE}"
echo " VARIANT         : ${VARIANT}"                | tee -a "${LOG_FILE}"
echo " NUM_RUNS        : ${NUM_RUNS}"               | tee -a "${LOG_FILE}"
echo " TEMPERATURE     : ${TEMPERATURE}"            | tee -a "${LOG_FILE}"
echo " MAX_TOKENS      : ${MAX_TOKENS}"             | tee -a "${LOG_FILE}"
echo " TIMEOUT         : ${TIMEOUT}"                | tee -a "${LOG_FILE}"
echo " START TIME      : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
echo "=============================================" | tee -a "${LOG_FILE}"

# ---------- 检查 vLLM Server 是否可用 ----------
echo "[INFO] 检查 vLLM Server 是否可用: ${BASE_URL}/models" | tee -a "${LOG_FILE}"

if ! curl -s "${BASE_URL}/models" >/dev/null 2>&1; then
    echo "[ERROR] 无法访问 vLLM Server: ${BASE_URL}/models" | tee -a "${LOG_FILE}"
    echo "[ERROR] 请确认 server 已经在对应 conda 环境中启动，并且端口正确。" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "[INFO] vLLM Server 可用" | tee -a "${LOG_FILE}"

# ---------- 构建 model-kwargs JSON ----------
MODEL_KWARGS=$(python3 -c "
import json
print(json.dumps({
    'model_name': '${MODEL_NAME}',
    'base_url': '${BASE_URL}',
    'api_key': 'EMPTY',
    'system_prompt': 'You are an expert in chemistry.',
    'temperature': ${TEMPERATURE},
    'max_tokens': ${MAX_TOKENS},
    'timeout': ${TIMEOUT}
}))
")

# ---------- 逐任务评测 ----------
for TASK_ID in "${TASK_IDS[@]}"; do
    INPUT_FILE="${DATA_DIR}/pattern_${TASK_ID}.csv"
    OUTPUT_FILE="${OUTPUT_DIR}/task_${TASK_ID}_result${SUFFIX_PART}.json"

    echo "" | tee -a "${LOG_FILE}"
    echo ">>> [TASK ${TASK_ID}] 开始评测: ${INPUT_FILE}" | tee -a "${LOG_FILE}"
    echo "    输出文件 : ${OUTPUT_FILE}" | tee -a "${LOG_FILE}"
    echo "    时间     : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"

    if [[ ! -f "${INPUT_FILE}" ]]; then
        echo "[ERROR] 输入文件不存在: ${INPUT_FILE}" | tee -a "${LOG_FILE}"
        exit 1
    fi

    python3 main.py \
        --task "${TASK_ID}" \
        --input "${INPUT_FILE}" \
        --output "${OUTPUT_FILE}" \
        --variant "${VARIANT}" \
        --num-runs "${NUM_RUNS}" \
        --model "model:OpenAIChatCompatModel" \
        --model-kwargs "${MODEL_KWARGS}" \
        2>&1 | tee -a "${LOG_FILE}"

    # mistral
    # python3 main.py \
    #     --task "${TASK_ID}" \
    #     --input "${INPUT_FILE}" \
    #     --output "${OUTPUT_FILE}" \
    #     --variant "${VARIANT}" \
    #     --num-runs "${NUM_RUNS}" \
    #     --model "model:OpenAICompletionCompatModel" \
    #     --model-kwargs "${MODEL_KWARGS}" \
    #     2>&1 | tee -a "${LOG_FILE}"

    echo "<<< [TASK ${TASK_ID}] 评测完成" | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "=============================================" | tee -a "${LOG_FILE}"
echo " 所有任务评测完成，开始汇总统计..." | tee -a "${LOG_FILE}"
echo "=============================================" | tee -a "${LOG_FILE}"

# ---------- 汇总统计 ----------
if [[ -n "${FILE_SUFFIX_SAFE}" ]]; then
    python3 summarize_results.py \
        --output-dir "${OUTPUT_DIR}" \
        --num-runs "${NUM_RUNS}" \
        --file-suffix "${FILE_SUFFIX_SAFE}" \
        --summary-file "${SUMMARY_FILE}" \
        2>&1 | tee -a "${LOG_FILE}"
else
    python3 summarize_results.py \
        --output-dir "${OUTPUT_DIR}" \
        --num-runs "${NUM_RUNS}" \
        --summary-file "${SUMMARY_FILE}" \
        2>&1 | tee -a "${LOG_FILE}"
fi

echo "" | tee -a "${LOG_FILE}"
echo "=============================================" | tee -a "${LOG_FILE}"
echo " 全部完成！" | tee -a "${LOG_FILE}"
echo " 汇总结果: ${SUMMARY_FILE}" | tee -a "${LOG_FILE}"
echo " 日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
echo "=============================================" | tee -a "${LOG_FILE}"
