# CRPRBench Evaluation Framework

中文 | [English](#english)

## 中文

## 1. 框架简介

本项目是一个面向**大语言模型化学任务**的自动化评估框架，主要用于评测模型在化学反应推理、反应物/产物预测、中间体预测等任务上的表现。

框架支持 5 类化学任务评估，并提供统一的模型调用接口。用户可以接入：

- OpenAI / DeepSeek 等 OpenAI-compatible API
- 本地 vLLM 部署的模型
- 自定义 HTTP JSON 推理服务
- 本地权重模型

评估过程中，框架会从模型回复中解析 SMILES，并计算以下指标：

- **Valid**：预测 SMILES 是否有效
- **Exact_match**：预测结果与标准答案规范化后是否完全一致
- **FTS**：基于 Morgan Fingerprint 的 Tanimoto 相似度

对于复杂回复，框架还支持通过 DeepSeek 对模型输出进行 SMILES 抽取与标准化解析。

---

## 2. 文件结构

```text
.
├── eval_framework/
│   ├── main.py                         # 主入口脚本
│   ├── model.py                        # 统一模型调用接口
│   ├── utils.py                        # SMILES 解析、指标计算、JSON 输出等工具函数
│   ├── eval_1.py                       # Task 1 评估脚本
│   ├── eval_2.py                       # Task 2 评估脚本
│   ├── eval_3.py                       # Task 3 评估脚本
│   ├── eval_4.py                       # Task 4 评估脚本
│   ├── eval_5.py                       # Task 5 评估脚本
│   ├── deepseek_parse.py               # 使用 DeepSeek 解析模型输出中的答案 SMILES
│   ├── summarize_results.py            # 汇总多个任务的评估结果
│   ├── run_vllm.sh                     # 启动本地 vLLM Server
│   └── run_all_eval.sh                 # 一键运行 5 个任务的评估
│
├── data/
│   ├── pattern_1.csv                   # Task 1 数据
│   ├── pattern_2.csv                   # Task 2 数据
│   ├── pattern_3.csv                   # Task 3 数据
│   ├── pattern_4.csv                   # Task 4 数据
│   └── pattern_5.csv                   # Task 5 数据
│
├── requirements.txt                    # Python 依赖
└── README.MD                           # 项目说明文档
```

---

## 3. 环境安装

建议使用 conda 创建独立环境：

```bash
conda create -n crprbench python=3.10
conda activate crprbench
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要使用 RDKit，也可以通过 conda 安装：

```bash
conda install -c conda-forge rdkit
```

如果使用 vLLM 本地部署模型，请在对应环境中安装 vLLM：

```bash
pip install vllm transformers
```

---

## 4. 数据格式

输入数据应为 CSV 文件。默认需要包含以下字段：

```text
Question_std
Question_cot
Question_chem
Answer
```

其中：

- `Question_std`：标准问题
- `Question_cot`：带推理提示的问题
- `Question_chem`：化学格式增强的问题
- `Answer`：标准答案，通常为 Python list 字符串，例如：

```text
['CCO', 'CCC']
```

或嵌套列表：

```text
[['CCO', 'CCC'], ['CCN', 'CCCl']]
```

不同任务会根据 `Answer` 的结构进行对应评估。

---

## 5. 使用方法

### 5.1 单个任务评估

可以使用 `main.py` 运行指定任务：

```bash
cd eval_framework

python main.py \
  --task 1 \
  --input ../data/pattern_1.csv \
  --output ./eval_outputs/task_1_result.json \
  --variant std \
  --num-runs 1 \
  --model "model:OpenAIChatCompatModel" \
  --model-kwargs '{"model_name":"local_model","base_url":"http://localhost:8000/v1","api_key":"EMPTY","temperature":0.01,"max_tokens":4096}'
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--task` | 任务编号，支持 `1` 到 `5` |
| `--input` | 输入 CSV 文件路径 |
| `--output` | 输出 JSON 文件路径 |
| `--variant` | 问题版本，可选 `std`、`cot`、`chem` |
| `--num-runs` | 每个样本重复评估次数 |
| `--model` | 模型类，格式为 `module:ClassName` |
| `--model-kwargs` | 模型初始化参数，JSON 字符串 |

---

### 5.2 问题版本选择

框架支持三种问题字段：

| `--variant` | 使用字段 |
|---|---|
| `std` | `Question_std` |
| `cot` | `Question_cot` |
| `chem` | `Question_chem` |

当使用 `--variant chem` 时，框架会调用 `deepseek_parse.py` 对模型回复进行答案 SMILES 抽取。

如果使用该模式，请先在 `deepseek_parse.py` 中配置 DeepSeek API Key：

```python
api_key = "YOUR_API_KEY"
```

---

### 5.3 使用本地 vLLM Server

进入评估框架目录后，首先启动 vLLM Server：

```bash
cd eval_framework

bash run_vllm.sh \
  "0" \
  "/path/to/local/model" \
  "local_model" \
  "0.0.0.0" \
  "8000"
```

参数依次为：

```text
CUDA_DEVICES
MODEL_PATH
SERVED_MODEL_NAME
HOST
PORT
```

启动成功后，默认 API 地址为：

```text
http://localhost:8000/v1
```

然后可以运行评估：

```bash
python main.py \
  --task 1 \
  --input ../data/pattern_1.csv \
  --output ./eval_outputs/task_1_result.json \
  --variant std \
  --num-runs 1 \
  --model "model:OpenAIChatCompatModel" \
  --model-kwargs '{"model_name":"local_model","base_url":"http://localhost:8000/v1","api_key":"EMPTY","temperature":0.01,"max_tokens":4096}'
```

---

### 5.4 一键评估所有任务

可以使用 `run_all_eval.sh` 一键评估 5 个任务：

```bash
cd eval_framework

bash run_all_eval.sh \
  "http://localhost:8000/v1" \
  "local_model" \
  "../data" \
  "./eval_outputs" \
  "MyModel"
```

参数说明：

| 参数 | 说明 |
|---|---|
| `BASE_URL` | vLLM OpenAI-compatible API 地址 |
| `MODEL_NAME` | vLLM served-model-name |
| `DATA_DIR` | 数据目录 |
| `OUTPUT_DIR` | 输出目录 |
| `FILE_SUFFIX` | 输出文件后缀，可选 |

运行后会生成：

```text
eval_framework/eval_outputs/
├── task_1_result_MyModel.json
├── task_2_result_MyModel.json
├── task_3_result_MyModel.json
├── task_4_result_MyModel.json
├── task_5_result_MyModel.json
├── summary_all_MyModel.json
└── run_all_eval_MyModel.log
```

---

## 6. 汇总结果

如果已经得到多个任务的评估结果，可以单独运行：

```bash
cd eval_framework

python summarize_results.py \
  --output-dir ./eval_outputs \
  --num-runs 1 \
  --summary-file ./eval_outputs/summary_all.json
```

如果结果文件带有后缀，例如：

```text
task_1_result_MyModel.json
```

则运行：

```bash
python summarize_results.py \
  --output-dir ./eval_outputs \
  --num-runs 1 \
  --file-suffix MyModel \
  --summary-file ./eval_outputs/summary_all_MyModel.json
```

---

## 7. 输出格式

每个任务会输出一个 JSON 文件，记录每个样本的：

- 输入问题
- 标准答案
- 模型原始回复
- 解析后的 SMILES
- 每次 run 的评估指标

示例：

```json
{
  "index": 0,
  "query": "...",
  "standard_answers": ["CCO"],
  "runs": [
    {
      "run_id": 1,
      "reply": "<SMILES>CCO</SMILES>",
      "predicted_smiles": "CCO",
      "results": {
        "Valid": 1,
        "Exact_match": 1,
        "FTS": 1.0
      }
    }
  ]
}
```

---

<a id="english"></a>

# English

## 1. Overview

This project is an automatic evaluation framework for **large language models on chemistry tasks**. It is mainly used to evaluate model performance on chemical reaction reasoning, reactant/product prediction, and intermediate prediction tasks.

The framework supports five types of chemistry tasks and provides a unified model calling interface. Users can connect it to:

- OpenAI / DeepSeek and other OpenAI-compatible APIs
- Local models served by vLLM
- Custom HTTP JSON inference services
- Local model weights

During evaluation, the framework parses SMILES from model responses and computes the following metrics:

- **Valid**: whether the predicted SMILES is valid
- **Exact_match**: whether the canonicalized prediction exactly matches the reference answer
- **FTS**: Tanimoto similarity based on Morgan fingerprints

For complex responses, the framework also supports using DeepSeek to extract and normalize answer SMILES from model outputs.

---

## 2. File Structure

```text
.
├── eval_framework/
│   ├── main.py                         # Main entry script
│   ├── model.py                        # Unified model calling interface
│   ├── utils.py                        # SMILES parsing, metric calculation, JSON output utilities
│   ├── eval_1.py                       # Task 1 evaluation script
│   ├── eval_2.py                       # Task 2 evaluation script
│   ├── eval_3.py                       # Task 3 evaluation script
│   ├── eval_4.py                       # Task 4 evaluation script
│   ├── eval_5.py                       # Task 5 evaluation script
│   ├── deepseek_parse.py               # Parse answer SMILES from model outputs using DeepSeek
│   ├── summarize_results.py            # Summarize evaluation results from multiple tasks
│   ├── run_vllm.sh                     # Start a local vLLM Server
│   └── run_all_eval.sh                 # Run all five evaluation tasks
│
├── data/
│   ├── pattern_1.csv                   # Task 1 data
│   ├── pattern_2.csv                   # Task 2 data
│   ├── pattern_3.csv                   # Task 3 data
│   ├── pattern_4.csv                   # Task 4 data
│   └── pattern_5.csv                   # Task 5 data
│
├── requirements.txt                    # Python dependencies
└── README.MD                           # Project documentation
```

---

## 3. Installation

We recommend creating an isolated conda environment:

```bash
conda create -n chem_eval python=3.10
conda activate chem_eval
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If RDKit is needed, it can also be installed via conda:

```bash
conda install -c conda-forge rdkit
```

If you use vLLM to serve local models, please install vLLM in the corresponding environment:

```bash
pip install vllm transformers
```

---

## 4. Data Format

The input data should be CSV files. By default, each file should contain the following columns:

```text
Question_std
Question_cot
Question_chem
Answer
```

where:

- `Question_std`: the standard question
- `Question_cot`: the question with reasoning prompts
- `Question_chem`: the chemistry-enhanced question
- `Answer`: the reference answer, usually stored as a Python list string, for example:

```text
['CCO', 'CCC']
```

or a nested list:

```text
[['CCO', 'CCC'], ['CCN', 'CCCl']]
```

Different tasks are evaluated according to the structure of the `Answer` field.

---

## 5. Usage

### 5.1 Evaluate a Single Task

You can run a specific task with `main.py`:

```bash
cd eval_framework

python main.py \
  --task 1 \
  --input ../data/pattern_1.csv \
  --output ./eval_outputs/task_1_result.json \
  --variant std \
  --num-runs 1 \
  --model "model:OpenAIChatCompatModel" \
  --model-kwargs '{"model_name":"local_model","base_url":"http://localhost:8000/v1","api_key":"EMPTY","temperature":0.01,"max_tokens":4096}'
```

Arguments:

| Argument | Description |
|---|---|
| `--task` | Task ID, from `1` to `5` |
| `--input` | Path to the input CSV file |
| `--output` | Path to the output JSON file |
| `--variant` | Question variant, chosen from `std`, `cot`, and `chem` |
| `--num-runs` | Number of repeated evaluations for each sample |
| `--model` | Model class, in the format `module:ClassName` |
| `--model-kwargs` | Model initialization arguments as a JSON string |

---

### 5.2 Question Variant Selection

The framework supports three question fields:

| `--variant` | Used field |
|---|---|
| `std` | `Question_std` |
| `cot` | `Question_cot` |
| `chem` | `Question_chem` |

When `--variant chem` is used, the framework calls `deepseek_parse.py` to extract answer SMILES from model responses.

If you use this mode, please configure the DeepSeek API key in `deepseek_parse.py` first:

```python
api_key = "YOUR_API_KEY"
```

---

### 5.3 Using a Local vLLM Server

Enter the evaluation framework directory and start the vLLM Server:

```bash
cd eval_framework

bash run_vllm.sh \
  "0" \
  "/path/to/local/model" \
  "local_model" \
  "0.0.0.0" \
  "8000"
```

The arguments are:

```text
CUDA_DEVICES
MODEL_PATH
SERVED_MODEL_NAME
HOST
PORT
```

After the server starts successfully, the default API endpoint is:

```text
http://localhost:8000/v1
```

Then run evaluation:

```bash
python main.py \
  --task 1 \
  --input ../data/pattern_1.csv \
  --output ./eval_outputs/task_1_result.json \
  --variant std \
  --num-runs 1 \
  --model "model:OpenAIChatCompatModel" \
  --model-kwargs '{"model_name":"local_model","base_url":"http://localhost:8000/v1","api_key":"EMPTY","temperature":0.01,"max_tokens":4096}'
```

---

### 5.4 Evaluate All Tasks

You can use `run_all_eval.sh` to evaluate all five tasks at once:

```bash
cd eval_framework

bash run_all_eval.sh \
  "http://localhost:8000/v1" \
  "local_model" \
  "../data" \
  "./eval_outputs" \
  "MyModel"
```

Arguments:

| Argument | Description |
|---|---|
| `BASE_URL` | OpenAI-compatible API endpoint of vLLM |
| `MODEL_NAME` | vLLM served model name |
| `DATA_DIR` | Data directory |
| `OUTPUT_DIR` | Output directory |
| `FILE_SUFFIX` | Optional suffix for output files |

After running the script, the following files will be generated:

```text
eval_framework/eval_outputs/
├── task_1_result_MyModel.json
├── task_2_result_MyModel.json
├── task_3_result_MyModel.json
├── task_4_result_MyModel.json
├── task_5_result_MyModel.json
├── summary_all_MyModel.json
└── run_all_eval_MyModel.log
```

---

## 6. Summarizing Results

If you have already obtained evaluation results from multiple tasks, you can summarize them separately:

```bash
cd eval_framework

python summarize_results.py \
  --output-dir ./eval_outputs \
  --num-runs 1 \
  --summary-file ./eval_outputs/summary_all.json
```

If the result files have a suffix, for example:

```text
task_1_result_MyModel.json
```

then run:

```bash
python summarize_results.py \
  --output-dir ./eval_outputs \
  --num-runs 1 \
  --file-suffix MyModel \
  --summary-file ./eval_outputs/summary_all_MyModel.json
```

---

## 7. Output Format

Each task outputs a JSON file that records the following information for each sample:

- Input question
- Reference answer
- Raw model response
- Parsed SMILES
- Evaluation metrics for each run

Example:

```json
{
  "index": 0,
  "query": "...",
  "standard_answers": ["CCO"],
  "runs": [
    {
      "run_id": 1,
      "reply": "<SMILES>CCO</SMILES>",
      "predicted_smiles": "CCO",
      "results": {
        "Valid": 1,
        "Exact_match": 1,
        "FTS": 1.0
      }
    }
  ]
}
```
