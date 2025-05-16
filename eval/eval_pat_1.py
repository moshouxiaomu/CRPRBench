import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import ast
import openai_api
import datetime
import time
from zoneinfo import ZoneInfo
import argparse
# import llama3_8b_instruct as llama

def canonical_smiles(smiles):
    """将SMILES转换为规范形式"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None
    except:
        return None

def parse_last_smiles(response):
    """从回复中提取最后一个<SMILES>标签内容"""
    matches = re.findall(r'<SMILES>(.*?)</SMILES>', response, re.DOTALL)
    return matches[-1].strip() if matches else None

def calculate_metrics(pred_smiles, true_smiles_list):
    """计算评估指标（支持多标准答案）"""
    # 验证预测SMILES有效性
    valid = 1 if Chem.MolFromSmiles(pred_smiles) else 0
    pred_canon = canonical_smiles(pred_smiles) if valid else None
    
    # 初始化指标
    max_fts = 0.0
    exact_match = 0
    
    # 仅当预测有效时进行计算
    if valid and pred_canon:
        try:
            mol_pred = Chem.MolFromSmiles(pred_canon)
            fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 2, nBits=2048)
            
            # 遍历所有标准答案
            for true_smiles in true_smiles_list:
                true_canon = true_smiles
                # 检查精确匹配
                if pred_canon == true_canon:
                    exact_match = 1
                
                # 计算相似度
                mol_true = Chem.MolFromSmiles(true_canon)
                fp_true = AllChem.GetMorganFingerprintAsBitVect(mol_true, 2, nBits=2048)
                similarity = DataStructs.TanimotoSimilarity(fp_pred, fp_true)
                
                # 保留最高相似度
                if similarity > max_fts:
                    max_fts = similarity
        except:
            pass
    
    return {
        'Valid': valid,
        'Exact_match': exact_match,
        'FTS': max_fts
    }

def evaluate_dataset(input_path, output_path, model_name):
    """执行完整评估流程"""
    # 读取数据
    df = pd.read_csv(input_path)
    df['Intermediate'] = df['Intermediate'].apply(ast.literal_eval)
    results = []
    total_samples_processed = 0
    valid_sum = 0
    exact_match_sum = 0
    fts_sum = 0
    # 遍历每个样本
    for index, row in tqdm(df.iterrows(), total=len(df), miniters=10):
        try:
            # 获取模型standard回复
            true_smiles = row['Intermediate']

            # 标准提问
            response_std = Eval_LLM.attempt_api_call(system_prompt, row['Question_std'], model=model_name)
            pred_smiles_std = parse_last_smiles(response_std)
            if pred_smiles_std == None:
                pred_smiles_std = response_std  # gpt没有输出<SMILES>就直接用输出
            metrics_std = {'Valid': 0, 'Exact_match': 0, 'FTS': 0.0}
            # 计算指标
            if pred_smiles_std:
                metrics_std = calculate_metrics(pred_smiles_std, true_smiles)

            # 创建独立字典
            result_std = row.to_dict()
            result_std.update({
                'Model_response': response_std,
                'Predicted_SMILES': pred_smiles_std,
                'Prompt_Type': 'Standard',
                **metrics_std
            })
            results.append(result_std)

            # 更新统计计数器
            total_samples_processed += 1
            valid_sum += metrics_std['Valid']
            exact_match_sum += metrics_std['Exact_match']
            fts_sum += metrics_std['FTS']
        
            # CoT提问
            # response_CoT = Eval_LLM.attempt_api_call(system_prompt, row['Question_CoT'], model=model_name)
            # pred_smiles_CoT = parse_last_smiles(response_CoT)
            # metrics_CoT = {'Valid': 0, 'Exact_match': 0, 'FTS': 0.0}
            # # 计算指标
            # if pred_smiles_CoT:
            #     metrics_CoT = calculate_metrics(pred_smiles_CoT, true_smiles)
            
            
            # 保存两种结果
            # result_cot = row.to_dict()
            # result_cot.update({
            #     'Model_response': response_CoT,
            #     'Predicted_SMILES': pred_smiles_CoT,
            #     'Prompt_Type': 'CoT',
            #     **metrics_CoT
            # })
            # results.append(result_cot)
        except Exception as e:
            print(f"[ERROR] Error processing sample {index}: {e}")
            print(f"[INFO] Processed {total_samples_processed} samples so far.")
            if total_samples_processed > 0:
                avg_valid = valid_sum / total_samples_processed
                avg_exact = exact_match_sum / total_samples_processed
                avg_fts = fts_sum / total_samples_processed
                print(f"[INFO] Avg Valid: {avg_valid:.4f}, Exact: {avg_exact:.4f}, FTS: {avg_fts:.4f}")
            # 继续处理下一个样本
    
    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    
    # 分组统计指标
    summary = result_df.groupby('Prompt_Type').agg({
        'Valid': 'mean',
        'Exact_match': 'mean',
        'FTS': 'mean'
    }).reset_index()
    
    print("\nEvaluation Summary:")
    for _, row in summary.iterrows():
        print(f"\nPrompt Type: {row['Prompt_Type']}")
        print(f"Validity_rate: {row['Valid']:.4f}")
        print(f"Exact_match_rate: {row['Exact_match']:.4f}")
        print(f"Average_FTS: {row['FTS']:.4f}")

def wait_until_target_time():
    # 设置时区为北京时间
    tz = ZoneInfo("Asia/Shanghai")
    now = datetime.datetime.now(tz)
    
    # 构造目标时间
    target_time = now.replace(hour=0, minute=25, second=0, microsecond=0)
    
    # 如果当前时间已经过了今天23点，则目标时间改为明天23点
    if now >= target_time:
        target_time += datetime.timedelta(days=1)
    
    # 计算需要休眠的时间差
    delta = target_time - now
    sleep_seconds = delta.total_seconds()
    
    print(f"距离0:25北京时间还剩{sleep_seconds:.2f}秒，开始休眠...")
    time.sleep(sleep_seconds)

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models on chemical dataset.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to evaluate (e.g., gpt-4o, deepseek).')
    parser.add_argument('--start_index', type=int, default=1, help='Start index for output files (default: 1).')
    parser.add_argument('--end_index', type=int, default=6, help='End index for output files (exclusive, default: 6).')
    parser.add_argument('--api_from', type=str, default="xunfei", help='End index for output files (exclusive, default: 6).')
    args = parser.parse_args()

    Eval_LLM = openai_api.OpenAIClient(api_from=args.api_from)
    system_prompt = 'You are an expert in chemistry.'

    # wait_until_target_time()
    input_path = '/home/xshe/KG/eval/benchmark/new_benchmark/new_pattern_1.csv'
    output_base_path = f'/home/xshe/KG/eval/benchmark/new_results/pat1_results_{args.model_name}_'
    for i in range(args.start_index, args.end_index):  # 生成3个输出文件
        output_path = f"{output_base_path}{i}.csv"
        evaluate_dataset(input_path, output_path, args.model_name)