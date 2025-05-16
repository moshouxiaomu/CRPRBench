import pandas as pd
import re
import ast
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import itertools
import openai_api
import datetime
import time
from zoneinfo import ZoneInfo
import argparse

def canonical_smiles(smiles):
    """将SMILES转换为规范形式"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None
    except:
        return None

def parse_two_smiles(response):
    """
    从回复中解析出最多两个 <SMILES> ... </SMILES> 的内容。
    如果只匹配到一个，则返回 [pred1, None]；
    如果一个都没匹配到，则返回 [None, None]。
    """
    matches = re.findall(r'<SMILES>(.*?)</SMILES>', response, re.DOTALL)
    # 取最后两个
    if len(matches) >= 2:
        return [matches[-1].strip(), matches[-2].strip()]
    elif len(matches) == 1:
        return [matches[0].strip(), None]
    else:
        return [None, None]

def evaluate_prediction_against_answer(pred_smiles, answer):
    """"评估单个预测与特定答案的匹配情况，返回指标"""
    if answer is None:
        if not pred_smiles:
            return {'Valid': 0, 'Exact_match': 0, 'FTS': 0.0}
        pred_mol = Chem.MolFromSmiles(pred_smiles)
        if pred_mol is None:
            return {'Valid': 0, 'Exact_match': 0, 'FTS': 0.0}
        else:
            return {'Valid': 1, 'Exact_match': 0, 'FTS': 0.0}
    
    # 处理答案有效性
    ans_mol = Chem.MolFromSmiles(answer)
    if ans_mol is None:
        raise ValueError(f"答案 {answer} 无效")
    ans_canon = Chem.MolToSmiles(ans_mol)

    # 处理预测有效性
    if not pred_smiles:
        return {'Valid': 0, 'Exact_match': 0, 'FTS': 0.0}
    pred_mol = Chem.MolFromSmiles(pred_smiles)
    if pred_mol is None:
        return {'Valid': 0, 'Exact_match': 0, 'FTS': 0.0}
    pred_canon = Chem.MolToSmiles(pred_mol)

    # 计算Exact Match
    exact_match = 1 if pred_canon == ans_canon else 0

    # 计算FTS
    fp_pred = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048)
    fp_ans = AllChem.GetMorganFingerprintAsBitVect(ans_mol, 2, nBits=2048)
    fts = DataStructs.TanimotoSimilarity(fp_pred, fp_ans)

    return {
        'Valid': 1,
        'Exact_match': exact_match,
        'FTS': fts
    }

def evaluate_two_predictions(pred1, pred2, answer_list):
    """评估两个预测与答案列表的最优匹配，返回最佳指标组合"""
    best_metrics1 = {'Valid': 0, 'Exact_match': 0, 'FTS': 0.0}
    best_metrics2 = {'Valid': 0, 'Exact_match': 0, 'FTS': 0.0}
    best_total_em = -1
    best_avg_fts = -1.0
    
    if not answer_list:
        m1 = evaluate_prediction_against_answer(pred1, None)
        m2 = evaluate_prediction_against_answer(pred2, None)
        return m1, m2, 0, 0.0
    
    possible_pairs = []
    if len(answer_list) >= 2:
        possible_pairs.extend(itertools.permutations(answer_list, 2))
    else:
        raise ValueError("标准答案只有一个")
    
    for ans1, ans2 in possible_pairs:
        m1 = evaluate_prediction_against_answer(pred1, ans1)
        m2 = evaluate_prediction_against_answer(pred2, ans2)
        total_em = m1['Exact_match'] + m2['Exact_match']
        avg_fts = (m1['FTS'] + m2['FTS']) / 2
        
        if (total_em > best_total_em) or \
           (total_em == best_total_em and avg_fts > best_avg_fts):
            best_total_em = total_em
            best_avg_fts = avg_fts
            best_metrics1 = m1
            best_metrics2 = m2
    
    return best_metrics1, best_metrics2, best_total_em, best_avg_fts


def evaluate_dataset(input_path, output_path, model_name):
    """执行完整评估流程"""
    df = pd.read_csv(input_path)
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # std评估
            response_std = Eval_LLM.attempt_api_call(system_prompt, row['Question_std'], model=model_name)
            # 1) 解析出两个 SMILES 串 (对应两个反应物组合)
            pred_smiles_std_1, pred_smiles_std_2 = parse_two_smiles(response_std)
            # 2) 解析答案列表
            answer_list = ast.literal_eval(row['Answer'])

            metrics_std_1, metrics_std_2, total_em_std, avg_fts_std = evaluate_two_predictions(
                pred_smiles_std_1, pred_smiles_std_2, list(answer_list)
            )
            valid_std = (metrics_std_1['Valid'] + metrics_std_2['Valid']) / 2
            exact_std = (metrics_std_1['Exact_match'] + metrics_std_2['Exact_match']) / 2
            fts_std = (metrics_std_1['FTS'] + metrics_std_2['FTS']) / 2
            
            # cot评估
            # response_cot = Eval_LLM.attempt_api_call(system_prompt, row['Question_cot'], model=model_name)
            # # 1) 解析出两个 SMILES 串 (对应两个反应物组合)
            # pred_cot_1, pred_cot_2 = parse_two_smiles(response_cot)

            # metrics_cot_1, metrics_cot_2, total_em_cot, avg_fts_cot = evaluate_two_predictions(
            #     pred_cot_1, pred_cot_2, list(answer_list)
            # )
            # valid_cot = (metrics_cot_1['Valid'] + metrics_cot_2['Valid']) / 2
            # exact_cot = (metrics_cot_1['Exact_match'] + metrics_cot_2['Exact_match']) / 2
            # fts_cot = (metrics_cot_1['FTS'] + metrics_cot_2['FTS']) / 2

            result = row.to_dict()
            result.update({
                'Model_response_std': response_std,
                # 'Model_response_cot': response_cot,
                # 标准问题预测结果
                'Pred_SMILES_std_1': pred_smiles_std_1,
                'Pred_SMILES_std_2': pred_smiles_std_2,
                'Valid_std_1': metrics_std_1['Valid'],
                'Exact_match_std_1': metrics_std_1['Exact_match'],
                'FTS_std_1': metrics_std_1['FTS'],
                'Valid_std_2': metrics_std_2['Valid'],
                'Exact_match_std_2': metrics_std_2['Exact_match'],
                'FTS_std_2': metrics_std_2['FTS'],
                'Valid_std': valid_std,
                'Exact_match_std': exact_std,
                'FTS_std': fts_std,
                # CoT问题预测结果
                # 'Pred_SMILES_cot_1': pred_cot_1,
                # 'Pred_SMILES_cot_2': pred_cot_2,
                # 'Valid_cot_1': metrics_cot_1['Valid'],
                # 'Exact_match_cot_1': metrics_cot_1['Exact_match'],
                # 'FTS_cot_1': metrics_cot_1['FTS'],
                # 'Valid_cot_2': metrics_cot_2['Valid'],
                # 'Exact_match_cot_2': metrics_cot_2['Exact_match'],
                # 'FTS_cot_2': metrics_cot_2['FTS'],
                # 'Valid_cot': valid_cot,
                # 'Exact_match_cot': exact_cot,
                # 'FTS_cot': fts_cot,
            })
            results.append(result)
        except:
            print("出现错误")
            continue
    
    # 存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    
    # 计算汇总指标
    summary = {
        'Validity_rate_std': result_df['Valid_std'].mean(),
        'Exact_match_rate_std': result_df['Exact_match_std'].mean(),
        'Average_FTS_std': result_df['FTS_std'].mean()
        # 'Validity_rate_cot': result_df['Valid_cot'].mean(),
        # 'Exact_match_rate_cot': result_df['Exact_match_cot'].mean(),
        # 'Average_FTS_cot': result_df['FTS_cot'].mean()
    }
    
    print("\nEvaluation Summary:")
    print("Standard Questions:")
    print(f"Validity Rate: {summary['Validity_rate_std']:.4f}")
    print(f"Exact Match Rate: {summary['Exact_match_rate_std']:.4f}")
    print(f"Average FTS: {summary['Average_FTS_std']:.4f}")
    # print("\nCoT Questions:")
    # print(f"Validity Rate: {summary['Validity_rate_cot']:.4f}")
    # print(f"Exact Match Rate: {summary['Exact_match_rate_cot']:.4f}")
    # print(f"Average FTS: {summary['Average_FTS_cot']:.4f}")
    return summary

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
    input_path = '/home/xshe/KG/eval/benchmark/new_benchmark/new_pattern_4.csv'
    output_base_path = f'/home/xshe/KG/eval/benchmark/new_results/pat4_results_{args.model_name}_'
    all_summary = []
    for i in range(args.start_index, args.end_index):  # 生成3个输出文件
        output_path = f"{output_base_path}{i}.csv"
        summary = evaluate_dataset(input_path, output_path, args.model_name)
        all_summary.append(summary)
    
    for summary in all_summary:
        print("\nEvaluation Summary:")
        print("Standard Questions:")
        print(f"Validity Rate: {summary['Validity_rate_std']:.4f}")
        print(f"Exact Match Rate: {summary['Exact_match_rate_std']:.4f}")
        print(f"Average FTS: {summary['Average_FTS_std']:.4f}")
        # print("\nCoT Questions:")
        # print(f"Validity Rate: {summary['Validity_rate_cot']:.4f}")
        # print(f"Exact Match Rate: {summary['Exact_match_rate_cot']:.4f}")
        # print(f"Average FTS: {summary['Average_FTS_cot']:.4f}")