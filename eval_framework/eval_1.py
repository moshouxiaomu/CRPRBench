# eval_1.py
# A -> B -> C
import json
from typing import Any, Dict, List
import pandas as pd
from tqdm import tqdm
from eval.eval_framework.deepseek_parse import parse_response

from utils import (
    read_answer_list_literal,
    parse_last_smiles,
    evaluate_single_vs_list,
    dump_json,
)

def evaluate_dataset(input_path: str,
                     output_path: str,
                     model: Any,
                     num_runs: int = 1,
                     question_field: str = "Question_std",
                     variant: str = 'std') -> None:
    """
    Task 1: For each sample, output exactly one prediction (take the last <SMILES>),
    then compare it against the column 'Intermediate' (a list of SMILES) and keep the best match.
    """
    df = pd.read_csv(input_path)
    # Intermediate: e.g., "['CCO', 'CCC', ...]"
    df['Answer'] = df['Answer'].apply(read_answer_list_literal)

    results: List[Dict] = []
    total_cnt = 0
    valid_sum = exact_sum = fts_sum = 0.0

    for index, row in tqdm(df.iterrows(), total=len(df), miniters=10):
        true_list = row['Answer']
        query = row[question_field]

        sample_rec = {
            "index": int(index),
            "query": query,
            "question_field": question_field,
            "standard_answers": true_list,
            "runs": []
        }

        for r in range(1, num_runs + 1):
            try:
                response = model.call(query)
                response_old = response
                if variant == "chem":
                    response = parse_response(query, response)
                pred_smiles = parse_last_smiles(response) or response  # fallback when no <SMILES> tags are present
                metrics = evaluate_single_vs_list(pred_smiles, true_list) if pred_smiles else {
                    'Valid': 0, 'Exact_match': 0, 'FTS': 0.0
                }
                if variant == "chem":
                    sample_rec["runs"].append({
                        "run_id": r,
                        "reply": response_old,
                        "reply_new": response,
                        "predicted_smiles": pred_smiles,
                        "results": metrics
                    })
                else:
                    sample_rec["runs"].append({
                        "run_id": r,
                        "reply": response,
                        "predicted_smiles": pred_smiles,
                        "results": metrics
                    })

                total_cnt += 1
                valid_sum += metrics['Valid']
                exact_sum += metrics['Exact_match']
                fts_sum += metrics['FTS']

            except Exception as e:
                sample_rec["runs"].append({
                    "run_id": r,
                    "error": f"{type(e).__name__}: {e}"
                })

        results.append(sample_rec)

    if total_cnt > 0:
        print(f"[SUMMARY] runs={num_runs}, samples={len(df)}")
        print(f"  Avg Valid       : {valid_sum / total_cnt:.4f}")
        print(f"  Avg Exact_match : {exact_sum / total_cnt:.4f}")
        print(f"  Avg FTS         : {fts_sum / total_cnt:.4f}")

    dump_json(output_path, results)
