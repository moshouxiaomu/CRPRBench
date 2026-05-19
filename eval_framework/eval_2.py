# eval_2.py
# A -> B -> C -> D  (two predictions + paired answers)
from typing import Any, Dict, List
import pandas as pd
from tqdm import tqdm
from eval.eval_framework.deepseek_parse import parse_response

from utils import (
    read_answer_list_literal,
    parse_two_smiles,
    evaluate_two_vs_paired_answers,
    dump_json,
)

def evaluate_dataset(input_path: str,
                     output_path: str,
                     model: Any,
                     num_runs: int = 1,
                     question_field: str = "Question_std",
                     variant: str = 'std') -> None:
    """
    Task 2: Produce two predictions (take the second-to-last and the last <SMILES>),
    then compare them against the column 'Intermediate' (a list of paired answers, e.g., [[a1, a2], ...]),
    evaluating over each pair and selecting the best.
    """
    df = pd.read_csv(input_path)
    results: List[Dict] = []

    total_cnt = 0
    valid_sum = exact_sum = fts_sum = 0.0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        answer_pairs = read_answer_list_literal(row['Answer'])
        query = row[question_field]

        sample_record = {
            "index": int(index),
            "query": query,
            "question_field": question_field,
            "standard_answers": answer_pairs,
            "runs": []
        }

        for r in range(1, num_runs + 1):
            try:
                response = model.call(query)
                response_old = response
                if variant == "chem":
                    response = parse_response(query, response)
                # Paired task: use the second-to-last and the last entries
                pred1, pred2 = parse_two_smiles(response, order="pair")

                m1, m2 = evaluate_two_vs_paired_answers(pred1, pred2, answer_pairs)

                valid_avg = (m1['Valid'] + m2['Valid']) / 2.0
                exact_avg = (m1['Exact_match'] + m2['Exact_match']) / 2.0
                fts_avg = (m1['FTS'] + m2['FTS']) / 2.0
                if variant == "chem":
                    sample_record["runs"].append({
                        "run_id": r,
                        "reply": response_old,
                        "reply_new": response,
                        "predicted_smiles_1": pred1,
                        "predicted_smiles_2": pred2,
                        "results": {
                            "pred1": m1,
                            "pred2": m2,
                            "avg": {
                                "Valid": valid_avg,
                                "Exact_match": exact_avg,
                                "FTS": fts_avg
                            }
                        }
                    })
                else:
                    sample_record["runs"].append({
                        "run_id": r,
                        "reply": response,
                        "predicted_smiles_1": pred1,
                        "predicted_smiles_2": pred2,
                        "results": {
                            "pred1": m1,
                            "pred2": m2,
                            "avg": {
                                "Valid": valid_avg,
                                "Exact_match": exact_avg,
                                "FTS": fts_avg
                            }
                        }
                    })

                total_cnt += 1
                valid_sum += valid_avg
                exact_sum += exact_avg
                fts_sum += fts_avg

            except Exception as e:
                sample_record["runs"].append({
                    "run_id": r,
                    "error": f"{type(e).__name__}: {e}"
                })

        results.append(sample_record)

    if total_cnt > 0:
        print(f"[SUMMARY] runs={num_runs}, samples={len(df)}")
        print(f"  Avg Valid       : {valid_sum / total_cnt:.4f}")
        print(f"  Avg Exact_match : {exact_sum / total_cnt:.4f}")
        print(f"  Avg FTS         : {fts_sum / total_cnt:.4f}")

    dump_json(output_path, results)
