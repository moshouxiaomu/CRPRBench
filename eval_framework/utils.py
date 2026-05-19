# utils.py
import re
import json
import ast
from typing import List, Optional, Sequence, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# ========= Basics & Parsing =========

def canonical_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to RDKit canonical form; return None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def parse_last_smiles(response: str) -> Optional[str]:
    """Extract the content of the last <SMILES>...</SMILES> block from a response; return None if not found."""
    matches = re.findall(r'<SMILES>(.*?)</SMILES>', response, re.DOTALL)
    return matches[-1].strip() if matches else None


def parse_two_smiles(response: str, order: str = "pair") -> Tuple[Optional[str], Optional[str]]:
    """
    Parse up to two <SMILES> ... </SMILES> entries from a response.
    Returns (pred1, pred2):
      - If >= 2 found:
          order="pair"     -> (second last, last)  (aligned with "paired answers" tasks)
          order="reverse"  -> (last, second last)  (aligned with "any two answers pairing" tasks)
      - If only 1 found: (that_single_value, None)
      - If 0 found:      (None, None)
    """
    matches = re.findall(r'<SMILES>(.*?)</SMILES>', response, re.DOTALL)
    matches = [m.strip() for m in matches]
    if len(matches) >= 2:
        if order == "pair":
            return matches[-2], matches[-1]
        else:
            return matches[-1], matches[-2]
    elif len(matches) == 1:
        return matches[0], None
    else:
        return None, None

# ========= Metric Computation =========

def _metric_invalid() -> Dict[str, float]:
    return {'Valid': 0, 'Exact_match': 0, 'FTS': 0.0}


def evaluate_prediction_against_answer(pred_smiles: Optional[str],
                                       answer: Optional[str]) -> Dict[str, float]:
    """
    Evaluate metrics for "single prediction vs single answer".
    - Valid: whether the prediction is a valid SMILES
    - Exact_match: whether canonical SMILES are exactly equal
    - FTS: Tanimoto similarity with MorganFP (radius=2, nBits=2048)
    Note: answer can be None (meaning no requirement at that position).
    """
    # Answer missing: only check whether the prediction format is valid
    if answer is None:
        if not pred_smiles:
            return _metric_invalid()
        pred_mol = Chem.MolFromSmiles(pred_smiles)
        return {'Valid': int(pred_mol is not None), 'Exact_match': 0, 'FTS': 0.0}

    # Handle answer validity
    ans_mol = Chem.MolFromSmiles(answer) if answer else None
    if ans_mol is None:
        raise ValueError(f"ERROR SMILES：{answer}")
    ans_canon = Chem.MolToSmiles(ans_mol)

    # Handle prediction validity
    if not pred_smiles:
        return _metric_invalid()
    pred_mol = Chem.MolFromSmiles(pred_smiles)
    if pred_mol is None:
        return _metric_invalid()
    pred_canon = Chem.MolToSmiles(pred_mol)

    # Compute metrics
    exact = int(pred_canon == ans_canon)
    fp_pred = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048)
    fp_ans = AllChem.GetMorganFingerprintAsBitVect(ans_mol, 2, nBits=2048)
    fts = DataStructs.TanimotoSimilarity(fp_pred, fp_ans)
    return {'Valid': 1, 'Exact_match': exact, 'FTS': float(fts)}


def evaluate_single_vs_list(pred_smiles: Optional[str],
                            true_smiles_list: Sequence[str]) -> Dict[str, float]:
    """
    Evaluate the best metric for "single prediction vs multiple acceptable answers" (Exact prioritized, then FTS).
    """
    if not pred_smiles:
        return _metric_invalid()
    pred_mol = Chem.MolFromSmiles(pred_smiles)
    if pred_mol is None:
        return _metric_invalid()
    pred_canon = Chem.MolToSmiles(pred_mol)
    fp_pred = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048)

    best = {'Valid': 1, 'Exact_match': 0, 'FTS': 0.0}
    for true_s in true_smiles_list:
        if not true_s:
            continue
        ans_mol = Chem.MolFromSmiles(true_s)
        if ans_mol is None:
            continue
        ans_canon = Chem.MolToSmiles(ans_mol)
        exact = int(pred_canon == ans_canon)
        fp_ans = AllChem.GetMorganFingerprintAsBitVect(ans_mol, 2, nBits=2048)
        fts = float(DataStructs.TanimotoSimilarity(fp_pred, fp_ans))
        # Prioritize exact match first, then FTS
        if (exact > best['Exact_match']) or (exact == best['Exact_match'] and fts > best['FTS']):
            best['Exact_match'] = exact
            best['FTS'] = fts
    return best


def evaluate_two_vs_paired_answers(pred1: Optional[str],
                                   pred2: Optional[str],
                                   answer_pairs: Sequence[Sequence[Optional[str]]]
                                   ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate "two predictions vs a list of paired answers", taking the best pairing
    (compute metrics for each pair of answers and pick the best).
    Selection rule: first maximize (pred1.exact + pred2.exact)/2, then maximize average FTS.
    """
    best_em, best_fts = -1.0, -1.0
    best_m1, best_m2 = _metric_invalid(), _metric_invalid()
    for pair in answer_pairs:
        a1 = pair[0] if len(pair) > 0 else None
        a2 = pair[1] if len(pair) > 1 else None
        m1 = evaluate_prediction_against_answer(pred1, a1)
        m2 = evaluate_prediction_against_answer(pred2, a2)
        avg_em = (m1['Exact_match'] + m2['Exact_match']) / 2.0
        avg_fts = (m1['FTS'] + m2['FTS']) / 2.0
        if (avg_em > best_em) or (avg_em == best_em and avg_fts > best_fts):
            best_em, best_fts = avg_em, avg_fts
            best_m1, best_m2 = m1, m2
    return best_m1, best_m2


def evaluate_two_vs_answer_list(pred1: Optional[str],
                                pred2: Optional[str],
                                answers: Sequence[str]
                                ) -> Tuple[Dict[str, float], Dict[str, float], int, float]:
    """
    Evaluate the best matching for "two predictions vs an unpaired list of answers":
    - From answers, pick two distinct answers to pair with (pred1, pred2) (check all permutations),
      and select the best by prioritizing total_exact, then avg_fts.
    Returns: (m1, m2, total_exact_matches, avg_fts_over_pair)
    """
    best_total_em, best_avg_fts = -1, -1.0
    best_m1, best_m2 = _metric_invalid(), _metric_invalid()

    # Same as the original script: if duplicate answers exist, still compare by permutations
    for i in range(len(answers)):
        for j in range(len(answers)):
            if i == j:
                continue
            a1, a2 = answers[i], answers[j]
            m1 = evaluate_prediction_against_answer(pred1, a1)
            m2 = evaluate_prediction_against_answer(pred2, a2)
            total_em = m1['Exact_match'] + m2['Exact_match']
            avg_fts = (m1['FTS'] + m2['FTS']) / 2.0
            if (total_em > best_total_em) or (total_em == best_total_em and avg_fts > best_avg_fts):
                best_total_em, best_avg_fts = total_em, avg_fts
                best_m1, best_m2 = m1, m2

    return best_m1, best_m2, best_total_em, best_avg_fts

# ========= IO =========

def read_answer_list_literal(s: str):
    """Safely parse a CSV string column into a Python list/nested list."""
    return ast.literal_eval(s)


def dump_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
