#!/usr/bin/env python3
"""
summarize_results.py
读取 5 个任务的 JSON 评测结果，统计每次 run 的指标和整体平均指标。
用法: python3 summarize_results.py --output-dir ./eval_outputs --num-runs 2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 指标提取
# ─────────────────────────────────────────────────────────────────────────────

def _extract_run_metrics(run: dict) -> Optional[Tuple[float, float, float]]:
    """
    从单条 run 记录中提取 (valid, exact_match, fts)。
    兼容两种格式:
      - eval_1/3/4/5: results -> { Valid, Exact_match, FTS }
      - eval_2:       results -> { pred1, pred2, avg: { Valid, Exact_match, FTS } }
    """
    if "error" in run:
        return None

    results = run.get("results")
    if results is None:
        return None

    # eval_2 格式: results.avg
    if "avg" in results:
        avg = results["avg"]
        return (
            float(avg.get("Valid", 0)),
            float(avg.get("Exact_match", 0)),
            float(avg.get("FTS", 0.0)),
        )

    # eval_1/3/4/5 格式: results 直接含指标
    if "Valid" in results:
        return (
            float(results.get("Valid", 0)),
            float(results.get("Exact_match", 0)),
            float(results.get("FTS", 0.0)),
        )

    return None


def compute_task_stats(result_path: str, num_runs: int) -> dict:
    """
    读取单个任务的 JSON 结果文件，按 run_id 分组统计指标。
    返回结构:
      {
        "task_file":   "task_1_result.json",
        "num_samples": 100,
        "per_run": {
            "1": {"valid": 0.95, "exact_match": 0.60, "fts": 0.82, "count": 100},
            "2": {...}
        },
        "overall": {"valid": ..., "exact_match": ..., "fts": ..., "count": 200}
      }
    """
    with open(result_path, "r", encoding="utf-8") as f:
        data: List[dict] = json.load(f)

    # 按 run_id 分组累加
    run_accum: Dict[int, Dict[str, float]] = {
        rid: {"valid": 0.0, "exact_match": 0.0, "fts": 0.0, "count": 0}
        for rid in range(1, num_runs + 1)
    }
    overall: Dict[str, float] = {"valid": 0.0, "exact_match": 0.0, "fts": 0.0, "count": 0}

    for sample in data:
        for run in sample.get("runs", []):
            rid = run.get("run_id", 1)
            metrics = _extract_run_metrics(run)
            if metrics is None:
                continue
            v, em, fts = metrics
            if rid in run_accum:
                run_accum[rid]["valid"]       += v
                run_accum[rid]["exact_match"] += em
                run_accum[rid]["fts"]         += fts
                run_accum[rid]["count"]       += 1
            overall["valid"]       += v
            overall["exact_match"] += em
            overall["fts"]         += fts
            overall["count"]       += 1

    def _avg(d: dict) -> dict:
        c = d["count"]
        if c == 0:
            return {"valid": 0.0, "exact_match": 0.0, "fts": 0.0, "count": 0}
        return {
            "valid":       round(d["valid"]       / c, 6),
            "exact_match": round(d["exact_match"] / c, 6),
            "fts":         round(d["fts"]         / c, 6),
            "count":       c,
        }

    return {
        "task_file":   os.path.basename(result_path),
        "num_samples": len(data),
        "per_run":     {str(rid): _avg(acc) for rid, acc in run_accum.items()},
        "overall":     _avg(overall),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 打印汇总表格
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{v:.4f}"


def print_summary_table(all_stats: List[dict], num_runs: int) -> None:
    sep = "=" * 78
    print(sep)
    print(f"{'EVALUATION SUMMARY':^78}")
    print(sep)

    for stat in all_stats:
        task_name = stat["task_file"].replace("_result.json", "")
        print(f"\n  Task: {task_name}  (samples={stat['num_samples']})")
        print(f"  {'Run':<10} {'Valid':>10} {'Exact_match':>14} {'FTS':>10} {'Count':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*14} {'-'*10} {'-'*8}")

        for rid in range(1, num_runs + 1):
            r = stat["per_run"].get(str(rid), {})
            print(
                f"  {'Run ' + str(rid):<10}"
                f"{_fmt(r.get('valid', 0)):>10}"
                f"{_fmt(r.get('exact_match', 0)):>14}"
                f"{_fmt(r.get('fts', 0)):>10}"
                f"{r.get('count', 0):>8}"
            )

        ov = stat["overall"]
        print(
            f"  {'Overall':<10}"
            f"{_fmt(ov.get('valid', 0)):>10}"
            f"{_fmt(ov.get('exact_match', 0)):>14}"
            f"{_fmt(ov.get('fts', 0)):>10}"
            f"{ov.get('count', 0):>8}"
        )

    # 全局平均（跨所有任务）
    print(f"\n{sep}")
    print("  GLOBAL AVERAGE (across all tasks)")
    print(f"  {'Metric':<16} {'Value':>10}")
    print(f"  {'-'*16} {'-'*10}")

    total_v = total_em = total_fts = total_cnt = 0.0
    for stat in all_stats:
        ov = stat["overall"]
        c = ov.get("count", 0)
        total_v   += ov.get("valid", 0)       * c
        total_em  += ov.get("exact_match", 0) * c
        total_fts += ov.get("fts", 0)         * c
        total_cnt += c

    if total_cnt > 0:
        print(f"  {'Valid':<16} {_fmt(total_v   / total_cnt):>10}")
        print(f"  {'Exact_match':<16} {_fmt(total_em  / total_cnt):>10}")
        print(f"  {'FTS':<16} {_fmt(total_fts / total_cnt):>10}")
        print(f"  {'Total count':<16} {int(total_cnt):>10}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results across all tasks")
    parser.add_argument("--output-dir",   default="./eval_outputs",
                        help="Directory containing task_N_result.json files")
    parser.add_argument("--num-runs",     type=int, default=2,
                        help="Number of runs per sample (must match evaluation setting)")
    parser.add_argument("--summary-file", default="",
                        help="Path to write the JSON summary (optional)")
    parser.add_argument("--file-suffix", default="",
                    help="Suffix of task result files, e.g. ChemDFM-R for task_1_result_ChemDFM-R.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_stats: List[dict] = []

    file_suffix = args.file_suffix.strip()

    for task_id in range(1, 6):
        if file_suffix:
            result_file = output_dir / f"task_{task_id}_result_{file_suffix}.json"
        else:
            result_file = output_dir / f"task_{task_id}_result.json"

        if not result_file.exists():
            print(f"[WARN] 结果文件不存在，跳过: {result_file}")
            continue
        try:
            stat = compute_task_stats(str(result_file), args.num_runs)
            all_stats.append(stat)
            print(f"[OK] 已读取 Task {task_id}: {result_file}")
        except Exception as e:
            print(f"[ERROR] 读取 Task {task_id} 失败: {e}")

    if not all_stats:
        print("[ERROR] 没有可用的评测结果，退出。")
        return

    print_summary_table(all_stats, args.num_runs)

    if args.summary_file:
        with open(args.summary_file, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] JSON 汇总已写入: {args.summary_file}")


if __name__ == "__main__":
    main()