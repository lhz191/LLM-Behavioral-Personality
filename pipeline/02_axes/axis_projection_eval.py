#!/usr/bin/env python3
"""Evaluate whether situation+thought projections recover risky/safe poles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import (
    FIELD_PRESETS,
    ROOT,
    axis_system_for_file,
    config_axis_scenario_dir,
    config_id,
    import_extractor,
    load_config,
    load_scenarios,
    write_json,
)


def unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


def extract_pair_hiddens(ex, scenarios: list[dict], fields: tuple[str, str], system_prompt: str) -> tuple[np.ndarray, np.ndarray]:
    pos_field, neg_field = fields
    pos, neg = [], []
    for i, scenario in enumerate(scenarios, start=1):
        print(f"[{i}/{len(scenarios)}] {scenario['id']}")
        h_pos = ex.extract_hidden(scenario["situation"], scenario[pos_field], system_prompt=system_prompt).numpy()
        h_neg = ex.extract_hidden(scenario["situation"], scenario[neg_field], system_prompt=system_prompt).numpy()
        pos.append(h_pos.astype(np.float32))
        neg.append(h_neg.astype(np.float32))
    return np.stack(pos), np.stack(neg)


def summarize(vals: list[float]) -> dict:
    arr = np.array(vals, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def evaluate(pos: np.ndarray, neg: np.ndarray, ids: list[str], *, leave_one_out: bool) -> dict:
    diffs = pos - neg
    records = []
    pair_correct = 0
    endpoint_correct = 0
    margins = []

    for i, sid in enumerate(ids):
        if leave_one_out:
            idx = [j for j in range(len(ids)) if j != i]
            axis = unit(np.mean(diffs[idx], axis=0))
            midpoint = (np.mean(pos[idx], axis=0) + np.mean(neg[idx], axis=0)) / 2
        else:
            axis = unit(np.mean(diffs, axis=0))
            midpoint = (np.mean(pos, axis=0) + np.mean(neg, axis=0)) / 2

        pos_proj = float(np.dot(pos[i] - midpoint, axis))
        neg_proj = float(np.dot(neg[i] - midpoint, axis))
        margin = pos_proj - neg_proj
        margins.append(margin)
        pair_ok = margin > 0
        endpoint_ok = pos_proj > 0 and neg_proj < 0
        pair_correct += int(pair_ok)
        endpoint_correct += int(endpoint_ok)
        records.append(
            {
                "id": sid,
                "risky_projection": pos_proj,
                "safe_projection": neg_proj,
                "pair_margin_risky_minus_safe": margin,
                "pair_correct": pair_ok,
                "endpoint_correct": endpoint_ok,
            }
        )

    return {
        "leave_one_out": leave_one_out,
        "n": len(ids),
        "pair_accuracy": pair_correct / len(ids),
        "endpoint_accuracy": endpoint_correct / len(ids),
        "margin_summary": summarize(margins),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bfi2c_organization", help="Config module name, e.g. bfi2c_organization")
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--preset", choices=sorted(FIELD_PRESETS), default="thought")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    eval_file_arg = args.eval_file or config_axis_scenario_dir(config) / "axis_fp.json"
    eval_file = ROOT / eval_file_arg if not eval_file_arg.is_absolute() else eval_file_arg
    scenarios = load_scenarios(eval_file)
    fields = FIELD_PRESETS[args.preset]
    system_prompt = axis_system_for_file(eval_file)
    ids = [scenario["id"] for scenario in scenarios]

    ex = import_extractor(args.layer)
    pos, neg = extract_pair_hiddens(ex, scenarios, fields, system_prompt)
    diffs = pos - neg
    axis = np.mean(diffs, axis=0)

    payload = {
        "config": {
            "config_id": config_id(config),
            "eval_file": str(eval_file),
            "preset": args.preset,
            "fields": list(fields),
            "layer": args.layer,
            "system_prompt": system_prompt,
            "sign": "positive projection means closer to risky/low-pole endpoint; negative means closer to safe/high-pole endpoint",
        },
        "axis_norm": float(np.linalg.norm(axis)),
        "self_axis_eval": evaluate(pos, neg, ids, leave_one_out=False),
        "leave_one_out_eval": evaluate(pos, neg, ids, leave_one_out=True),
    }

    out = args.out or eval_file.with_name(f"{eval_file.stem}_{args.preset}_projection_eval_l{args.layer}.json")
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
