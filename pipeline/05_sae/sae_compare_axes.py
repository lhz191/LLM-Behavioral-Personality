#!/usr/bin/env python3
"""Compare thought vs response axes in SAE feature space."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import (
    FIELD_PRESETS,
    ROOT,
    WORKSPACE_ROOT,
    axis_system_for_file,
    config_axis_scenario_dir,
    config_id,
    config_sae_dir,
    load_config,
    write_json,
)

DEFAULT_SAE_BASE = WORKSPACE_ROOT.parent / "sae_emergent_misalignment/andyrdt_dictionary_learning"
DEFAULT_MODEL_PATH = DEFAULT_SAE_BASE / "models/Llama-3.1-8B-Instruct"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@torch.no_grad()
def extract_hidden(model, tokenizer, prompt: str, response: str, layer: int, system_prompt: str, device: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[layer + 1]

    prompt_formatted = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_len = len(tokenizer(prompt_formatted, return_tensors="pt")["input_ids"][0])
    resp_end = inputs["input_ids"].shape[1] - 1
    if prompt_len >= resp_end:
        raise ValueError(f"Invalid response boundaries: start={prompt_len}, end={resp_end}")
    return hidden[0, prompt_len:resp_end, :].mean(dim=0).float().cpu()


@torch.no_grad()
def sae_features(sae, activation: torch.Tensor, device: str) -> np.ndarray:
    act = activation.to(device).to(torch.float32).unsqueeze(0)
    return sae.encode(act).squeeze(0).cpu().numpy()


def summarize_axis(entries: list[dict], n_features: int, threshold_frac: float) -> dict:
    threshold = max(2, math.ceil(len(entries) * threshold_frac))
    counts: Counter[int] = Counter()
    values: defaultdict[int, list[float]] = defaultdict(list)

    for entry in entries:
        for fid, val in entry["features"].items():
            counts[fid] += 1
            values[fid].append(val)

    mean_vec = np.zeros(n_features, dtype=np.float32)
    for fid, vals in values.items():
        mean_vec[fid] = float(np.mean(vals))

    shared = []
    for fid, count in counts.items():
        if count < threshold:
            continue
        vals = values[fid]
        mean = float(np.mean(vals))
        shared.append(
            {
                "feature": int(fid),
                "freq": int(count),
                "n": len(entries),
                "mean_diff": mean,
                "abs_mean_diff": abs(mean),
                "direction": "risky_up" if mean > 0 else "safe_up",
            }
        )
    shared.sort(key=lambda x: (x["freq"], x["abs_mean_diff"]), reverse=True)

    return {
        "n_scenarios": len(entries),
        "n_features": n_features,
        "threshold_frac": threshold_frac,
        "threshold_count": threshold,
        "mean_sae_axis_norm": float(np.linalg.norm(mean_vec)),
        "n_unique_features": len(counts),
        "n_shared_features": len(shared),
        "shared_features": shared[:80],
        "_mean_vec": mean_vec,
    }


def strip_internal(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if not k.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bfi2c_organization", help="Config module name, e.g. bfi2c_organization")
    parser.add_argument("--scenario-file", type=Path)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sae-base", type=Path, default=Path(os.environ.get("SAE_BASE", DEFAULT_SAE_BASE)))
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    scenario_file = args.scenario_file or config_axis_scenario_dir(config) / "axis_fp.json"
    scenario_file = ROOT / scenario_file if not scenario_file.is_absolute() else scenario_file
    system_prompt = axis_system_for_file(scenario_file)

    sys.path.insert(0, str(args.sae_base))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_path = args.sae_base / f"pretrained_saes/resid_post_layer_{args.layer}/trainer_1"
    print(f"Layer {args.layer} | Device: {device}")
    print(f"Scenario file: {scenario_file}")
    print(f"SAE: {sae_path}")

    with scenario_file.open(encoding="utf-8") as f:
        scenarios = json.load(f)["scenarios"]

    print("Loading model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    print("Loading SAE...")
    sae = BatchTopKSAE.from_pretrained(sae_path / "ae.pt").to(device).eval()

    fields = FIELD_PRESETS
    axis_entries: dict[str, list[dict]] = {name: [] for name in fields}
    n_features: int | None = None

    for i, scenario in enumerate(scenarios, start=1):
        print(f"[{i:02d}/{len(scenarios)}] {scenario['id']}")
        for axis_name, (pos_field, neg_field) in fields.items():
            h_pos = extract_hidden(model, tokenizer, scenario["situation"], scenario[pos_field], args.layer, system_prompt, device)
            h_neg = extract_hidden(model, tokenizer, scenario["situation"], scenario[neg_field], args.layer, system_prompt, device)
            f_pos = sae_features(sae, h_pos, device)
            f_neg = sae_features(sae, h_neg, device)
            diff = f_pos - f_neg
            if n_features is None:
                n_features = len(diff)
            active = np.where(np.abs(diff) > 1e-6)[0]
            vals = diff[active]
            order = np.argsort(np.abs(vals))[::-1]
            features = {int(active[j]): float(vals[j]) for j in order}
            axis_entries[axis_name].append(
                {
                    "scenario_id": scenario["id"],
                    "n_diff_features": len(features),
                    "features": features,
                }
            )
            print(f"  {axis_name:<8s} diff_features={len(features):4d} norm={np.linalg.norm(diff):.3f}")

    assert n_features is not None
    thought_summary = summarize_axis(axis_entries["thought"], n_features, args.threshold)
    response_summary = summarize_axis(axis_entries["response"], n_features, args.threshold)

    thought_vec = thought_summary["_mean_vec"]
    response_vec = response_summary["_mean_vec"]
    thought_shared = {x["feature"]: x for x in thought_summary["shared_features"]}
    response_shared = {x["feature"]: x for x in response_summary["shared_features"]}
    overlap = sorted(set(thought_shared) & set(response_shared))
    union = set(thought_shared) | set(response_shared)
    overlap_rows = []
    for fid in overlap:
        t = thought_shared[fid]
        r = response_shared[fid]
        overlap_rows.append(
            {
                "feature": fid,
                "thought_freq": t["freq"],
                "response_freq": r["freq"],
                "thought_mean_diff": t["mean_diff"],
                "response_mean_diff": r["mean_diff"],
                "same_direction": (t["mean_diff"] > 0) == (r["mean_diff"] > 0),
            }
        )
    overlap_rows.sort(key=lambda x: min(x["thought_freq"], x["response_freq"]), reverse=True)

    payload = {
        "config": {
            "config_id": config_id(config),
            "scenario_file": str(scenario_file),
            "layer": args.layer,
            "system_prompt": system_prompt,
            "sae_path": str(sae_path),
            "model_path": str(args.model_path),
            "threshold": args.threshold,
            "fields": fields,
            "sign": "positive = first field in FIELD_PRESETS stronger; negative = second field stronger",
        },
        "comparison": {
            "mean_sae_axis_cosine": cosine(thought_vec, response_vec),
            "thought_shared_count": thought_summary["n_shared_features"],
            "response_shared_count": response_summary["n_shared_features"],
            "shared_feature_overlap": len(overlap),
            "shared_feature_jaccard": len(overlap) / (len(union) + 1e-12),
            "overlap_features": overlap_rows[:80],
        },
        "axes": {
            "thought": strip_internal(thought_summary),
            "response": strip_internal(response_summary),
        },
        "per_scenario": {
            "thought": axis_entries["thought"],
            "response": axis_entries["response"],
        },
    }
    out = args.out or config_sae_dir(config) / f"{config_id(config)}_{scenario_file.stem}_sae_axis_features_layer{args.layer}.json"
    write_json(ROOT / out if not out.is_absolute() else out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
