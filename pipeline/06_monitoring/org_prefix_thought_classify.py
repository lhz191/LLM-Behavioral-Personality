#!/usr/bin/env python3
"""Classify behavior-axis alignment from early thought-token prefixes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import FIELD_PRESETS, ROOT, axis_system_for_file, cosine, load_scenarios, write_json

SAE_BASE = Path("/mnt/shared-storage-user/liuhaoze/sae_emergent_misalignment/andyrdt_dictionary_learning")
MODEL_PATH = SAE_BASE / "models/Llama-3.1-8B-Instruct"


def unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


def parse_named_file(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected name=path, got {spec}")
    name, path = spec.split("=", 1)
    return name, Path(path)


def parse_prefixes(spec: str) -> list[int | str]:
    values: list[int | str] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        values.append("full" if item.lower() == "full" else int(item))
    return values


@torch.no_grad()
def prompt_last(model, tokenizer, situation: str, system_prompt: str, layer: int, device: str) -> np.ndarray:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": situation},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return outputs.hidden_states[layer + 1][0, -1, :].float().cpu().numpy().astype(np.float32)


@torch.no_grad()
def segment_hiddens(model, tokenizer, situation: str, text: str, system_prompt: str, layer: int, device: str) -> np.ndarray:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": situation},
        {"role": "assistant", "content": text},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[layer + 1][0].float().cpu()

    prompt_formatted = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_len = len(tokenizer(prompt_formatted, return_tensors="pt")["input_ids"][0])
    end = max(prompt_len + 1, inputs["input_ids"].shape[1] - 1)
    return hidden[prompt_len:end, :].numpy().astype(np.float32)


def prefix_mean(hiddens: np.ndarray, prefix: int | str) -> tuple[np.ndarray, int]:
    n_tokens = hiddens.shape[0]
    k = n_tokens if prefix == "full" else min(int(prefix), n_tokens)
    k = max(k, 1)
    return hiddens[:k].mean(axis=0), k


def build_axis(model, tokenizer, path: Path, layer: int, device: str, first_n: int | None) -> dict:
    scenarios = load_scenarios(path)
    if first_n:
        scenarios = scenarios[:first_n]
    system_prompt = axis_system_for_file(path)
    pos_field, neg_field = FIELD_PRESETS["thought"]
    diffs = []
    print(f"Axis from {path} ({len(scenarios)} scenarios)")
    for i, scenario in enumerate(scenarios, start=1):
        print(f"  [{i}/{len(scenarios)}] {scenario['id']}")
        pos = segment_hiddens(model, tokenizer, scenario["situation"], scenario[pos_field], system_prompt, layer, device).mean(axis=0)
        neg = segment_hiddens(model, tokenizer, scenario["situation"], scenario[neg_field], system_prompt, layer, device).mean(axis=0)
        diffs.append(pos - neg)
    axis = np.mean(diffs, axis=0)
    return {
        "path": str(path),
        "system_prompt": system_prompt,
        "n": len(scenarios),
        "axis_norm": float(np.linalg.norm(axis)),
        "unit_axis": unit(axis),
    }


def score_vector(vec: np.ndarray, axes: dict[str, dict]) -> dict:
    scores = {}
    for name, info in axes.items():
        cos = cosine(vec, info["unit_axis"])
        scores[name] = {
            "cos": cos,
            "abs_cos": abs(cos),
            "pole": "risky/low-pole" if cos >= 0 else "safe/high-pole",
        }
    best_abs = max(scores, key=lambda name: scores[name]["abs_cos"])
    return {"scores": scores, "best_abs_axis": best_abs}


def eval_file(
    model,
    tokenizer,
    path: Path,
    expected_axis: str,
    axes: dict[str, dict],
    prefixes: list[int | str],
    layer: int,
    device: str,
    first_n: int | None,
) -> dict:
    scenarios = load_scenarios(path)
    if first_n:
        scenarios = scenarios[:first_n]
    system_prompt = axis_system_for_file(path)
    pos_field, neg_field = FIELD_PRESETS["thought"]

    metrics = {
        str(prefix): {
            "prefix_mean_raw": {"correct_axis": 0, "correct_pole": 0},
            "prefix_mean_shift": {"correct_axis": 0, "correct_pole": 0},
        }
        for prefix in prefixes
    }
    records = []

    for i, scenario in enumerate(scenarios, start=1):
        print(f"Eval {expected_axis} [{i}/{len(scenarios)}] {scenario['id']}")
        base = prompt_last(model, tokenizer, scenario["situation"], system_prompt, layer, device)
        for label, field in (("risky", pos_field), ("safe", neg_field)):
            hiddens = segment_hiddens(model, tokenizer, scenario["situation"], scenario[field], system_prompt, layer, device)
            prefix_results = {}
            for prefix in prefixes:
                mean_vec, used_tokens = prefix_mean(hiddens, prefix)
                scored_by_method = {
                    "prefix_mean_raw": score_vector(mean_vec, axes),
                    "prefix_mean_shift": score_vector(mean_vec - base, axes),
                }
                for method, scored in scored_by_method.items():
                    expected_score = scored["scores"][expected_axis]
                    expected_pole = "risky/low-pole" if label == "risky" else "safe/high-pole"
                    bucket = metrics[str(prefix)][method]
                    bucket["correct_axis"] += int(scored["best_abs_axis"] == expected_axis)
                    bucket["correct_pole"] += int(expected_score["pole"] == expected_pole)
                prefix_results[str(prefix)] = {
                    "used_tokens": used_tokens,
                    "methods": scored_by_method,
                }
            records.append(
                {
                    "id": scenario["id"],
                    "thought_label": label,
                    "expected_axis": expected_axis,
                    "n_tokens": int(hiddens.shape[0]),
                    "prefixes": prefix_results,
                }
            )

    n = len(records)
    return {
        "expected_axis": expected_axis,
        "n_thoughts": n,
        "prefix_metrics": {
            prefix: {
                method: {
                    "best_abs_axis_accuracy": counts["correct_axis"] / n if n else None,
                    "expected_axis_pole_accuracy": counts["correct_pole"] / n if n else None,
                }
                for method, counts in by_method.items()
            }
            for prefix, by_method in metrics.items()
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-file", action="append", required=True, help="name=path. Repeatable.")
    parser.add_argument("--eval-file", action="append", required=True, help="name=path. Repeatable.")
    parser.add_argument("--prefixes", default="3,5,10,20,full")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--first-n-axis", type=int)
    parser.add_argument("--first-n-eval", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(SAE_BASE))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prefixes = parse_prefixes(args.prefixes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Layer {args.layer} | Device: {device} | Prefixes: {prefixes}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    axes = {}
    axis_meta = {}
    for spec in args.axis_file:
        name, rel = parse_named_file(spec)
        path = ROOT / rel if not rel.is_absolute() else rel
        axes[name] = build_axis(model, tokenizer, path, args.layer, device, args.first_n_axis)
        axis_meta[name] = {k: v for k, v in axes[name].items() if k != "unit_axis"}

    evaluations = {}
    for spec in args.eval_file:
        name, rel = parse_named_file(spec)
        path = ROOT / rel if not rel.is_absolute() else rel
        evaluations[name] = eval_file(model, tokenizer, path, name, axes, prefixes, args.layer, device, args.first_n_eval)

    payload = {
        "config": {
            "layer": args.layer,
            "axis_files": args.axis_file,
            "eval_files": args.eval_file,
            "prefixes": [str(prefix) for prefix in prefixes],
            "first_n_axis": args.first_n_axis,
            "first_n_eval": args.first_n_eval,
            "methods": {
                "prefix_mean_raw": "cos(mean_hidden(first k thought tokens | situation), axis)",
                "prefix_mean_shift": "cos(mean_hidden(first k thought tokens | situation) - last_hidden(situation_prompt), axis)",
            },
        },
        "axes": axis_meta,
        "evaluations": evaluations,
    }
    out = ROOT / args.out if not args.out.is_absolute() else args.out
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
