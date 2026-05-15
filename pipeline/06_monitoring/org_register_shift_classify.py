#!/usr/bin/env python3
"""Classify response segment shifts by closest register-specific behavior axis."""

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


@torch.no_grad()
def extract_response_mean(model, tokenizer, situation: str, text: str, system_prompt: str, layer: int, device: str) -> np.ndarray:
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
    return hidden[prompt_len:end, :].mean(dim=0).numpy().astype(np.float32)


@torch.no_grad()
def extract_response_last(model, tokenizer, situation: str, text: str, system_prompt: str, layer: int, device: str) -> np.ndarray:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": situation},
        {"role": "assistant", "content": text},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return outputs.hidden_states[layer + 1][0, -1, :].float().cpu().numpy().astype(np.float32)


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


def build_axis(model, tokenizer, path: Path, preset: str, layer: int, device: str, first_n: int | None) -> dict:
    scenarios = load_scenarios(path)
    if first_n:
        scenarios = scenarios[:first_n]
    system_prompt = axis_system_for_file(path)
    pos_field, neg_field = FIELD_PRESETS[preset]
    diffs = []
    print(f"Axis from {path} ({len(scenarios)} scenarios)")
    for i, scenario in enumerate(scenarios, start=1):
        print(f"  [{i}/{len(scenarios)}] {scenario['id']}")
        pos = extract_response_mean(model, tokenizer, scenario["situation"], scenario[pos_field], system_prompt, layer, device)
        neg = extract_response_mean(model, tokenizer, scenario["situation"], scenario[neg_field], system_prompt, layer, device)
        diffs.append(pos - neg)
    axis = np.mean(diffs, axis=0)
    return {
        "path": str(path),
        "system_prompt": system_prompt,
        "n": len(scenarios),
        "axis_norm": float(np.linalg.norm(axis)),
        "unit_axis": unit(axis),
    }


def score_shift(shift: np.ndarray, axes: dict[str, dict]) -> dict:
    scores = {}
    for name, info in axes.items():
        cos = cosine(shift, info["unit_axis"])
        scores[name] = {
            "cos": cos,
            "abs_cos": abs(cos),
            "pole": "risky/low-pole" if cos >= 0 else "safe/high-pole",
        }
    best_abs = max(scores, key=lambda name: scores[name]["abs_cos"])
    best_pos = max(scores, key=lambda name: scores[name]["cos"])
    return {"scores": scores, "best_abs_axis": best_abs, "best_positive_axis": best_pos}


def eval_file(
    model,
    tokenizer,
    path: Path,
    expected_axis: str,
    preset: str,
    axes: dict[str, dict],
    layer: int,
    device: str,
    first_n: int | None,
) -> dict:
    scenarios = load_scenarios(path)
    if first_n:
        scenarios = scenarios[:first_n]
    system_prompt = axis_system_for_file(path)
    pos_field, neg_field = FIELD_PRESETS[preset]
    records = []
    methods = {
        "segment_mean_raw": {"correct_abs": 0, "correct_pos": 0},
        "segment_mean_shift": {"correct_abs": 0, "correct_pos": 0},
        "last_token_shift": {"correct_abs": 0, "correct_pos": 0},
    }
    for i, scenario in enumerate(scenarios, start=1):
        print(f"Eval {expected_axis} [{i}/{len(scenarios)}] {scenario['id']}")
        base = prompt_last(model, tokenizer, scenario["situation"], system_prompt, layer, device)
        for label, field in (("risky", pos_field), ("safe", neg_field)):
            segment_mean = extract_response_mean(model, tokenizer, scenario["situation"], scenario[field], system_prompt, layer, device)
            segment_last = extract_response_last(model, tokenizer, scenario["situation"], scenario[field], system_prompt, layer, device)
            scored_by_method = {
                "segment_mean_raw": score_shift(segment_mean, axes),
                "segment_mean_shift": score_shift(segment_mean - base, axes),
                "last_token_shift": score_shift(segment_last - base, axes),
            }
            for method, scored in scored_by_method.items():
                methods[method]["correct_abs"] += int(scored["best_abs_axis"] == expected_axis)
                methods[method]["correct_pos"] += int(scored["best_positive_axis"] == expected_axis)
            records.append(
                {
                    "id": scenario["id"],
                    "segment_label": label,
                    "expected_axis": expected_axis,
                    "methods": scored_by_method,
                }
            )
    n = len(records)
    return {
        "expected_axis": expected_axis,
        "n_segments": n,
        "methods": {
            method: {
                "best_abs_accuracy": counts["correct_abs"] / n if n else None,
                "best_positive_accuracy": counts["correct_pos"] / n if n else None,
            }
            for method, counts in methods.items()
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-file", action="append", required=True, help="name=path. Repeatable.")
    parser.add_argument("--eval-file", action="append", required=True, help="name=path. Repeatable.")
    parser.add_argument("--preset", choices=sorted(FIELD_PRESETS), default="thought")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--first-n-axis", type=int)
    parser.add_argument("--first-n-eval", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(SAE_BASE))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Layer {args.layer} | Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    axes = {}
    axis_meta = {}
    for spec in args.axis_file:
        name, rel = parse_named_file(spec)
        path = ROOT / rel if not rel.is_absolute() else rel
        axes[name] = build_axis(model, tokenizer, path, args.preset, args.layer, device, args.first_n_axis)
        axis_meta[name] = {k: v for k, v in axes[name].items() if k != "unit_axis"}

    evaluations = {}
    for spec in args.eval_file:
        name, rel = parse_named_file(spec)
        path = ROOT / rel if not rel.is_absolute() else rel
        evaluations[name] = eval_file(model, tokenizer, path, name, args.preset, axes, args.layer, device, args.first_n_eval)

    payload = {
        "config": {
            "layer": args.layer,
            "preset": args.preset,
            "axis_files": args.axis_file,
            "eval_files": args.eval_file,
            "first_n_axis": args.first_n_axis,
            "first_n_eval": args.first_n_eval,
            "methods": {
                "segment_mean_raw": "cos(mean_hidden(selected segment tokens | situation), axis)",
                "segment_mean_shift": "cos(mean_hidden(selected segment tokens | situation) - last_hidden(situation_prompt), axis)",
                "last_token_shift": "cos(last_hidden(situation+selected segment) - last_hidden(situation_prompt), axis)",
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
