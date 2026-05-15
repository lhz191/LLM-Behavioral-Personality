#!/usr/bin/env python3
"""Classify natural coeff-0 thought prefixes into FP/task/advice axes."""

from __future__ import annotations

import argparse
import json
import re
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
        if item:
            values.append("full" if item.lower() == "full" else int(item))
    return values


def strip_thought_prefix(text: str) -> str:
    return re.sub(r"^\s*Thought:\s*", "", text, flags=re.IGNORECASE)


def truncate_at_answer(text: str) -> str:
    match = re.search(
        r"(?:\n\s*(?:Answer:|Action:|Incident Review:|Vendor Comparison:)|\bI\s+(?:choose|chose)\s+Option\s+[AB]\b|\bI(?:'ll| will| would)\s+choose\s+Option\s+[AB]\b|\bOption\s+[AB]\s*:)",
        text,
        flags=re.IGNORECASE,
    )
    return text[: match.start()].strip() if match else text.strip()


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


def coeff_zero_response(item: dict, axis_name: str) -> str | None:
    responses = item.get("axes", {}).get(axis_name, {}).get("responses", {})
    if "coeff_+0.00" in responses:
        return responses["coeff_+0.00"]["text"]
    for info in responses.values():
        if info.get("coeff") == 0:
            return info.get("text")
    return None


def score(vec: np.ndarray, axes: dict[str, dict]) -> dict:
    scores = {}
    for name, axis in axes.items():
        cos = cosine(vec, axis["unit_axis"])
        scores[name] = {"cos": cos, "abs_cos": abs(cos), "pole": "risky/low-pole" if cos >= 0 else "safe/high-pole"}
    best_abs = max(scores, key=lambda name: scores[name]["abs_cos"])
    return {"scores": scores, "best_abs_axis": best_abs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-file", action="append", required=True, help="name=path. Repeatable.")
    parser.add_argument("--eval-steering", action="append", required=True, help="expected_name=steering_json. Repeatable.")
    parser.add_argument("--steering-axis", default="thought", choices=["thought", "response"])
    parser.add_argument("--prefixes", default="3,5,10,20,full")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--first-n-axis", type=int)
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

    metrics = {
        str(prefix): {
            "prefix_mean_raw": {"correct": 0, "n": 0},
            "prefix_mean_shift": {"correct": 0, "n": 0},
        }
        for prefix in prefixes
    }
    by_expected: dict[str, dict] = {}
    records = []

    for spec in args.eval_steering:
        expected, rel = parse_named_file(spec)
        path = ROOT / rel if not rel.is_absolute() else rel
        steering = json.load(open(path, encoding="utf-8"))
        system_prompt = steering["config"]["system_prompt"]
        by_expected.setdefault(expected, {str(prefix): {"n": 0, "raw_correct": 0, "shift_correct": 0} for prefix in prefixes})
        for item in steering["results"]:
            full_text = coeff_zero_response(item, args.steering_axis)
            if not full_text:
                continue
            thought_text = truncate_at_answer(strip_thought_prefix(full_text))
            if not thought_text:
                thought_text = strip_thought_prefix(full_text)
            base = prompt_last(model, tokenizer, item["situation"], system_prompt, args.layer, device)
            hiddens = segment_hiddens(model, tokenizer, item["situation"], "Thought:" + thought_text, system_prompt, args.layer, device)
            prefix_results = {}
            for prefix in prefixes:
                mean_vec, used_tokens = prefix_mean(hiddens, prefix)
                raw = score(mean_vec, axes)
                shifted = score(mean_vec - base, axes)
                for method, scored in (("prefix_mean_raw", raw), ("prefix_mean_shift", shifted)):
                    metrics[str(prefix)][method]["n"] += 1
                    metrics[str(prefix)][method]["correct"] += int(scored["best_abs_axis"] == expected)
                by_expected[expected][str(prefix)]["n"] += 1
                by_expected[expected][str(prefix)]["raw_correct"] += int(raw["best_abs_axis"] == expected)
                by_expected[expected][str(prefix)]["shift_correct"] += int(shifted["best_abs_axis"] == expected)
                prefix_results[str(prefix)] = {
                    "used_tokens": used_tokens,
                    "prefix_mean_raw": raw,
                    "prefix_mean_shift": shifted,
                }
            records.append(
                {
                    "expected_axis": expected,
                    "scenario_id": item["scenario_id"],
                    "thought_prefix_text": thought_text,
                    "n_tokens": int(hiddens.shape[0]),
                    "prefixes": prefix_results,
                }
            )

    summary = {
        prefix: {
            method: vals["correct"] / vals["n"] if vals["n"] else None
            for method, vals in methods.items()
        }
        for prefix, methods in metrics.items()
    }
    expected_summary = {
        expected: {
            prefix: {
                "n": vals["n"],
                "prefix_mean_raw": vals["raw_correct"] / vals["n"] if vals["n"] else None,
                "prefix_mean_shift": vals["shift_correct"] / vals["n"] if vals["n"] else None,
            }
            for prefix, vals in by_prefix.items()
        }
        for expected, by_prefix in by_expected.items()
    }
    payload = {
        "config": {
            "layer": args.layer,
            "axis_files": args.axis_file,
            "eval_steering": args.eval_steering,
            "steering_axis": args.steering_axis,
            "prefixes": [str(prefix) for prefix in prefixes],
            "first_n_axis": args.first_n_axis,
        },
        "axes": axis_meta,
        "summary": summary,
        "expected_summary": expected_summary,
        "records": records,
    }
    out = ROOT / args.out if not args.out.is_absolute() else args.out
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
