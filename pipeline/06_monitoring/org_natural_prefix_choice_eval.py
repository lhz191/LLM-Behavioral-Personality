#!/usr/bin/env python3
"""Evaluate whether natural coeff-0 thought prefixes predict final A/B choices."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import FIELD_PRESETS, ROOT, axis_system_for_file, cosine, load_scenarios, parse_option_choice, write_json

SAE_BASE = Path("/mnt/shared-storage-user/liuhaoze/sae_emergent_misalignment/andyrdt_dictionary_learning")
MODEL_PATH = SAE_BASE / "models/Llama-3.1-8B-Instruct"


def unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


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
    # For online monitoring we only use the generated thought-ish prefix before
    # explicit answer markers when they exist.
    match = re.search(
        r"(?:\n\s*Answer:|\bI\s+(?:choose|chose)\s+Option\s+[AB]\b|\bI(?:'ll| will| would)\s+choose\s+Option\s+[AB]\b|\bOption\s+[AB]\s*:)",
        text,
        flags=re.IGNORECASE,
    )
    return text[: match.start()].strip() if match else text.strip()


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


def build_axis(model, tokenizer, axis_file: Path, layer: int, device: str, first_n: int | None) -> dict:
    scenarios = load_scenarios(axis_file)
    if first_n:
        scenarios = scenarios[:first_n]
    system_prompt = axis_system_for_file(axis_file)
    pos_field, neg_field = FIELD_PRESETS["thought"]
    diffs = []
    print(f"Axis from {axis_file} ({len(scenarios)} scenarios)")
    for i, scenario in enumerate(scenarios, start=1):
        print(f"  [{i}/{len(scenarios)}] {scenario['id']}")
        pos = segment_hiddens(model, tokenizer, scenario["situation"], scenario[pos_field], system_prompt, layer, device).mean(axis=0)
        neg = segment_hiddens(model, tokenizer, scenario["situation"], scenario[neg_field], system_prompt, layer, device).mean(axis=0)
        diffs.append(pos - neg)
    axis = np.mean(diffs, axis=0)
    return {
        "path": str(axis_file),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering-file", type=Path, required=True)
    parser.add_argument("--axis-file", type=Path, required=True)
    parser.add_argument("--steering-axis", default="thought", choices=["thought", "response"])
    parser.add_argument("--prefixes", default="3,5,10,20,full")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--first-n-axis", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(SAE_BASE))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    steering_path = ROOT / args.steering_file if not args.steering_file.is_absolute() else args.steering_file
    axis_path = ROOT / args.axis_file if not args.axis_file.is_absolute() else args.axis_file
    prefixes = parse_prefixes(args.prefixes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Layer {args.layer} | Device: {device} | Prefixes: {prefixes}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    axis = build_axis(model, tokenizer, axis_path, args.layer, device, args.first_n_axis)
    steering = __import__("json").load(open(steering_path, encoding="utf-8"))
    target_system = steering["config"]["system_prompt"]

    metrics = {str(prefix): {"correct": 0, "n": 0, "skipped": 0} for prefix in prefixes}
    records = []
    for item in steering["results"]:
        full_text = coeff_zero_response(item, args.steering_axis)
        if not full_text:
            continue
        choice = parse_option_choice(full_text)
        if choice not in {"A", "B"}:
            for prefix in prefixes:
                metrics[str(prefix)]["skipped"] += 1
            records.append({"scenario_id": item["scenario_id"], "choice": choice, "skipped": "unparsed_choice", "text": full_text})
            continue
        thought_text = truncate_at_answer(strip_thought_prefix(full_text))
        if not thought_text:
            thought_text = strip_thought_prefix(full_text)
        hiddens = segment_hiddens(model, tokenizer, item["situation"], "Thought:" + thought_text, target_system, args.layer, device)
        prefix_scores = {}
        expected_sign = 1 if choice == "A" else -1
        for prefix in prefixes:
            vec, used_tokens = prefix_mean(hiddens, prefix)
            score = cosine(vec, axis["unit_axis"])
            pred_choice = "A" if score >= 0 else "B"
            key = str(prefix)
            metrics[key]["n"] += 1
            metrics[key]["correct"] += int((1 if pred_choice == "A" else -1) == expected_sign)
            prefix_scores[key] = {
                "used_tokens": used_tokens,
                "cos": score,
                "pred_choice": pred_choice,
                "correct": pred_choice == choice,
            }
        records.append(
            {
                "scenario_id": item["scenario_id"],
                "choice": choice,
                "thought_prefix_text": thought_text,
                "n_tokens": int(hiddens.shape[0]),
                "prefix_scores": prefix_scores,
            }
        )

    summary = {
        prefix: {
            "n": vals["n"],
            "skipped": vals["skipped"],
            "accuracy": vals["correct"] / vals["n"] if vals["n"] else None,
        }
        for prefix, vals in metrics.items()
    }
    payload = {
        "config": {
            "layer": args.layer,
            "steering_file": str(steering_path),
            "axis_file": str(axis_path),
            "steering_axis": args.steering_axis,
            "prefixes": [str(prefix) for prefix in prefixes],
            "prediction_rule": "positive cosine predicts Option A (low organization); negative predicts Option B (high organization)",
        },
        "axis": {k: v for k, v in axis.items() if k != "unit_axis"},
        "summary": summary,
        "records": records,
    }
    out = ROOT / args.out if not args.out.is_absolute() else args.out
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
