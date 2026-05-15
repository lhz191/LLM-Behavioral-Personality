#!/usr/bin/env python3
"""Evaluate prompt-conditioned thought shifts against a behavior axis."""

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


@torch.no_grad()
def hidden_for_messages(model, tokenizer, messages: list[dict], layer: int, device: str) -> tuple[torch.Tensor, list[int]]:
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return outputs.hidden_states[layer + 1][0].float().cpu(), inputs["input_ids"][0].cpu().tolist()


@torch.no_grad()
def prompt_last_hidden(model, tokenizer, situation: str, system_prompt: str, layer: int, device: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": situation},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return outputs.hidden_states[layer + 1][0, -1, :].float().cpu()


@torch.no_grad()
def thought_hiddens(model, tokenizer, situation: str, thought: str, system_prompt: str, layer: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": situation},
    ]
    prompt_formatted = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_len = len(tokenizer(prompt_formatted, return_tensors="pt")["input_ids"][0])

    full_messages = prompt_messages + [{"role": "assistant", "content": thought}]
    hidden, ids = hidden_for_messages(model, tokenizer, full_messages, layer, device)
    end = max(prompt_len + 1, len(ids) - 1)
    thought_tokens = hidden[prompt_len:end, :]
    if thought_tokens.shape[0] == 0:
        thought_tokens = hidden[-1:, :]
    return hidden[-1, :], thought_tokens.mean(dim=0)


def build_original_axis(records: list[dict]) -> np.ndarray:
    diffs = [record["risky_thought_mean"] - record["safe_thought_mean"] for record in records]
    return unit(np.mean(diffs, axis=0))


def eval_shift(records: list[dict], axis: np.ndarray, key: str) -> dict:
    pair_correct = 0
    margins = []
    rows = []
    for record in records:
        risky_shift = record[f"risky_{key}"] - record["prompt_last"]
        safe_shift = record[f"safe_{key}"] - record["prompt_last"]
        risky_score = float(np.dot(risky_shift, axis))
        safe_score = float(np.dot(safe_shift, axis))
        risky_cos = cosine(risky_shift, axis)
        safe_cos = cosine(safe_shift, axis)
        margin = risky_score - safe_score
        margins.append(margin)
        ok = margin > 0
        pair_correct += int(ok)
        rows.append(
            {
                "id": record["id"],
                "risky_projection": risky_score,
                "safe_projection": safe_score,
                "risky_cosine": risky_cos,
                "safe_cosine": safe_cos,
                "pair_margin_risky_minus_safe": margin,
                "pair_correct": ok,
            }
        )
    return {
        "n": len(records),
        "pair_accuracy": pair_correct / len(records),
        "margin_summary": summarize(margins),
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(FIELD_PRESETS), default="thought")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.preset != "thought":
        parser.error("org_thought_shift_eval currently supports --preset thought only")

    sys.path.insert(0, str(SAE_BASE))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_file = ROOT / args.eval_file if not args.eval_file.is_absolute() else args.eval_file
    scenarios = load_scenarios(eval_file)
    system_prompt = axis_system_for_file(eval_file)
    pos_field, neg_field = FIELD_PRESETS[args.preset]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Layer {args.layer} | Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    records = []
    for i, scenario in enumerate(scenarios, start=1):
        print(f"[{i}/{len(scenarios)}] {scenario['id']}")
        prompt_last = prompt_last_hidden(model, tokenizer, scenario["situation"], system_prompt, args.layer, device).numpy()
        risky_last, risky_mean = thought_hiddens(model, tokenizer, scenario["situation"], scenario[pos_field], system_prompt, args.layer, device)
        safe_last, safe_mean = thought_hiddens(model, tokenizer, scenario["situation"], scenario[neg_field], system_prompt, args.layer, device)
        records.append(
            {
                "id": scenario["id"],
                "prompt_last": prompt_last.astype(np.float32),
                "risky_last": risky_last.numpy().astype(np.float32),
                "safe_last": safe_last.numpy().astype(np.float32),
                "risky_thought_mean": risky_mean.numpy().astype(np.float32),
                "safe_thought_mean": safe_mean.numpy().astype(np.float32),
            }
        )

    axis = build_original_axis(records)
    payload = {
        "config": {
            "eval_file": str(eval_file),
            "preset": args.preset,
            "fields": [pos_field, neg_field],
            "layer": args.layer,
            "system_prompt": system_prompt,
            "axis": "unit(mean(mean_hidden(risky_thought_tokens) - mean_hidden(safe_thought_tokens)))",
            "score": "projection of prompt-conditioned shift onto axis",
        },
        "last_token_shift_eval": eval_shift(records, axis, "last"),
        "thought_mean_shift_eval": eval_shift(records, axis, "thought_mean"),
    }
    out = args.out or eval_file.with_name(f"{eval_file.stem}_{args.preset}_shift_eval_l{args.layer}.json")
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
