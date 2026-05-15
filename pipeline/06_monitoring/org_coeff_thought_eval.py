#!/usr/bin/env python3
"""Evaluate steered thought samples by coefficient labels."""

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

from common_org import FIELD_PRESETS, ROOT, axis_system_for_file, cosine, load_scenarios, parse_option_choice, write_json

SAE_BASE = Path("/mnt/shared-storage-user/liuhaoze/sae_emergent_misalignment/andyrdt_dictionary_learning")
MODEL_PATH = SAE_BASE / "models/Llama-3.1-8B-Instruct"


def unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


def parse_named_file(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected name=path, got {spec}")
    name, path = spec.split("=", 1)
    return name, Path(path)


def parse_coeffs(spec: str) -> list[float]:
    return [float(item.strip()) for item in spec.split(",") if item.strip()]


def coeff_key(coeff: float) -> str:
    return f"coeff_{coeff:+.2f}"


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
def segment_mean(model, tokenizer, situation: str, text: str, system_prompt: str, layer: int, device: str) -> np.ndarray:
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
        pos = segment_mean(model, tokenizer, scenario["situation"], scenario[pos_field], system_prompt, layer, device)
        neg = segment_mean(model, tokenizer, scenario["situation"], scenario[neg_field], system_prompt, layer, device)
        diffs.append(pos - neg)
    axis = np.mean(diffs, axis=0)
    return {
        "path": str(path),
        "system_prompt": system_prompt,
        "n": len(scenarios),
        "axis_norm": float(np.linalg.norm(axis)),
        "unit_axis": unit(axis),
    }


def response_for_coeff(item: dict, axis_name: str, coeff: float) -> str | None:
    responses = item.get("axes", {}).get(axis_name, {}).get("responses", {})
    key = coeff_key(coeff)
    if key in responses:
        return responses[key].get("text")
    for info in responses.values():
        if float(info.get("coeff", float("nan"))) == coeff:
            return info.get("text")
    return None


def score(vec: np.ndarray, axes: dict[str, dict]) -> dict:
    scores = {}
    for name, axis in axes.items():
        cos = cosine(vec, axis["unit_axis"])
        scores[name] = {
            "cos": cos,
            "abs_cos": abs(cos),
            "pole": "risky/low-pole" if cos >= 0 else "safe/high-pole",
        }
    best_abs = max(scores, key=lambda name: scores[name]["abs_cos"])
    return {"scores": scores, "best_abs_axis": best_abs}


def expected_pole(coeff: float) -> str | None:
    if coeff > 0:
        return "risky/low-pole"
    if coeff < 0:
        return "safe/high-pole"
    return None


def expected_pole_from_choice(text: str) -> tuple[str | None, str]:
    choice = parse_option_choice(text)
    if choice == "A":
        return "risky/low-pole", choice
    if choice == "B":
        return "safe/high-pole", choice
    return None, choice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-file", action="append", required=True, help="name=path. Repeatable.")
    parser.add_argument("--eval-steering", action="append", required=True, help="expected_name=steering_json. Repeatable.")
    parser.add_argument("--steering-axis", default="thought", choices=["thought", "response"])
    parser.add_argument("--coeffs", default="-4,-2,0,2,4")
    parser.add_argument("--pole-source", choices=["coeff", "choice"], default="coeff")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--first-n-axis", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(SAE_BASE))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    coeffs = parse_coeffs(args.coeffs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Layer {args.layer} | Device: {device} | Coeffs: {coeffs}")
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

    totals = {"register_correct": 0, "register_n": 0, "pole_correct": 0, "pole_n": 0}
    by_expected: dict[str, dict] = {}
    by_coeff: dict[str, dict] = {}
    records = []

    for spec in args.eval_steering:
        expected_axis, rel = parse_named_file(spec)
        path = ROOT / rel if not rel.is_absolute() else rel
        steering = json.load(open(path, encoding="utf-8"))
        system_prompt = steering["config"]["system_prompt"]
        by_expected.setdefault(expected_axis, {"register_correct": 0, "register_n": 0, "pole_correct": 0, "pole_n": 0})

        for item in steering["results"]:
            for coeff in coeffs:
                full_text = response_for_coeff(item, args.steering_axis, coeff)
                if not full_text:
                    continue
                thought = truncate_at_answer(strip_thought_prefix(full_text))
                if not thought:
                    thought = strip_thought_prefix(full_text)
                vec = segment_mean(model, tokenizer, item["situation"], "Thought:" + thought, system_prompt, args.layer, device)
                scored = score(vec, axes)
                if args.pole_source == "choice":
                    pole, choice = expected_pole_from_choice(full_text)
                else:
                    pole = expected_pole(coeff)
                    choice = None
                expected_score = scored["scores"][expected_axis]

                register_ok = scored["best_abs_axis"] == expected_axis
                totals["register_n"] += 1
                totals["register_correct"] += int(register_ok)
                by_expected[expected_axis]["register_n"] += 1
                by_expected[expected_axis]["register_correct"] += int(register_ok)

                coeff_label = f"{coeff:g}"
                by_coeff.setdefault(coeff_label, {"register_correct": 0, "register_n": 0, "pole_correct": 0, "pole_n": 0})
                by_coeff[coeff_label]["register_n"] += 1
                by_coeff[coeff_label]["register_correct"] += int(register_ok)

                pole_ok = None
                if pole is not None:
                    pole_ok = expected_score["pole"] == pole
                    totals["pole_n"] += 1
                    totals["pole_correct"] += int(pole_ok)
                    by_expected[expected_axis]["pole_n"] += 1
                    by_expected[expected_axis]["pole_correct"] += int(pole_ok)
                    by_coeff[coeff_label]["pole_n"] += 1
                    by_coeff[coeff_label]["pole_correct"] += int(pole_ok)

                records.append(
                    {
                        "expected_axis": expected_axis,
                        "scenario_id": item["scenario_id"],
                        "coeff": coeff,
                        "choice": choice,
                        "expected_pole": pole,
                        "thought_text": thought,
                        "best_abs_axis": scored["best_abs_axis"],
                        "register_correct": register_ok,
                        "expected_axis_pole": expected_score["pole"],
                        "pole_correct": pole_ok,
                        "scores": scored["scores"],
                    }
                )

    def summarize(bucket: dict) -> dict:
        return {
            "register_accuracy": bucket["register_correct"] / bucket["register_n"] if bucket["register_n"] else None,
            "register_n": bucket["register_n"],
            "pole_accuracy": bucket["pole_correct"] / bucket["pole_n"] if bucket["pole_n"] else None,
            "pole_n": bucket["pole_n"],
        }

    payload = {
        "config": {
            "layer": args.layer,
            "axis_files": args.axis_file,
            "eval_steering": args.eval_steering,
            "steering_axis": args.steering_axis,
            "coeffs": coeffs,
            "pole_source": args.pole_source,
            "first_n_axis": args.first_n_axis,
            "pole_rule": "choice source: Option A=risky/low-pole, Option B=safe/high-pole; coeff source: +coeff=risky/low-pole, -coeff=safe/high-pole, 0 excluded",
        },
        "axes": axis_meta,
        "summary": summarize(totals),
        "by_expected_axis": {name: summarize(bucket) for name, bucket in by_expected.items()},
        "by_coeff": {name: summarize(bucket) for name, bucket in by_coeff.items()},
        "records": records,
    }
    out = ROOT / args.out if not args.out.is_absolute() else args.out
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
