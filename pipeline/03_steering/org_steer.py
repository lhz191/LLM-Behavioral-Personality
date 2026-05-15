#!/usr/bin/env python3
"""Steer Organization thought/response axes into FP-open or task-open prompts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import (
    FIELD_PRESETS,
    FP_OPEN,
    FP_SYSTEM,
    MODELGEN_10X,
    STEERING_DIR,
    TASK_OPEN,
    TASK_SYSTEM,
    TASK_V2,
    axis_metrics,
    axis_system_for_file,
    cosine,
    extract_diffs,
    import_extractor,
    load_scenarios,
    parse_coeffs,
    write_json,
)


TARGETS = {
    "fp_open": (FP_OPEN, FP_SYSTEM),
    "task_open": (TASK_OPEN, TASK_SYSTEM),
    "task_v2": (TASK_V2, TASK_SYSTEM),
}


def axis_names(spec: str) -> list[str]:
    return ["thought", "response"] if spec == "both" else [x.strip() for x in spec.split(",") if x.strip()]


def select_prompts(path: Path, first_n: int | None) -> list[dict]:
    scenarios = load_scenarios(path)
    return scenarios[:first_n] if first_n is not None else scenarios


def generate_steered(ex, prompt: str, axis: np.ndarray, coeff: float, system_prompt: str, max_new_tokens: int, prefix: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    with torch.no_grad():
        formatted = ex.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
        inputs = ex.tokenizer(formatted, return_tensors="pt").to(ex.model.device)

        handles = []
        if coeff != 0:
            steering = torch.tensor(axis, dtype=torch.float32)

            def hook(_module, _inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                hidden = hidden + coeff * steering.to(hidden.device, hidden.dtype)
                return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

            handles.append(ex.model.model.layers[ex.LAYER].register_forward_hook(hook))

        try:
            out_ids = ex.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=ex.tokenizer.eos_token_id,
            )
        finally:
            for handle in handles:
                handle.remove()

    text = ex.tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return prefix + text


def run_axis(ex, name: str, axis: np.ndarray, prompts: list[dict], coeffs: list[float], system_prompt: str, max_new_tokens: int, prefix: str) -> list[dict]:
    results = []
    for i, scenario in enumerate(prompts, start=1):
        print(f"[{name}] [{i}/{len(prompts)}] {scenario['id']}")
        entry = {
            "scenario_id": scenario["id"],
            "situation": scenario["situation"],
            "ai_construct": scenario.get("ai_construct"),
            "reference_risky": scenario.get("risky"),
            "reference_safe": scenario.get("safe"),
            "responses": {},
        }
        for coeff in coeffs:
            started = time.time()
            text = generate_steered(ex, scenario["situation"], axis, coeff, system_prompt, max_new_tokens, prefix)
            elapsed = time.time() - started
            label = f"coeff_{coeff:+.2f}"
            entry["responses"][label] = {
                "coeff": coeff,
                "n_words": len(text.split()),
                "time_s": round(elapsed, 1),
                "text": text,
            }
            print(f"  {label:>11} ({elapsed:4.1f}s, {len(text.split()):3d}w): {text[:100].replace(chr(10), ' ')}...")
        results.append(entry)
    return results


def grouped_results(axis_results: dict[str, list[dict]], prompts: list[dict]) -> list[dict]:
    """Group outputs by scenario so thought/response axes are easy to compare."""
    by_id: dict[str, dict] = {}
    for scenario in prompts:
        sid = scenario["id"]
        by_id[sid] = {
            "scenario_id": sid,
            "situation": scenario["situation"],
            "ai_construct": scenario.get("ai_construct"),
            "reference_risky": scenario.get("risky"),
            "reference_safe": scenario.get("safe"),
            "axes": {},
        }

    for axis_name, entries in axis_results.items():
        for entry in entries:
            by_id[entry["scenario_id"]]["axes"][axis_name] = {
                "responses": entry["responses"],
            }

    return list(by_id.values())


def compact_name(path: Path) -> str:
    stem = path.stem
    known = {
        "scenarios_factor_thought_preference_v2_10x": "fp",
        "scenarios_task_factor_thought_preference_v2_n40": "task",
        "scenarios_advice_factor_thought_preference_v2_n40": "advice",
        "scenarios_task_choice_probe_v2_first5": "task_choice",
        "scenarios_fp_choice_probe_v2_first5": "fp_choice",
        "scenarios_task_natural_choice_probe_v2_first5": "task_natural_choice",
        "scenarios_fp_natural_choice_probe_v2_first5": "fp_natural_choice",
    }
    if stem in known:
        return known[stem]
    for prefix in ("scenarios_", "scenario_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    return stem.replace("_factor_thought_preference_v2", "").replace("_preference_v2", "")


def coeff_label(coeffs: list[float]) -> str:
    if not coeffs:
        return "coeffs"
    return f"m{abs(coeffs[0]):g}_p{abs(coeffs[-1]):g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-source", type=Path, default=MODELGEN_10X)
    parser.add_argument("--axes", default="both", help="thought, response, or both")
    parser.add_argument("--target", choices=sorted(TARGETS), default="task_open")
    parser.add_argument("--prompt-file", type=Path, help="Override target prompt file")
    parser.add_argument("--system", choices=["fp", "task"], help="Override generation system prompt")
    parser.add_argument("--first-n", type=int, default=4)
    parser.add_argument("--coeffs", default="-5:5:1")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--forced-prefix", default="Thought:")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    target_file, target_system = TARGETS[args.target]
    if args.prompt_file:
        target_file = args.prompt_file
    if args.system:
        target_system = FP_SYSTEM if args.system == "fp" else TASK_SYSTEM

    selected_axes = axis_names(args.axes)
    coeffs = parse_coeffs(args.coeffs)
    prompts = select_prompts(target_file, args.first_n)
    axis_system = axis_system_for_file(args.axis_source)

    bundles = {}
    payload = {
        "config": {
            "layer": args.layer,
            "axis_source_file": str(args.axis_source),
            "axis_source_system_prompt": axis_system,
            "target": args.target,
            "prompt_file": str(target_file),
            "prompt_subset": f"first {args.first_n}" if args.first_n else "all",
            "system_prompt": target_system,
            "forced_prefix": args.forced_prefix,
            "coefficients": coeffs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "sign": "+coeff = low-organization/ad-hoc pole; -coeff = high-organization/systematic pole",
        },
        "axis_info": {},
        "result_layout": "scenario_grouped",
        "results": [],
    }

    for name in selected_axes:
        ids, _items, diffs, norms = extract_diffs(
            args.axis_source,
            FIELD_PRESETS[name],
            layer=args.layer,
            system_prompt=axis_system,
        )
        axis = np.mean(diffs, axis=0).astype(np.float32)
        unit_axis = axis / (np.linalg.norm(axis) + 1e-12)
        bundles[name] = unit_axis
        payload["axis_info"][name] = {
            "fields": list(FIELD_PRESETS[name]),
            **axis_metrics(ids, diffs, norms, split_samples=5000, seed=13),
        }

    if "thought" in bundles and "response" in bundles:
        payload["axis_info"]["response_vs_thought_cosine"] = cosine(bundles["response"], bundles["thought"])

    ex = import_extractor(args.layer)
    axis_results = {}
    for name, axis in bundles.items():
        axis_results[name] = run_axis(
            ex,
            name,
            axis,
            prompts,
            coeffs,
            target_system,
            args.max_new_tokens,
            args.forced_prefix,
        )
    payload["results"] = grouped_results(axis_results, prompts)

    axis_label = "_".join(selected_axes)
    out = args.out or (
        STEERING_DIR
        / f"{compact_name(args.axis_source)}_{axis_label}_axes_to_{compact_name(target_file)}_first{len(prompts)}_l{args.layer}_{coeff_label(coeffs)}.json"
    )
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
