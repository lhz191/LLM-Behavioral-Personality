#!/usr/bin/env python3
"""Steer configured thought/response axes into choice or open prompts."""

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
    FP_SYSTEM,
    TASK_SYSTEM,
    ROOT,
    axis_metrics,
    axis_system_for_file,
    config_axis_scenario_dir,
    config_id,
    config_steering_dir,
    config_test_scenario_dir,
    coeff_label,
    cosine,
    extract_diffs,
    import_extractor,
    item_number_field,
    load_config,
    load_scenarios,
    parse_coeffs,
    write_json,
)


CHOICE_TARGET_FILES = {
    "fp_choice": "choice_fp.json",
    "task_choice": "choice_task.json",
    "advice_daily_choice": "choice_advice_daily.json",
    "advice_task_choice": "choice_advice_task.json",
}

TARGET_SYSTEMS = {
    "fp_choice": FP_SYSTEM,
    "task_choice": TASK_SYSTEM,
    "advice_daily_choice": TASK_SYSTEM,
    "advice_task_choice": TASK_SYSTEM,
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
        "axis_fp": "fp",
        "axis_task": "task",
        "axis_advice": "advice",
        "axis_advice_daily": "advice_daily",
        "choice_fp": "fp_choice",
        "choice_task": "task_choice",
        "choice_advice_daily": "advice_daily_choice",
        "choice_advice_task": "advice_task_choice",
    }
    if stem in known:
        return known[stem]
    for prefix in ("scenarios_", "scenario_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    return stem.replace("_factor_thought_preference_v2", "").replace("_preference_v2", "")


def resolve_path(path: Path) -> Path:
    return ROOT / path if not path.is_absolute() else path


def resolve_target(config, target: str) -> tuple[Path, str]:
    if target not in CHOICE_TARGET_FILES:
        raise ValueError(f"Unsupported target: {target}")
    return config_test_scenario_dir(config) / CHOICE_TARGET_FILES[target], TARGET_SYSTEMS[target]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bfi2c_organization", help="Config module name, e.g. bfi2c_organization")
    parser.add_argument("--axis-source", type=Path)
    parser.add_argument("--axes", default="both", help="thought, response, or both")
    parser.add_argument("--target", choices=sorted(CHOICE_TARGET_FILES), default="task_choice")
    parser.add_argument("--prompt-file", type=Path, help="Override target prompt file")
    parser.add_argument("--system", choices=["fp", "task"], help="Override generation system prompt")
    parser.add_argument("--first-n", type=int, default=4)
    parser.add_argument("--coeffs", default="-5:5:1")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--forced-prefix", default="Thought:")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    axis_source = resolve_path(args.axis_source) if args.axis_source else config_axis_scenario_dir(config) / "axis_fp.json"
    target_file, target_system = resolve_target(config, args.target)
    if args.prompt_file:
        target_file = resolve_path(args.prompt_file)
    if args.system:
        target_system = FP_SYSTEM if args.system == "fp" else TASK_SYSTEM

    selected_axes = axis_names(args.axes)
    coeffs = parse_coeffs(args.coeffs)
    prompts = select_prompts(target_file, args.first_n)
    axis_system = axis_system_for_file(axis_source)
    item_field = item_number_field(config)

    bundles = {}
    payload = {
        "config": {
            "config_id": config_id(config),
            "layer": args.layer,
            "axis_source_file": str(axis_source),
            "axis_source_system_prompt": axis_system,
            "target": args.target,
            "prompt_file": str(target_file),
            "prompt_subset": f"first {args.first_n}" if args.first_n else "all",
            "system_prompt": target_system,
            "forced_prefix": args.forced_prefix,
            "coefficients": coeffs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "sign": "+coeff steers toward the first field in FIELD_PRESETS (risky/low-pole); -coeff steers toward the second field (safe/high-pole)",
        },
        "axis_info": {},
        "result_layout": "scenario_grouped",
        "results": [],
    }

    for name in selected_axes:
        ids, _items, diffs, norms = extract_diffs(
            axis_source,
            FIELD_PRESETS[name],
            layer=args.layer,
            system_prompt=axis_system,
            item_field=item_field,
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
        config_steering_dir(config)
        / f"{compact_name(axis_source)}_{axis_label}_axes_to_{compact_name(target_file)}_first{len(prompts)}_l{args.layer}_{coeff_label(coeffs)}.json"
    )
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
