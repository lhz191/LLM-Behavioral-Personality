#!/usr/bin/env python3
"""Steer prompts with SAE shared-feature axes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import FP_SYSTEM, ROOT, TASK_SYSTEM, WORKSPACE_ROOT, config_id, config_sae_dir, load_config, load_scenarios, parse_coeffs, write_json

DEFAULT_SAE_BASE = WORKSPACE_ROOT.parent / "sae_emergent_misalignment/andyrdt_dictionary_learning"
DEFAULT_MODEL_PATH = DEFAULT_SAE_BASE / "models/Llama-3.1-8B-Instruct"


def get_decoder_weight(sae) -> torch.Tensor:
    if hasattr(sae, "decoder") and hasattr(sae.decoder, "weight"):
        return sae.decoder.weight
    if hasattr(sae, "W_dec"):
        return sae.W_dec
    raise AttributeError("Could not find SAE decoder weight")


def decoder_column(decoder: torch.Tensor, feature_id: int) -> torch.Tensor:
    if decoder.shape[1] > decoder.shape[0]:
        return decoder[:, feature_id]
    return decoder[feature_id, :]


def shared_features(entries: list[dict], threshold_frac: float) -> dict[int, dict]:
    threshold = max(2, math.ceil(len(entries) * threshold_frac))
    counts: Counter[int] = Counter()
    vals: defaultdict[int, list[float]] = defaultdict(list)
    for entry in entries:
        for fid_str, value in entry["features"].items():
            fid = int(fid_str)
            counts[fid] += 1
            vals[fid].append(float(value))

    out = {}
    for fid, count in counts.items():
        if count >= threshold:
            mean_diff = float(np.mean(vals[fid]))
            out[fid] = {
                "freq": int(count),
                "n": len(entries),
                "mean_diff": mean_diff,
                "direction": "risky_up" if mean_diff > 0 else "safe_up",
            }
    return dict(sorted(out.items(), key=lambda kv: (kv[1]["freq"], abs(kv[1]["mean_diff"])), reverse=True))


def build_sae_axis(decoder: torch.Tensor, features: dict[int, dict]) -> tuple[torch.Tensor, float]:
    pieces = []
    for fid, info in features.items():
        pieces.append(decoder_column(decoder, fid).detach().float().cpu() * float(info["mean_diff"]))
    if not pieces:
        raise ValueError("No shared features selected")
    raw = torch.stack(pieces).sum(dim=0)
    raw_norm = float(raw.norm())
    return raw / (raw.norm() + 1e-8), raw_norm


def generate_steered(
    model,
    tokenizer,
    prompt: str,
    axis: torch.Tensor,
    coeff: float,
    layer: int,
    max_new_tokens: int,
    system_prompt: str,
    forced_prefix: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + forced_prefix
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    handles = []

    if coeff != 0:
        target_layer = model.model.layers[layer]

        def hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            steering = axis.to(hidden.device, hidden.dtype)
            hidden = hidden + coeff * steering
            return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

        handles.append(target_layer.register_forward_hook(hook))

    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()

    text = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return forced_prefix + text


def run_axis(
    model,
    tokenizer,
    axis_name: str,
    axis: torch.Tensor,
    prompts: list[dict],
    coeffs: list[float],
    layer: int,
    max_new_tokens: int,
    system_prompt: str,
    forced_prefix: str,
) -> list[dict]:
    results = []
    for i, scenario in enumerate(prompts, start=1):
        print(f"[{axis_name}] [{i}/{len(prompts)}] {scenario['id']}")
        entry = {
            "scenario_id": scenario["id"],
            "situation": scenario["situation"],
            "reference_risky": scenario.get("risky"),
            "reference_safe": scenario.get("safe"),
            "responses": {},
        }
        for coeff in coeffs:
            started = time.time()
            text = generate_steered(
                model,
                tokenizer,
                scenario["situation"],
                axis,
                coeff,
                layer,
                max_new_tokens,
                system_prompt,
                forced_prefix,
            )
            elapsed = time.time() - started
            label = f"coeff_{coeff:+.2f}"
            entry["responses"][label] = {
                "coeff": coeff,
                "n_words": len(text.split()),
                "time_s": round(elapsed, 1),
                "text": text,
            }
            print(f"  {label:>11} ({elapsed:4.1f}s): {text[:100].replace(chr(10), ' ')}...")
        results.append(entry)
    return results


def grouped_results(axis_results: dict[str, list[dict]], prompts: list[dict]) -> list[dict]:
    by_id = {
        scenario["id"]: {
            "scenario_id": scenario["id"],
            "situation": scenario["situation"],
            "reference_risky": scenario.get("risky"),
            "reference_safe": scenario.get("safe"),
            "axes": {},
        }
        for scenario in prompts
    }
    for axis_name, entries in axis_results.items():
        for entry in entries:
            by_id[entry["scenario_id"]]["axes"][axis_name] = {"responses": entry["responses"]}
    return list(by_id.values())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bfi2c_organization", help="Config module name, e.g. bfi2c_organization")
    parser.add_argument("--features-file", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--system", choices=["fp", "task"], default="task")
    parser.add_argument("--first-n", type=int, default=5)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--coeffs", default="-30:30:10")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--forced-prefix", default="Thought:")
    parser.add_argument("--sae-base", type=Path, default=Path(os.environ.get("SAE_BASE", DEFAULT_SAE_BASE)))
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    system_prompt = FP_SYSTEM if args.system == "fp" else TASK_SYSTEM

    sys.path.insert(0, str(args.sae_base))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

    features_file = ROOT / args.features_file if not args.features_file.is_absolute() else args.features_file
    with features_file.open(encoding="utf-8") as f:
        feature_data = json.load(f)

    selected = {}
    for axis_name in ("thought", "response"):
        selected[axis_name] = shared_features(feature_data["per_scenario"][axis_name], args.threshold)
        print(f"{axis_name}: {len(selected[axis_name])} shared features at threshold {args.threshold}")
        for fid, info in list(selected[axis_name].items())[:12]:
            print(f"  F{fid}: freq={info['freq']}/{info['n']} mean={info['mean_diff']:+.4f} {info['direction']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_path = args.sae_base / f"pretrained_saes/resid_post_layer_{args.layer}/trainer_1"
    print(f"Layer {args.layer} | Device: {device}")
    print("Loading model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    print("Loading SAE...")
    sae = BatchTopKSAE.from_pretrained(sae_path / "ae.pt").to(device).eval()
    decoder = get_decoder_weight(sae).detach().float().cpu()
    axes = {}
    axis_info = {}
    for axis_name, feats in selected.items():
        axis, raw_norm = build_sae_axis(decoder, feats)
        axes[axis_name] = axis
        axis_info[axis_name] = {
            "n_shared_features": len(feats),
            "shared_features": {str(fid): info for fid, info in feats.items()},
            "raw_decoder_sum_norm": raw_norm,
            "unit_norm": float(axis.norm()),
        }
    axis_info["response_vs_thought_sae_axis_cosine"] = cosine(axes["response"], axes["thought"])

    prompt_file = ROOT / args.prompt_file if not args.prompt_file.is_absolute() else args.prompt_file
    prompts = load_scenarios(prompt_file)
    if args.first_n:
        prompts = prompts[: args.first_n]
    coeffs = parse_coeffs(args.coeffs)

    axis_results = {}
    for axis_name, axis in axes.items():
        axis_results[axis_name] = run_axis(
            model,
            tokenizer,
            axis_name,
            axis,
            prompts,
            coeffs,
            args.layer,
            args.max_new_tokens,
            system_prompt,
            args.forced_prefix,
        )

    payload = {
        "config": {
            "config_id": config_id(config),
            "features_file": str(features_file),
            "prompt_file": str(prompt_file),
            "layer": args.layer,
            "threshold": args.threshold,
            "sae_path": str(sae_path),
            "model_path": str(args.model_path),
            "coefficients": coeffs,
            "max_new_tokens": args.max_new_tokens,
            "system_prompt": system_prompt,
            "forced_prefix": args.forced_prefix,
            "sign": "+coeff follows positive shared-feature mean_diff direction; -coeff follows negative direction",
        },
        "axis_info": axis_info,
        "result_layout": "scenario_grouped",
        "results": grouped_results(axis_results, prompts),
    }
    out = args.out or config_sae_dir(config) / f"{config_id(config)}_{prompt_file.stem}_shared_features_l{args.layer}.json"
    out = ROOT / out if not out.is_absolute() else out
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
