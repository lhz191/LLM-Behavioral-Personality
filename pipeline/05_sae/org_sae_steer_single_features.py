#!/usr/bin/env python3
"""Steer Organization task probes with individual SAE decoder features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import ROOT, TASK_SYSTEM, load_scenarios, parse_coeffs, write_json

SAE_BASE = Path("/mnt/shared-storage-user/liuhaoze/sae_emergent_misalignment/andyrdt_dictionary_learning")
MODEL_PATH = SAE_BASE / "models/Llama-3.1-8B-Instruct"


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


def feature_metadata(features_file: Path, feature_ids: list[int]) -> dict[str, dict]:
    if not features_file:
        return {}
    with features_file.open(encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for fid in feature_ids:
        sid = str(fid)
        out[sid] = {}
        for axis_name in ("thought", "response"):
            for item in data["axes"][axis_name]["shared_features"]:
                if item["feature"] == fid:
                    out[sid][axis_name] = item
    return out


def generate_steered(model, tokenizer, prompt: str, vector: torch.Tensor, coeff: float, layer: int, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": TASK_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Thought:"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    handles = []

    if coeff != 0:
        target_layer = model.model.layers[layer]

        def hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            steering = vector.to(hidden.device, hidden.dtype)
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
    return "Thought:" + text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--features-file", type=Path)
    parser.add_argument("--feature-ids", nargs="+", type=int, required=True)
    parser.add_argument("--first-n", type=int, default=5)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--coeffs", default="-5:5:1")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(SAE_BASE))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_path = SAE_BASE / f"pretrained_saes/resid_post_layer_{args.layer}/trainer_1"
    print(f"Layer {args.layer} | Device: {device}")
    print(f"Features: {args.feature_ids}")

    print("Loading model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    print("Loading SAE...")
    sae = BatchTopKSAE.from_pretrained(sae_path / "ae.pt").to(device).eval()
    decoder = get_decoder_weight(sae).detach().float().cpu()

    prompts = load_scenarios(args.prompt_file)
    if args.first_n:
        prompts = prompts[: args.first_n]
    coeffs = parse_coeffs(args.coeffs)
    meta = feature_metadata(args.features_file, args.feature_ids) if args.features_file else {}

    results = []
    for fid in args.feature_ids:
        raw = decoder_column(decoder, fid).detach().float().cpu()
        vector = raw / (raw.norm() + 1e-8)
        print(f"\nFeature F{fid} decoder_norm={float(raw.norm()):.4f}")
        feature_result = {
            "feature_id": fid,
            "decoder_norm": float(raw.norm()),
            "metadata": meta.get(str(fid), {}),
            "scenarios": [],
        }
        for i, scenario in enumerate(prompts, start=1):
            print(f"  [{i}/{len(prompts)}] {scenario['id']}")
            entry = {
                "scenario_id": scenario["id"],
                "situation": scenario["situation"],
                "reference_risky": scenario.get("risky"),
                "reference_safe": scenario.get("safe"),
                "responses": {},
            }
            for coeff in coeffs:
                started = time.time()
                text = generate_steered(model, tokenizer, scenario["situation"], vector, coeff, args.layer, args.max_new_tokens)
                elapsed = time.time() - started
                label = f"coeff_{coeff:+.2f}"
                entry["responses"][label] = {
                    "coeff": coeff,
                    "n_words": len(text.split()),
                    "time_s": round(elapsed, 1),
                    "text": text,
                }
                print(f"    {label:>11} ({elapsed:4.1f}s): {text[:90].replace(chr(10), ' ')}...")
            feature_result["scenarios"].append(entry)
        results.append(feature_result)

    payload = {
        "config": {
            "prompt_file": str(args.prompt_file),
            "features_file": str(args.features_file) if args.features_file else None,
            "feature_ids": args.feature_ids,
            "layer": args.layer,
            "coefficients": coeffs,
            "max_new_tokens": args.max_new_tokens,
            "system_prompt": TASK_SYSTEM,
            "sign": "+coeff adds the SAE decoder feature; -coeff subtracts it",
        },
        "results": results,
    }
    out = ROOT / args.out if not args.out.is_absolute() else args.out
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
