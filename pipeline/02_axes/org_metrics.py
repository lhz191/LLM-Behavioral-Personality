#!/usr/bin/env python3
"""Compute Organization thought/response diff metrics."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import (
    AXES_DIR,
    FIELD_PRESETS,
    MODELGEN_10X,
    OLD_THOUGHT,
    axis_metrics,
    axis_system_for_file,
    cosine,
    extract_diffs,
    pairwise_values,
    split_half,
    write_json,
)


def field_names(spec: str) -> list[str]:
    return ["thought", "response"] if spec == "both" else [x.strip() for x in spec.split(",") if x.strip()]


def by_item_metrics(ids: list[str], items: list[int | None], diffs: np.ndarray, norms: dict[str, float], samples: int, seed: int) -> tuple[dict, dict]:
    groups = defaultdict(list)
    for idx, item in enumerate(items):
        if item is not None:
            groups[str(item)].append(idx)

    metrics = {}
    axes = {}
    for item, idxs in sorted(groups.items()):
        arr = diffs[idxs]
        axes[item] = np.mean(arr, axis=0)
        vals = pairwise_values(arr)
        metrics[item] = {
            "n": len(idxs),
            "ids": [ids[i] for i in idxs],
            "axis_norm": float(np.linalg.norm(axes[item])),
            "mean_norm": float(np.mean([norms[ids[i]] for i in idxs])),
            "pairwise_mean": float(np.mean(vals)),
            "pairwise_median": float(np.median(vals)),
            "pairwise_min": float(np.min(vals)),
            "pairwise_max": float(np.max(vals)),
            "split_half": split_half(arr, samples, seed),
        }
    return metrics, axes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-file", type=Path, default=MODELGEN_10X)
    parser.add_argument("--fields", default="both", help="thought, response, or both")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--split-half-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--compare-old", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    selected = field_names(args.fields)
    axis_system = axis_system_for_file(args.scenario_file)
    payload = {
        "config": {
            "layer": args.layer,
            "scenario_file": str(args.scenario_file),
            "fields": selected,
            "system_prompt": axis_system,
            "split_half_samples": args.split_half_samples,
            "seed": args.seed,
        },
        "metrics": {},
    }

    bundles = {}
    for name in selected:
        fields = FIELD_PRESETS[name]
        ids, items, diffs, norms = extract_diffs(args.scenario_file, fields, layer=args.layer, system_prompt=axis_system)
        bundles[name] = (ids, items, diffs, norms)
        item_metrics, item_axes = by_item_metrics(ids, items, diffs, norms, args.split_half_samples, args.seed)
        item_axis_cos = {
            left: {right: cosine(item_axes[left], item_axes[right]) for right in sorted(item_axes)}
            for left in sorted(item_axes)
        }
        payload["metrics"][name] = {
            "fields": list(fields),
            "overall": axis_metrics(ids, diffs, norms, split_samples=args.split_half_samples, seed=args.seed),
            "by_item": item_metrics,
            "item_axis_cosine_matrix": item_axis_cos,
        }

    if "thought" in bundles and "response" in bundles:
        payload["response_vs_thought"] = {
            "axis_cosine": cosine(np.mean(bundles["response"][2], axis=0), np.mean(bundles["thought"][2], axis=0))
        }

    if args.compare_old and "thought" in bundles:
        if not OLD_THOUGHT.exists():
            parser.error(f"--compare-old requested, but old thought scenario file is missing: {OLD_THOUGHT}")
        old_ids, _old_items, old_diffs, old_norms = extract_diffs(OLD_THOUGHT, FIELD_PRESETS["thought"], layer=args.layer)
        payload["old_4item_thought"] = axis_metrics(
            old_ids,
            old_diffs,
            old_norms,
            split_samples=args.split_half_samples,
            seed=args.seed,
        )
        payload["metrics"]["thought"]["overall"]["cosine_with_old_4item_thought_axis"] = cosine(
            np.mean(bundles["thought"][2], axis=0),
            np.mean(old_diffs, axis=0),
        )

    out = args.out or AXES_DIR / f"bfi2c_org_{args.scenario_file.stem}_{'_'.join(selected)}_metrics_layer{args.layer}.json"
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
