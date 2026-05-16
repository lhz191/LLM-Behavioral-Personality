#!/usr/bin/env python3
"""Compute thought/response diff metrics for a configured behavior axis."""

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
    FIELD_PRESETS,
    axis_metrics,
    axis_system_for_file,
    config_axes_dir,
    config_axis_scenario_dir,
    config_id,
    cosine,
    extract_diffs,
    item_number_field,
    load_config,
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
    parser.add_argument("--config", default="bfi2c_organization", help="Config module name, e.g. bfi2c_organization")
    parser.add_argument("--scenario-file", type=Path)
    parser.add_argument("--fields", default="both", help="thought, response, or both")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--split-half-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--compare-old", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    scenario_file = args.scenario_file or config_axis_scenario_dir(config) / "axis_fp.json"
    item_field = item_number_field(config)
    selected = field_names(args.fields)
    axis_system = axis_system_for_file(scenario_file)
    payload = {
        "config": {
            "config_id": config_id(config),
            "layer": args.layer,
            "scenario_file": str(scenario_file),
            "fields": selected,
            "item_field": item_field,
            "system_prompt": axis_system,
            "split_half_samples": args.split_half_samples,
            "seed": args.seed,
        },
        "metrics": {},
    }

    bundles = {}
    for name in selected:
        fields = FIELD_PRESETS[name]
        ids, items, diffs, norms = extract_diffs(
            scenario_file,
            fields,
            layer=args.layer,
            system_prompt=axis_system,
            item_field=item_field,
        )
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
        old_thought = config_axis_scenario_dir(config) / "axis_fp.json"
        if not old_thought.exists():
            parser.error(f"--compare-old requested, but old thought scenario file is missing: {old_thought}")
        old_ids, _old_items, old_diffs, old_norms = extract_diffs(
            old_thought,
            FIELD_PRESETS["thought"],
            layer=args.layer,
            item_field=item_field,
        )
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

    out = args.out or config_axes_dir(config) / f"{config_id(config)}_{scenario_file.stem}_{'_'.join(selected)}_metrics_layer{args.layer}.json"
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
