#!/usr/bin/env python3
"""Compute metrics and PCA diagnostics in one file per behavior axis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from axis_metrics import by_item_metrics, field_names
from axis_pca import centered_pca, uncentered_pca
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
    write_json,
)


def nested_get(data: dict, path: tuple[str, ...]):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def compare_metric(field_payloads: dict[str, dict], path: tuple[str, ...]) -> dict:
    return {field: nested_get(payload, path) for field, payload in field_payloads.items()}


def pca_summary(pca_payloads: dict[str, dict]) -> dict:
    summary = {}
    for mode, variance_key in (
        ("centered_pca", "explained_variance_ratio"),
        ("uncentered_pca", "energy_ratio"),
    ):
        available = {field: payload.get(mode) for field, payload in pca_payloads.items() if payload.get(mode)}
        if not available:
            continue
        summary[mode] = {
            "mean_axis_norm": {field: payload.get("mean_axis_norm") for field, payload in available.items()},
            "pc1_cosine_with_mean_axis_oriented": {
                field: payload.get("pc1_cosine_with_mean_axis_oriented") for field, payload in available.items()
            },
            "pc1_variance_or_energy": {
                field: payload["component_info"][0].get(variance_key) for field, payload in available.items()
            },
            "pc1_cumulative_variance_or_energy": {
                field: payload["component_info"][0].get(f"cumulative_{variance_key}") for field, payload in available.items()
            },
        }
    return summary


def build_comparison(field_payloads: dict[str, dict], pca_payloads: dict[str, dict]) -> dict:
    pairwise_keys = ("mean", "median", "min", "max", "p10", "p25", "p75", "p90")
    overall = {
        "n": compare_metric(field_payloads, ("overall", "n")),
        "axis_norm": compare_metric(field_payloads, ("overall", "axis_norm")),
        "mean_norm": compare_metric(field_payloads, ("overall", "mean_norm")),
        "split_half": compare_metric(field_payloads, ("overall", "split_half")),
    }
    overall.update(
        {f"pairwise_{key}": compare_metric(field_payloads, ("overall", "pairwise", key)) for key in pairwise_keys}
    )
    return {
        "overall": overall,
        "pca": pca_summary(pca_payloads),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bfi2c_organization", help="Config module name, e.g. bfi2c_organization")
    parser.add_argument("--scenario-file", type=Path)
    parser.add_argument("--fields", default="both", help="thought, response, or both")
    parser.add_argument("--pca-mode", choices=["centered", "uncentered", "both"], default="both")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--split-half-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    scenario_file = args.scenario_file or config_axis_scenario_dir(config) / "axis_fp.json"
    item_field = item_number_field(config)
    selected = field_names(args.fields)
    axis_system = axis_system_for_file(scenario_file)

    field_payloads = {}
    pca_payloads = {}
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
        field_payloads[name] = {
            "fields": list(fields),
            "overall": axis_metrics(ids, diffs, norms, split_samples=args.split_half_samples, seed=args.seed),
            "by_item": item_metrics,
            "item_axis_cosine_matrix": {
                left: {right: cosine(item_axes[left], item_axes[right]) for right in sorted(item_axes)}
                for left in sorted(item_axes)
            },
        }

        pca_payloads[name] = {}
        if args.pca_mode in {"centered", "both"}:
            pca_payloads[name]["centered_pca"] = centered_pca(diffs, ids, items, args.n_components)
        if args.pca_mode in {"uncentered", "both"}:
            pca_payloads[name]["uncentered_pca"] = uncentered_pca(diffs, ids, args.n_components)

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
            "pca_mode": args.pca_mode,
            "n_components": args.n_components,
        },
        "comparison": build_comparison(field_payloads, pca_payloads),
        "details_by_field": {
            field: {
                **field_payloads[field],
                "pca": pca_payloads[field],
            }
            for field in selected
        },
    }

    if "thought" in bundles and "response" in bundles:
        payload["comparison"]["response_vs_thought"] = {
            "axis_cosine": cosine(np.mean(bundles["response"][2], axis=0), np.mean(bundles["thought"][2], axis=0))
        }

    out = args.out or config_axes_dir(config) / f"{config_id(config)}_{scenario_file.stem}_diagnostics_layer{args.layer}.json"
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
