#!/usr/bin/env python3
"""Run centered and/or uncentered PCA on configured behavior-axis diff vectors."""

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
    config_axes_dir,
    config_axis_scenario_dir,
    config_id,
    cosine,
    extract_diffs,
    item_number_field,
    load_config,
    write_json,
)


def uncentered_pca(x: np.ndarray, ids: list[str], n_components: int) -> dict:
    mean_axis = np.mean(x, axis=0)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    energy = (s**2) / np.sum(s**2)
    comps = vt[:n_components].astype(np.float32)
    if cosine(comps[0], mean_axis) < 0:
        comps[0] = -comps[0]
    return {
        "mean_axis_norm": float(np.linalg.norm(mean_axis)),
        "pc1_cosine_with_mean_axis_oriented": cosine(comps[0], mean_axis),
        "component_info": [
            {
                "pc": k,
                "energy_ratio": float(energy[k - 1]),
                "cumulative_energy_ratio": float(np.sum(energy[:k])),
                "cosine_with_mean_axis": cosine(pc, mean_axis),
                "projection_mean": float(np.mean(x @ pc)),
                "projection_std": float(np.std(x @ pc)),
                "top_positive_projection_ids": [ids[i] for i in np.argsort(-(x @ pc))[:5]],
                "top_negative_projection_ids": [ids[i] for i in np.argsort(x @ pc)[:5]],
            }
            for k, pc in enumerate(comps, start=1)
        ],
    }


def centered_pca(x: np.ndarray, ids: list[str], items: list[int | None], n_components: int) -> dict:
    mean_axis = np.mean(x, axis=0)
    centered = x - mean_axis
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    explained = (s**2) / np.sum(s**2)
    comps = vt[:n_components].astype(np.float32)

    groups = defaultdict(list)
    for idx, item in enumerate(items):
        if item is not None:
            groups[str(item)].append(idx)
    item_axes = {item: np.mean(x[idxs], axis=0) for item, idxs in groups.items()}

    info = []
    for k, pc in enumerate(comps, start=1):
        raw_proj = x @ pc
        centered_proj = centered @ pc
        info.append(
            {
                "pc": k,
                "explained_variance_ratio": float(explained[k - 1]),
                "cumulative_explained_variance_ratio": float(np.sum(explained[:k])),
                "cosine_with_mean_axis": cosine(pc, mean_axis),
                "cosine_with_item_axes": {item: cosine(pc, axis) for item, axis in sorted(item_axes.items())},
                "by_item_projection": {
                    item: {
                        "mean_raw_projection": float(np.mean(raw_proj[idxs])),
                        "mean_centered_projection": float(np.mean(centered_proj[idxs])),
                        "std_centered_projection": float(np.std(centered_proj[idxs])),
                    }
                    for item, idxs in sorted(groups.items())
                },
                "top_positive_projection_ids": [ids[i] for i in np.argsort(-centered_proj)[:5]],
                "top_negative_projection_ids": [ids[i] for i in np.argsort(centered_proj)[:5]],
            }
        )

    pc1 = comps[0]
    if cosine(pc1, mean_axis) < 0:
        pc1 = -pc1
    return {
        "mean_axis_norm": float(np.linalg.norm(mean_axis)),
        "pc1_cosine_with_mean_axis_oriented": cosine(pc1, mean_axis),
        "component_info": info,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bfi2c_organization", help="Config module name, e.g. bfi2c_organization")
    parser.add_argument("--scenario-file", type=Path)
    parser.add_argument("--fields", choices=sorted(FIELD_PRESETS), default="thought")
    parser.add_argument("--mode", choices=["centered", "uncentered", "both"], default="both")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    scenario_file = args.scenario_file or config_axis_scenario_dir(config) / "axis_fp.json"
    item_field = item_number_field(config)
    ids, items, diffs, _norms = extract_diffs(
        scenario_file,
        FIELD_PRESETS[args.fields],
        layer=args.layer,
        item_field=item_field,
    )
    payload = {
        "config": {
            "config_id": config_id(config),
            "layer": args.layer,
            "scenario_file": str(scenario_file),
            "fields": list(FIELD_PRESETS[args.fields]),
            "axis_direction": f"{FIELD_PRESETS[args.fields][0]} - {FIELD_PRESETS[args.fields][1]}",
            "item_field": item_field,
            "n_components": args.n_components,
        }
    }
    if args.mode in {"centered", "both"}:
        payload["centered_pca"] = centered_pca(diffs, ids, items, args.n_components)
    if args.mode in {"uncentered", "both"}:
        payload["uncentered_pca"] = uncentered_pca(diffs, ids, args.n_components)

    out = args.out or config_axes_dir(config) / f"{config_id(config)}_{scenario_file.stem}_{args.fields}_{args.mode}_pca_layer{args.layer}.json"
    write_json(out, payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
