#!/usr/bin/env python3
"""Regroup steering JSONs by scenario for side-by-side inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import ROOT, write_json


def load_json(path: Path) -> dict:
    import json

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def regroup_axes(payload: dict) -> list[dict]:
    grouped: dict[str, dict] = {}
    for axis_name, entries in payload.get("results", {}).items():
        for entry in entries:
            sid = entry["scenario_id"]
            grouped.setdefault(
                sid,
                {
                    "scenario_id": sid,
                    "situation": entry.get("situation"),
                    "ai_construct": entry.get("ai_construct"),
                    "reference_risky": entry.get("reference_risky"),
                    "reference_safe": entry.get("reference_safe"),
                    "axes": {},
                },
            )
            grouped[sid]["axes"][axis_name] = {"responses": entry.get("responses", {})}
    return list(grouped.values())


def regroup_features(payload: dict) -> list[dict]:
    grouped: dict[str, dict] = {}
    for feature_result in payload.get("results", []):
        fid = str(feature_result["feature_id"])
        for entry in feature_result.get("scenarios", []):
            sid = entry["scenario_id"]
            grouped.setdefault(
                sid,
                {
                    "scenario_id": sid,
                    "situation": entry.get("situation"),
                    "reference_risky": entry.get("reference_risky"),
                    "reference_safe": entry.get("reference_safe"),
                    "features": {},
                },
            )
            grouped[sid]["features"][fid] = {
                "metadata": feature_result.get("metadata", {}),
                "decoder_norm": feature_result.get("decoder_norm"),
                "responses": entry.get("responses", {}),
            }
    return list(grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kind", choices=["auto", "axes", "features"], default="auto")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    in_path = ROOT / args.input if not args.input.is_absolute() else args.input
    payload = load_json(in_path)

    kind = args.kind
    if kind == "auto":
        results = payload.get("results")
        kind = "axes" if isinstance(results, dict) else "features"

    grouped = regroup_axes(payload) if kind == "axes" else regroup_features(payload)
    out_payload = {
        "config": payload.get("config", {}),
        "source_file": str(in_path),
        "result_layout": f"scenario_grouped_{kind}",
        "results": grouped,
    }
    out = args.out or in_path.with_name(f"{in_path.stem}_grouped.json")
    write_json(out, out_payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
