#!/usr/bin/env python3
"""Summarize A/B choices in Organization steering results."""

from __future__ import annotations

import argparse
import sys
import json
from collections import Counter, defaultdict
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import ROOT, coeff_key, parse_option_choice, write_json


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_response_groups(payload: dict):
    results = payload.get("results", [])
    if isinstance(results, dict):
        for axis_name, entries in results.items():
            for entry in entries:
                yield entry["scenario_id"], "axis", axis_name, entry.get("responses", {})
        return

    for scenario in results:
        sid = scenario.get("scenario_id")
        if "axes" in scenario:
            for axis_name, axis_data in scenario["axes"].items():
                yield sid, "axis", axis_name, axis_data.get("responses", {})
        if "features" in scenario:
            for feature_id, feature_data in scenario["features"].items():
                yield sid, "feature", feature_id, feature_data.get("responses", {})
        if "responses" in scenario:
            yield sid, "result", "default", scenario.get("responses", {})


def summarize(payload: dict, base_coeff: float) -> dict:
    base_label = coeff_key(base_coeff)
    by_group: dict[str, dict] = {}
    by_coeff: dict[str, Counter] = defaultdict(Counter)
    base_counts: Counter = Counter()

    for sid, group_type, name, responses in iter_response_groups(payload):
        key = f"{group_type}:{name}"
        by_group.setdefault(key, {"type": group_type, "name": name, "by_coeff": {}, "per_scenario": {}})
        scenario_summary = {}
        for label, item in responses.items():
            choice = parse_option_choice(item.get("text", ""))
            coeff = item.get("coeff")
            scenario_summary[label] = {"coeff": coeff, "choice": choice}
            by_group[key]["by_coeff"].setdefault(label, Counter())
            by_group[key]["by_coeff"][label][choice] += 1
            by_coeff[label][choice] += 1
            if label == base_label:
                base_counts[choice] += 1
        by_group[key]["per_scenario"][sid] = scenario_summary

    normalized_groups = {}
    for key, value in by_group.items():
        normalized_groups[key] = {
            "type": value["type"],
            "name": value["name"],
            "by_coeff": {label: dict(counts) for label, counts in sorted(value["by_coeff"].items())},
            "per_scenario": value["per_scenario"],
        }

    return {
        "base_coeff": base_coeff,
        "base_label": base_label,
        "base_counts": dict(base_counts),
        "overall_by_coeff": {label: dict(counts) for label, counts in sorted(by_coeff.items())},
        "groups": normalized_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-coeff", type=float, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    in_path = ROOT / args.input if not args.input.is_absolute() else args.input
    payload = load_json(in_path)
    out_payload = {
        "source_file": str(in_path),
        "config": payload.get("config", {}),
        "choice_summary": summarize(payload, args.base_coeff),
    }
    out = args.out or in_path.with_name(f"{in_path.stem}_choice_summary.json")
    write_json(out, out_payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
