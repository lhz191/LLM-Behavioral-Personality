#!/usr/bin/env python3
"""Shared helpers for the BFI2C Organization scripts."""

from __future__ import annotations

import itertools
import json
import random
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/shared-storage-user/liuhaoze/llm_behavior_test")
ORG_DIR = ROOT / "conscientiousness/bfi2c/bfi2c_organization"
AXES_DIR = ORG_DIR / "results/axes"
STEERING_DIR = ORG_DIR / "results/steering"

MODELGEN_10X = ORG_DIR / "scenarios_factor_thought_modelgen_10x_v2.json"
OLD_THOUGHT = ORG_DIR / "scenarios_thought.json"
FP_OPEN = ORG_DIR / "scenarios_open.json"
TASK_OPEN = (
    ROOT
    / "llm_specific_tasks_lhz/conscientiousness/disorganized_task(v2有效果，效果不错)/scenarios_thought_only_open.json"
)
TASK_V2 = (
    ROOT
    / "llm_specific_tasks_lhz/conscientiousness/disorganized_task(v2有效果，效果不错)/scenarios_thought_only_v2_attitude.json"
)

FP_SYSTEM = (
    "You are the person in this situation. Start your response with a thought, "
    "then describe what you would actually do in first person and explain why. "
    "Be concrete about your actions."
)
TASK_SYSTEM = "You are an assistant. Start your response with a thought, then give your answer."

FIELD_PRESETS = {
    "thought": ("risky_thought", "safe_thought"),
    "response": ("risky", "safe"),
}


def axis_system_for_file(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    register = payload.get("register")
    if register is None:
        scenarios = payload.get("scenarios", [])
        if scenarios:
            register = scenarios[0].get("register")
    return TASK_SYSTEM if register in {"advice", "task"} else FP_SYSTEM


def load_scenarios(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload["scenarios"]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pairwise_values(diffs: np.ndarray) -> list[float]:
    return [
        cosine(diffs[i], diffs[j])
        for i in range(len(diffs))
        for j in range(i + 1, len(diffs))
    ]


def summarize(vals: list[float]) -> dict:
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
    }


def split_half(diffs: np.ndarray, n_samples: int = 5000, seed: int = 13) -> float | None:
    n = len(diffs)
    if n < 4:
        return None
    half = n // 2
    vals = []
    seen = set()

    if n <= 12:
        iterator = itertools.combinations(range(n), half)
    else:
        rng = random.Random(seed + n)

        def sampled():
            while len(vals) < n_samples:
                yield tuple(sorted(rng.sample(range(n), half)))

        iterator = sampled()

    for left in iterator:
        right = tuple(i for i in range(n) if i not in left)
        key = tuple(sorted((tuple(left), right)))
        if key in seen:
            continue
        seen.add(key)
        vals.append(cosine(np.mean(diffs[list(left)], axis=0), np.mean(diffs[list(right)], axis=0)))
        if n > 12 and len(vals) >= n_samples:
            break
    return float(np.mean(vals)) if vals else None


def parse_coeffs(spec: str) -> list[float]:
    if ":" not in spec:
        vals = [float(x) for x in spec.split(",") if x.strip()]
    else:
        start, stop, step = [float(x) for x in spec.split(":")]
        if step == 0:
            raise ValueError("Coefficient step must not be zero")
        vals = []
        cur = start
        def in_range(value: float) -> bool:
            return value <= stop + 1e-9 if step > 0 else value >= stop - 1e-9

        while in_range(cur):
            vals.append(round(cur, 10))
            cur += step
    return [int(v) if float(v).is_integer() else v for v in vals]


def coeff_key(coeff: float) -> str:
    return f"coeff_{coeff:+.2f}"


def parse_option_choice(text: str) -> str:
    lowered = text.lower()
    patterns = [
        (r"\b(?:i would choose|i choose|i'd choose|i'll choose|i will choose|i would go with|i'll go with|i go with|my choice is|choice:)\s*(?:option\s*)?a\b", "A"),
        (r"\b(?:i would choose|i choose|i'd choose|i'll choose|i will choose|i would go with|i'll go with|i go with|my choice is|choice:)\s*(?:option\s*)?b\b", "B"),
        (r"\boption\s*a\b", "A"),
        (r"\boption\s*b\b", "B"),
    ]
    hits = []
    for pattern, label in patterns:
        match = re.search(pattern, lowered)
        if match:
            hits.append((match.start(), label))
    return sorted(hits)[0][1] if hits else "?"


def import_extractor(layer: int):
    sae_dir = ROOT / "sae_analysis"
    if str(sae_dir) not in sys.path:
        sys.path.insert(0, str(sae_dir))
    sys.argv = [sys.argv[0], str(layer)]
    import extract_format_comparison as ex  # type: ignore

    if ex.LAYER != layer:
        raise RuntimeError(f"extract_format_comparison already loaded at layer {ex.LAYER}; use a fresh process")
    return ex


def extract_diffs(
    scenario_file: Path,
    fields: tuple[str, str],
    *,
    layer: int,
    system_prompt: str = FP_SYSTEM,
    first_n: int | None = None,
) -> tuple[list[str], list[int | None], np.ndarray, dict[str, float]]:
    ex = import_extractor(layer)
    scenarios = load_scenarios(scenario_file)
    if first_n is not None:
        scenarios = scenarios[:first_n]

    pos_field, neg_field = fields
    ids, items, diffs, norms = [], [], [], {}
    print(f"Layer {layer} | {scenario_file} | {pos_field} - {neg_field}")
    for i, scenario in enumerate(scenarios, start=1):
        sid = scenario["id"]
        print(f"[{i}/{len(scenarios)}] {sid}")
        h_pos = ex.extract_hidden(scenario["situation"], scenario[pos_field], system_prompt=system_prompt)
        h_neg = ex.extract_hidden(scenario["situation"], scenario[neg_field], system_prompt=system_prompt)
        diff = (h_pos - h_neg).numpy().astype(np.float32)
        ids.append(sid)
        items.append(scenario.get("bfi2_item"))
        diffs.append(diff)
        norms[sid] = float(np.linalg.norm(diff))
    return ids, items, np.stack(diffs), norms


def axis_metrics(ids: list[str], diffs: np.ndarray, norms: dict[str, float], *, split_samples: int, seed: int) -> dict:
    pairs = pairwise_values(diffs)
    axis = np.mean(diffs, axis=0)
    return {
        "n": len(ids),
        "axis_norm": float(np.linalg.norm(axis)),
        "mean_norm": float(np.mean([norms[sid] for sid in ids])),
        "pairwise": summarize(pairs),
        "split_half": split_half(diffs, split_samples, seed),
    }
