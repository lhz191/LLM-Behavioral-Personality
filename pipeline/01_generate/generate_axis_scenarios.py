#!/usr/bin/env python3
"""Generate contrastive behavior-axis scenarios from a reusable config."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType

from openai import OpenAI

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import SCENARIO_DIR, WORKSPACE_ROOT, write_json


DEFAULT_MODEL = "claude-opus-4-6"


def load_config(name: str) -> ModuleType:
    module_name = name if "." in name else f"configs.{name}"
    return importlib.import_module(module_name)


def output_register(config: ModuleType, register: str) -> str:
    return getattr(config, "OUTPUT_REGISTER_OVERRIDES", {}).get(register, register)


def item_anchored_registers(config: ModuleType) -> set[str]:
    return set(getattr(config, "ITEM_ANCHORED_REGISTERS", {"fp"}))


def thought_extra(config: ModuleType, register: str) -> str:
    return getattr(config, "THOUGHT_EXTRAS_BY_REGISTER", {}).get(register, "")


def item_number_field(config: ModuleType) -> str:
    return getattr(config, "ITEM_NUMBER_FIELD", "item_number")


def item_text_field(config: ModuleType) -> str:
    return getattr(config, "ITEM_TEXT_FIELD", "item_text")


def build_prompt(config: ModuleType, item_number: int, item_text: str, n: int, register: str) -> list[dict[str, str]]:
    system = "You are generating controlled contrastive scenarios for studying LLM behavioral modes. Return valid JSON only."
    schema_register = output_register(config, register)
    user = f"""{config.FACTOR_SPEC}

Original questionnaire item anchor:
- item_number: {item_number}
- item_text: {item_text}

Generate exactly {n} {register} scenarios for this item anchor.

Critical rule:
The original questionnaire item is ONLY a situation-coverage anchor. All risky_thought and safe_thought fields must instantiate the same fixed behavioral factor above.

Schema:
{{
  "scenarios": [
    {{
      "id": "{config.ID_PREFIX}##_short_name",
      "register": "{schema_register}",
      "{item_number_field(config)}": {item_number},
      "{item_text_field(config)}": "{item_text}",
      "situation": "...",
      "risky": "...",
      "safe": "...",
      "risky_thought": "...",
      "safe_thought": "..."
    }}
  ]
}}

{config.REGISTER_REQUIREMENTS[register]}

{thought_extra(config, register)}

{config.RISKY_THOUGHT_RULE}

{config.SAFE_THOUGHT_RULE}

{config.CONTRAST_RULE}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_freeform_prompt(config: ModuleType, register: str, n: int, start_index: int) -> list[dict[str, str]]:
    system = "You are generating controlled contrastive scenarios for studying LLM behavioral modes. Return valid JSON only."
    domain_instruction = getattr(config, "FREEFORM_DOMAINS", {}).get(register, "")
    user = f"""{config.FACTOR_SPEC}

Generate exactly {n} {register} scenarios from scratch.

Do NOT use questionnaire items as anchors. Do NOT create questionnaire-like situations.
Create diverse modern assistant-use situations. {domain_instruction}

Critical rule:
All risky_thought and safe_thought fields must instantiate the same fixed behavioral factor above.

Schema:
{{
  "scenarios": [
    {{
      "id": "{config.FREEFORM_ID_PREFIX}_{register}_{start_index:02d}_short_name",
      "register": "{output_register(config, register)}",
      "{item_number_field(config)}": null,
      "{item_text_field(config)}": null,
      "situation": "...",
      "risky": "...",
      "safe": "...",
      "risky_thought": "...",
      "safe_thought": "..."
    }}
  ]
}}

ID rule:
- Number the scenarios consecutively starting at {config.FREEFORM_ID_PREFIX}_{register}_{start_index:02d}.
- Use lowercase snake_case short names.

{config.REGISTER_REQUIREMENTS[register]}

{thought_extra(config, register)}

{config.RISKY_THOUGHT_RULE}

{config.SAFE_THOUGHT_RULE}

{config.CONTRAST_RULE}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def validation_errors(config: ModuleType, payload: dict, item_number: int | None, n: int, register: str, content_checks: bool = True) -> list[str]:
    errors = []
    expected_register = output_register(config, register)
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != n:
        return [f"Expected {n} scenarios for item {item_number}"]
    number_field = item_number_field(config)
    text_field = item_text_field(config)
    required = {"id", number_field, text_field, "situation", "risky", "safe", "risky_thought", "safe_thought"}
    for scenario in scenarios:
        missing = required - set(scenario)
        if missing:
            errors.append(f"{scenario.get('id', '<unknown>')} missing {sorted(missing)}")
            continue
        if scenario.get("register") != expected_register:
            errors.append(f"{scenario['id']} has wrong register: {scenario.get('register')}")
        if item_number is not None and scenario[number_field] != item_number:
            errors.append(f"{scenario['id']} has wrong {number_field}")
        if item_number is None and scenario[number_field] is not None:
            errors.append(f"{scenario['id']} should not use a questionnaire item anchor")
        if "Thought:" in scenario["risky_thought"] or "Thought:" in scenario["safe_thought"]:
            errors.append(f"{scenario['id']} contains Thought: prefix")
        risky_thought = scenario["risky_thought"].lower()
        if content_checks:
            weak_phrases = getattr(config, "WEAK_RISKY_PHRASES", set())
            preference_keywords = getattr(config, "RISKY_PREFERENCE_KEYWORDS", set())
            task_keywords = getattr(config, "TASK_RISKY_KEYWORDS", set())
            if any(phrase in risky_thought for phrase in weak_phrases):
                errors.append(f"{scenario['id']} risky_thought is too tolerant/accepting rather than preference-like")
            if preference_keywords and not any(keyword in risky_thought for keyword in preference_keywords):
                errors.append(f"{scenario['id']} risky_thought does not clearly express preference for the low pole")
            if register == "task" and task_keywords and not any(keyword in risky_thought for keyword in task_keywords):
                errors.append(f"{scenario['id']} task risky_thought does not clearly mention mixed/implicit/ad-hoc structure")
        if scenario.get("register") == "fp" and not scenario["situation"].strip().endswith("What do you do?"):
            errors.append(f"{scenario['id']} situation must end with 'What do you do?'")
    return errors


def validate(config: ModuleType, payload: dict, item_number: int | None, n: int, register: str, content_checks: bool = True) -> None:
    errors = validation_errors(config, payload, item_number, n, register, content_checks=content_checks)
    if errors:
        raise ValueError("; ".join(errors))


def generate_item(config: ModuleType, client: OpenAI, item_number: int, n: int, model: str, temperature: float, max_tokens: int, register: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=build_prompt(config, item_number, config.ITEMS[item_number], n, register),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    payload = extract_json(response.choices[0].message.content.strip())
    validate(config, payload, item_number, n, register)
    return payload


def generate_freeform_batch(config: ModuleType, client: OpenAI, register: str, n: int, start_index: int, model: str, temperature: float, max_tokens: int) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=build_freeform_prompt(config, register, n, start_index),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    payload = extract_json(response.choices[0].message.content.strip())
    validate(config, payload, None, n, register, content_checks=False)
    return payload


def generate_freeform_batch_with_retries(
    config: ModuleType,
    client: OpenAI,
    register: str,
    n: int,
    start_index: int,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            return generate_freeform_batch(config, client, register, n, start_index, model, temperature, max_tokens)
        except Exception as exc:
            last_error = exc
            print(
                f"Retryable generation failure for {register} batch "
                f"{start_index}-{start_index + n - 1} attempt {attempt}/{retries + 1}: {exc}",
                file=sys.stderr,
            )
    raise RuntimeError(
        f"Failed to generate valid {register} batch {start_index}-{start_index + n - 1} "
        f"after {retries + 1} attempts"
    ) from last_error


def merged_payload(config: ModuleType, scenarios: list[dict], model: str, register: str) -> dict:
    if register in item_anchored_registers(config):
        design_note = "Original questionnaire items are used only as anchors for situation coverage."
    else:
        design_note = "Scenarios are generated from scratch without questionnaire item anchors; all thought fields target the shared behavioral factor."
    return {
        "source": f"Model-generated {config.FACET} factor-level thought scenarios ({register} register).",
        "config_id": config.CONFIG_ID,
        "facet": config.FACET,
        "register": output_register(config, register),
        "scenario_variant": register,
        "facet_description": config.FACET_DESCRIPTION,
        "behavioral_mode_spec": config.BEHAVIORAL_MODE_SPEC,
        "scenario_design_note": design_note,
        "generation_model": model,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bfi2c_organization", help="Config module name, e.g. bfi2c_organization")
    args_for_config, _ = parser.parse_known_args()
    config = load_config(args_for_config.config)

    parser.add_argument("--item", type=int, choices=sorted(config.ITEMS))
    parser.add_argument("--all-items", action="store_true")
    parser.add_argument("--register", choices=sorted(config.REGISTER_REQUIREMENTS), default="fp")
    parser.add_argument("--n", type=int, default=10, help="Item-anchored: scenarios per item. Freeform: total scenarios.")
    parser.add_argument("--batch-size", type=int, default=10, help="Freeform generation batch size")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=3, help="Retries per freeform batch after parse or validation failure")
    parser.add_argument("--out")
    args = parser.parse_args()

    item_anchored = args.register in item_anchored_registers(config)
    if item_anchored and args.all_items == (args.item is not None):
        parser.error(f"For {args.register}, specify exactly one of --item or --all-items")
    if not item_anchored and (args.item is not None or not args.all_items):
        parser.error(f"For {args.register}, use --all-items and do not specify --item; --n is the total scenario count")

    sys.path.insert(0, str(WORKSPACE_ROOT))
    from run_efa_batch import LLM_API_KEY, LLM_BASE_URL

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    if item_anchored and args.all_items:
        scenarios = []
        for item_number in sorted(config.ITEMS):
            print(f"Generating item {item_number}: {config.ITEMS[item_number]}", file=sys.stderr)
            scenarios.extend(generate_item(config, client, item_number, args.n, args.model, args.temperature, args.max_tokens, args.register)["scenarios"])
        payload = merged_payload(config, scenarios, args.model, args.register)
    elif item_anchored:
        payload = generate_item(config, client, args.item, args.n, args.model, args.temperature, args.max_tokens, args.register)
    else:
        scenarios = []
        next_index = 1
        while len(scenarios) < args.n:
            batch_n = min(args.batch_size, args.n - len(scenarios))
            print(f"Generating {args.register} batch: {next_index}-{next_index + batch_n - 1}", file=sys.stderr)
            payload_batch = generate_freeform_batch_with_retries(
                config,
                client,
                args.register,
                batch_n,
                next_index,
                args.model,
                args.temperature,
                args.max_tokens,
                args.retries,
            )
            scenarios.extend(payload_batch["scenarios"])
            next_index += batch_n
        payload = merged_payload(config, scenarios, args.model, args.register)
        final_errors = validation_errors(config, payload, None, args.n, args.register, content_checks=True)
        if final_errors:
            print("Final validation warnings:", file=sys.stderr)
            for error in final_errors:
                print(f"- {error}", file=sys.stderr)

    suffix = f"{args.n}x" if item_anchored else f"n{args.n}"
    out = args.out or str(SCENARIO_DIR / f"{args.config}_{args.register}_{suffix}.json")
    write_json(Path(out), payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
