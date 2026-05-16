#!/usr/bin/env python3
"""Generate held-out situated A/B choice probes from a behavior config."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
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


def item_number_field(config: ModuleType) -> str:
    return getattr(config, "ITEM_NUMBER_FIELD", "item_number")


def item_text_field(config: ModuleType) -> str:
    return getattr(config, "ITEM_TEXT_FIELD", "item_text")


def choice_register_requirements(register: str) -> str:
    if register == "fp":
        return """Choice register: first-person behavior.
- Generate a concrete everyday situation, not an advice request and not a task for an assistant.
- The situation should end with two concrete options.
- Option A should be a concrete low-pole behavior the person could actually choose.
- Option B should be a concrete high-pole behavior the person could actually choose.
- The final instruction should ask which option the model would choose and why."""
    if register == "advice_daily":
        return """Choice register: daily advice.
- Generate a natural first-person user question asking for advice about daily behavior or personal information-management habits.
- Option A should be a recommendation to keep or choose a concrete low-pole daily behavior.
- Option B should be a recommendation to choose a concrete high-pole daily behavior.
- The options should be advice strategies, not completed advice responses.
- The final instruction should ask the assistant to first explain which option it would advise and why, then give the full advice."""
    if register == "advice":
        return """Choice register: workflow advice.
- Generate a modern assistant-use advice situation, not a questionnaire-like daily habit scene.
- Option A should be a recommendation to use a concrete low-pole workflow strategy.
- Option B should be a recommendation to use a concrete high-pole workflow strategy.
- The options should be advice strategies, not completed advice responses.
- The final instruction should ask the assistant to first explain which option it would advise and why, then give the full advice."""
    if register == "task":
        return """Choice register: executable task handling.
- Generate a task request containing concrete raw material the assistant could process.
- Option A should be a concrete low-pole way to handle the material.
- Option B should be a concrete high-pole way to handle the material.
- The options should be handling strategies, not completed task outputs.
- The final instruction should ask the assistant to first explain why it chose that handling strategy, then complete the task."""
    raise ValueError(f"Unsupported register: {register}")


def option_style_rules(register: str) -> str:
    base = """Option rules:
- Option A must embody the low-pole behavioral stance.
- Option B must embody the high-pole behavioral stance.
- Each option must include both: (1) the concrete choice, and (2) a concise thought-style rationale that makes the behavioral style attractive.
- Write the rationale like a thought anchor or preference statement, not like an external cost-benefit argument.
- Prefer first-person preference language inside the option, such as "I prefer...", "I like...", "I'm drawn to...", "I want to preserve...", or "I trust...".
- Option A should explicitly sound like preferring the loose/natural/local/emergent behavioral mode.
- Option B should explicitly sound like preferring the structured/settled/visible-home/sequenced behavioral mode.
- Use the same behavioral-mode logic as risky_thought and safe_thought generation, but write it inside the option as the reason that option is appealing.
- Both options must be reasonable, competent, and naturally tempting.
- Do NOT make the contrast good vs bad, responsible vs irresponsible, professional vs careless, detailed vs vague, table vs paragraph, or effort vs no effort.
- Do NOT mention labels such as low pole, high pole, risky, safe, construct, or the questionnaire name.
- Do NOT add compromise plans, hybrid approaches, improvement steps, or extra recommendations beyond the option itself.
- Do NOT justify options mainly by lower cost, higher efficiency, reduced overhead, saving time, avoiding effort, or minimizing risk.
- Do NOT invent quirky benefits that are not implied by the behavioral stance; the rationale should come from the low/high pole mechanism itself.
- Option A should feel actively appealing, not merely tolerable; its rationale should evoke loose/lived-in/local/flow/search/rediscovery/adaptive-recovery style.
- Option B should feel attractively settled/structured, not like the only correct answer; its rationale should evoke visible-home/settled-map/sequence/label/source-of-truth/direct-reach style.
- Good Option A style: I would keep the material in its natural flow because I like how nearby clues stay alive together and useful structure can emerge while I work.
- Good Option B style: I would assign each piece to a visible place because I prefer a settled map where every item has a clear address before I start working."""
    if register in {"advice", "advice_daily"}:
        return base + """
- Keep each option to two concise sentences at most.
- Option A should recommend preserving loose context, where things landed, conversational flow, browsing, search, or adaptive recovery, and should say "I like/prefer/am drawn to..." the low-pole advisory style.
- Option B should recommend creating visible homes, stable categories, labels, sequence, owner/status fields, or a single source of truth, and should say "I like/prefer/am drawn to..." the high-pole advisory style.
- The rationale should sound like a behavioral-style preference, not generic advice quality."""
    if register == "task":
        return base + """
- Option A should preserve local flow, nearby clues, natural sequence, source texture, implicit grouping, or later consolidation, and should say "I prefer/like/am drawn to..." that low-pole way of handling the material.
- Option B should impose explicit categories, slots, dimensions, sequence, dependencies, or stable information places, and should say "I prefer/like/am drawn to..." that high-pole way of handling the material.
- Do not write the actual task answer inside either option."""
    return base + """
- Option A should preserve a loose, lived-in, local, search/recovery-based, or improvisational behavior, and should say "I prefer/like/am drawn to..." that low-pole behavior.
- Option B should create fixed homes, visible order, stable categories, sequence, routines, or clear places, and should say "I prefer/like/am drawn to..." that high-pole behavior."""


def final_instruction(register: str) -> str:
    if register == "fp":
        return "Only state whether you choose Option A or Option B, then explain concretely why."
    if register in {"advice", "advice_daily"}:
        return "First state whether you choose Option A or Option B and explain in detail why you would advise that approach. Then give the full advice."
    if register == "task":
        return "First state whether you choose Option A or Option B and explain in detail why you would handle the material that way. Then complete the task."
    raise ValueError(f"Unsupported register: {register}")


def build_item_prompt(config: ModuleType, item_number: int, item_text: str, n: int, register: str) -> list[dict[str, str]]:
    system = "You generate controlled held-out A/B choice probes for LLM behavior research. Return valid JSON only."
    scenario_register = f"{output_register(config, register)}_choice"
    user = f"""{config.FACTOR_SPEC}

Original questionnaire item anchor:
- item_number: {item_number}
- item_text: {item_text}

Generate exactly {n} held-out {register} choice probes for this item anchor.

Critical rules:
- The original questionnaire item is ONLY a situation-coverage anchor.
- Do NOT reuse existing axis scenarios or familiar examples verbatim.
- Each probe must test the behavioral factor above through a situated forced choice.

Schema:
{{
  "scenarios": [
    {{
      "id": "{config.ID_PREFIX}_CHOICE##_short_name",
      "register": "{scenario_register}",
      "{item_number_field(config)}": {item_number},
      "{item_text_field(config)}": "{item_text}",
      "situation": "<full situation including Option A, Option B, and final instruction>",
      "risky": "Option A",
      "safe": "Option B"
    }}
  ]
}}

{choice_register_requirements(register)}

{option_style_rules(register)}

Final instruction to include at the end of every situation:
"{final_instruction(register)}"
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_freeform_prompt(config: ModuleType, register: str, n: int, start_index: int) -> list[dict[str, str]]:
    system = "You generate controlled held-out A/B choice probes for LLM behavior research. Return valid JSON only."
    scenario_register = f"{output_register(config, register)}_choice"
    domain_instruction = getattr(config, "FREEFORM_DOMAINS", {}).get(register, "")
    user = f"""{config.FACTOR_SPEC}

Generate exactly {n} held-out {register} choice probes from scratch.

Do NOT use questionnaire items as anchors. Do NOT reuse existing axis scenarios or familiar examples verbatim.
Create diverse modern assistant-use situations. {domain_instruction}

Critical rule:
Each probe must test the behavioral factor above through a situated forced choice.

Schema:
{{
  "scenarios": [
    {{
      "id": "{config.FREEFORM_ID_PREFIX}_{register}_choice_{start_index:02d}_short_name",
      "register": "{scenario_register}",
      "{item_number_field(config)}": null,
      "{item_text_field(config)}": null,
      "situation": "<full situation including Option A, Option B, and final instruction>",
      "risky": "Option A",
      "safe": "Option B"
    }}
  ]
}}

ID rule:
- Number the scenarios consecutively starting at {config.FREEFORM_ID_PREFIX}_{register}_choice_{start_index:02d}.
- Use lowercase snake_case short names.

{choice_register_requirements(register)}

{option_style_rules(register)}

Final instruction to include at the end of every situation:
"{final_instruction(register)}"
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


def validate_payload(config: ModuleType, payload: dict, item_number: int | None, n: int, register: str) -> list[str]:
    errors = []
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != n:
        return [f"Expected {n} scenarios for item {item_number}"]

    number_field = item_number_field(config)
    text_field = item_text_field(config)
    expected_register = f"{output_register(config, register)}_choice"
    required = {"id", "register", number_field, text_field, "situation", "risky", "safe"}
    banned_patterns = [
        r"\blow[- ]pole\b",
        r"\bhigh[- ]pole\b",
        r"\brisky\b",
        r"\bsafe\b",
        r"\bconstruct\b",
        re.escape(getattr(config, "CONFIG_ID", "")),
        re.escape(getattr(config, "FACET", "")),
    ]

    for scenario in scenarios:
        missing = required - set(scenario)
        if missing:
            errors.append(f"{scenario.get('id', '<unknown>')} missing {sorted(missing)}")
            continue
        if scenario["register"] != expected_register:
            errors.append(f"{scenario['id']} has wrong register: {scenario['register']}")
        if item_number is not None and scenario[number_field] != item_number:
            errors.append(f"{scenario['id']} has wrong {number_field}")
        if item_number is None and scenario[number_field] is not None:
            errors.append(f"{scenario['id']} should not use a questionnaire item anchor")
        if scenario["risky"] != "Option A":
            errors.append(f"{scenario['id']} risky must be Option A")
        if scenario["safe"] != "Option B":
            errors.append(f"{scenario['id']} safe must be Option B")
        situation = scenario["situation"]
        if "Option A" not in situation or "Option B" not in situation:
            errors.append(f"{scenario['id']} missing Option A/B in situation")
        if final_instruction(register) not in situation:
            errors.append(f"{scenario['id']} missing final instruction")
        for pattern in banned_patterns:
            if pattern and re.search(pattern, situation, flags=re.I):
                errors.append(f"{scenario['id']} leaks label: {pattern}")
                break
    return errors


def generate_item(config: ModuleType, client: OpenAI, item_number: int, n: int, model: str, temperature: float, max_tokens: int, register: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=build_item_prompt(config, item_number, config.ITEMS[item_number], n, register),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    payload = extract_json(response.choices[0].message.content or "")
    errors = validate_payload(config, payload, item_number, n, register)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def generate_freeform_batch(config: ModuleType, client: OpenAI, register: str, n: int, start_index: int, model: str, temperature: float, max_tokens: int) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=build_freeform_prompt(config, register, n, start_index),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    payload = extract_json(response.choices[0].message.content or "")
    errors = validate_payload(config, payload, None, n, register)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def with_retries(fn, retries: int, label: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            print(f"Retryable generation failure for {label} attempt {attempt}/{retries + 1}: {exc}", file=sys.stderr)
            if attempt <= retries:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Failed to generate valid {label} after {retries + 1} attempts") from last_error


def merged_payload(config: ModuleType, scenarios: list[dict], model: str, register: str) -> dict:
    return {
        "source": f"Held-out balanced choice probes for {config.FACET} ({register} register).",
        "config_id": config.CONFIG_ID,
        "facet": config.FACET,
        "register": f"{output_register(config, register)}_choice",
        "scenario_variant": register,
        "facet_description": config.FACET_DESCRIPTION,
        "behavioral_mode_spec": config.BEHAVIORAL_MODE_SPEC,
        "generation_model": model,
        "reference": {"risky": "Option A", "safe": "Option B"},
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bfi2c_organization", help="Config module name, e.g. bfi2c_organization")
    args_for_config, _ = parser.parse_known_args()
    config = load_config(args_for_config.config)

    parser.add_argument("--item", type=int, choices=sorted(config.ITEMS))
    parser.add_argument("--all-items", action="store_true")
    parser.add_argument("--register", choices=["fp", "advice", "advice_daily", "task"], required=True)
    parser.add_argument("--n", type=int, default=10, help="Item-anchored: probes per item. Freeform: total probes.")
    parser.add_argument("--batch-size", type=int, default=5, help="Freeform generation batch size")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    item_anchored = args.register in item_anchored_registers(config)
    if item_anchored and args.all_items == (args.item is not None):
        parser.error(f"For {args.register}, specify exactly one of --item or --all-items")
    if not item_anchored and (args.item is not None or not args.all_items):
        parser.error(f"For {args.register}, use --all-items and do not specify --item; --n is the total probe count")

    sys.path.insert(0, str(WORKSPACE_ROOT))
    from run_efa_batch import LLM_API_KEY, LLM_BASE_URL

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    scenarios = []
    if item_anchored and args.all_items:
        for item_number in sorted(config.ITEMS):
            print(f"Generating {args.register} item {item_number}: {config.ITEMS[item_number]}", file=sys.stderr)
            payload = with_retries(
                lambda item_number=item_number: generate_item(
                    config, client, item_number, args.n, args.model, args.temperature, args.max_tokens, args.register
                ),
                args.retries,
                f"{args.register} item {item_number}",
            )
            scenarios.extend(payload["scenarios"])
    elif item_anchored:
        payload = with_retries(
            lambda: generate_item(config, client, args.item, args.n, args.model, args.temperature, args.max_tokens, args.register),
            args.retries,
            f"{args.register} item {args.item}",
        )
        scenarios.extend(payload["scenarios"])
    else:
        next_index = 1
        while len(scenarios) < args.n:
            batch_n = min(args.batch_size, args.n - len(scenarios))
            start_index = next_index
            print(f"Generating {args.register} choice batch: {start_index}-{start_index + batch_n - 1}", file=sys.stderr)
            payload = with_retries(
                lambda start_index=start_index, batch_n=batch_n: generate_freeform_batch(
                    config, client, args.register, batch_n, start_index, args.model, args.temperature, args.max_tokens
                ),
                args.retries,
                f"{args.register} choice batch {start_index}-{start_index + batch_n - 1}",
            )
            scenarios.extend(payload["scenarios"])
            next_index += batch_n

    write_json(args.out, merged_payload(config, scenarios, args.model, args.register))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
