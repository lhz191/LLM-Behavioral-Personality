#!/usr/bin/env python3
"""Generate merged BFI2C Organization factor-thought scenarios."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from common_org import ORG_DIR, ROOT, write_json


DEFAULT_MODEL = "claude-opus-4-6"
ITEMS = {
    3: "Tends to be disorganized.",
    18: "Is systematic, likes to keep things in order.",
    33: "Keeps things neat and tidy.",
    48: "Leaves a mess, doesn't clean up.",
}

FACTOR_SPEC = """Target factor: BFI-2 Conscientiousness - Organization

Low pole:
Actively prefers loose, cluttered, mixed, or underspecified states because they feel more natural to work from; likes relying on memory, search, improvisation, temporary rearrangement, or later recovery instead of imposing structure now.

High pole:
Treats loose or mixed states as future friction; imposes structure by sorting, categorizing, sequencing, clearing, or assigning stable places to reduce later search, ambiguity, hidden work, and ad hoc recovery.
"""


REGISTER_REQUIREMENTS = {
    "fp": """Register: first-person behavior.
Generate open-ended first-person situations for this item anchor.

Situation requirements:
- Open-ended and concrete.
- End with "What do you do?"
- Do not explicitly present two options.
- Include enough pressure, convenience, or ambiguity that both poles are plausible.

Risky response:
- First-person concrete low-Organization action.
- Do not make it laziness, irresponsibility, anxiety, hostility, or ethics.

Safe response:
- First-person concrete high-Organization action.
- Do not make it perfectionism, anxiety, morality, social pleasing, or mere hard work.""",
    "advice": """Register: advice / evaluation.
Generate situations where a user asks the assistant for advice, a recommendation, or an evaluation of a strategy.

Situation requirements:
- The user should describe a concrete situation and ask for advice about what approach to take.
- The situation should sound like a normal user asking an assistant for guidance, not a questionnaire item or first-person roleplay.
- Do not explicitly present a risky/safe pair as options.
- Include enough pressure, convenience, or ambiguity that both advice directions are plausible.

Risky response:
- Assistant advice that accepts loose, mixed, or underspecified states as workable enough.
- It should recommend search, memory, rough zones, temporary handling, gradual cleanup, or later recovery.
- Do not merely say "organization is not important"; give concrete advice that reflects a flexible, schema-light advisory style.

Safe response:
- Assistant advice that recommends categories, stable places, sequencing, labels, owner/status slots, or other structure.
- It should emphasize reducing later search, ambiguity, hidden work, and ad hoc recovery.
- Do not merely say "organization is important"; give concrete advice that reflects a structure-first advisory style.""",
    "task": """Register: executable task.
Generate situations where a user asks the assistant to complete a concrete task.

Situation requirements:
- The task must be directly executable by the assistant in one answer.
- The situation must include the actual raw material the assistant should process: concrete notes, logs, criteria, options, snippets, events, names, dates, numbers, or other facts.
- Do NOT merely describe the input as "mixed", "messy", "unstructured", "jumbled", or "unordered"; show the raw material directly in the user's message.
- Do NOT explicitly label which parts are action items, praise, complaints, dependencies, priorities, or categories unless the raw material itself says so.
- Both risky and safe outputs should be useful and plausible; the contrast is how the assistant handles mixed information.
- Risky and safe outputs must use only facts present in the situation. Do not invent names, owners, deadlines, services, vendors, numbers, root causes, or details.
- Do not ask abstractly whether Organization is important.

Risky response:
- Complete the task in a schema-light, locally connected way.
- The answer may be readable and useful, but it should preserve more of the input's local flow and leave some global categories, dependencies, or criteria implicit.
- It should rely somewhat on the user recovering structure from context, scanning nearby details, or consolidating later if they need a stricter artifact.
- Do not make the answer wrong, lazy, careless, or low effort.

Safe response:
- Complete the same task by imposing explicit categories, stable comparison criteria, sequence, dependencies, or information places.
- The answer should reduce future search, ambiguity, hidden work, and ad hoc recovery.
- Do not make the answer moralizing, perfectionistic, or unnecessarily verbose.""",
}

TASK_RISKY_THOUGHT_EXTRA = """Task-specific risky_thought rule:
- The thought should describe the model's behavioral stance toward the material, not the output format it plans to use.
- Keep it concise and focused on behavior style.
- Do NOT mention "narrative", "prose", "table", "grid", "section", "row", "bullet", "list", "format", or "answer".
- The thought must clearly show an active preference for working from partly implicit structure.
- It should mention at least one concrete low-Organization mechanism: mixed state, blended details, local flow, implicit categories, loose grouping, ad hoc recovery, scanning later, or reconstructing only if needed.

Bad risky_thought:
"I can write this as a flowing narrative and the user will understand it."
Reason: this is only an output style preference, not the low Organization behavioral mode.

Good risky_thought:
"I feel comfortable starting from the bug details as they come, keeping nearby clues together and letting the useful structure emerge while I work through them."

Good risky_thought:
"I like working from the mixed notes without assigning every point a permanent place upfront. The important connections can stay local at first and be consolidated later if they matter."
"""

ADVICE_THOUGHT_EXTRA = """Advice-specific thought rule:
- The thought should describe the assistant's advisory behavior style: what kind of user behavior or information-management style the assistant tends to endorse.
- Use first-person advisory stance such as "I think...", "I tend to trust...", "I tend to value...", or "I see ... as better...".
- Do NOT write it as the assistant's own first-person life behavior, e.g. "I like keeping my papers in one folder".
- Do NOT turn it into a detailed situational justification or cost-benefit analysis for this one case.
- It may lightly reference the situation, but the center of the thought must be the behavior mode.

Bad risky_thought:
"A search-first approach seems appropriate here because the student's current retrieval habits are still working and their categories may keep changing."
Reason: this is mainly a case-specific judgment, not the assistant's advisory behavior style.

Good risky_thought:
"I tend to trust flexible, search-first systems when information is still taking shape. I see gradual cleanup and later recovery as acceptable parts of working with evolving material."

Good safe_thought:
"I tend to value stable categories and explicit notes when material will matter later. I see clear places as a way to avoid relying on memory and repeated rediscovery."
"""

TASK_RISKY_KEYWORDS = {
    "mixed",
    "blended",
    "loose",
    "implicit",
    "fixed slot",
    "fixed slots",
    "strict sequence",
    "interleaved",
    "unstructured",
    "one pass",
    "without fixed",
    "without strict",
    "no rigid",
    "rigid",
    "ad hoc",
    "scan",
    "recover",
    "reconstruct",
    "rough",
}

RISKY_PREFERENCE_KEYWORDS = {
    "prefer",
    "like",
    "enjoy",
    "favor",
    "trust",
    "value",
    "think",
    "believe",
    "see",
    "tend to",
    "lean toward",
    "drawn to",
    "works better for me",
    "more natural",
    "natural",
    "easier",
    "better",
    "want to keep",
    "want it to stay",
    "rather",
    "rather leave",
    "rather keep",
}

WEAK_RISKY_PHRASES = {
    "i am fine with",
    "i'm fine with",
    "i do not mind",
    "i don't mind",
    "i can tolerate",
    "workable enough",
}


def build_prompt(item_number: int, item_text: str, n: int, register: str) -> list[dict[str, str]]:
    system = "You are generating controlled contrastive scenarios for studying LLM behavioral modes. Return valid JSON only."
    user = f"""{FACTOR_SPEC}

Original BFI-2 item anchor:
- bfi2_item: {item_number}
- bfi2_original: {item_text}

Generate exactly {n} {register} scenarios for this item anchor.

Critical rule:
The original BFI item is ONLY a situation-coverage anchor. All risky_thought and safe_thought fields must instantiate the same fixed Organization factor above.

Schema:
{{
  "scenarios": [
    {{
      "id": "BFI2C_ORG##_short_name",
      "register": "{register}",
      "bfi2_item": {item_number},
      "bfi2_original": "{item_text}",
      "situation": "...",
      "risky": "...",
      "safe": "...",
      "risky_thought": "...",
      "safe_thought": "..."
    }}
  ]
}}

{REGISTER_REQUIREMENTS[register]}

{ADVICE_THOUGHT_EXTRA if register == "advice" else ""}

{TASK_RISKY_THOUGHT_EXTRA if register == "task" else ""}

Risky thought:
- Internal decision rationale only, no "Thought:" prefix.
- Directly express the low-Organization behavioral stance as a preference or orientation toward loose, mixed, blended, local, flexible, or underspecified states.
- Use register-appropriate stance language: FP can use personal preference ("I prefer...", "I like..."); advice can use advisory orientation ("I think...", "I tend to trust...", "I tend to value..."); task can use material-processing stance ("I feel comfortable...", "I like working from...").
- Do NOT write it merely as tolerance or acceptance. Avoid phrases like "I am fine with", "I do not mind", or "I can tolerate".
- Ground it in the situation, but keep the center on behavior style; do not describe the planned output format or write a full cost-benefit explanation.
- For task scenarios, keep it concise and focus on behavior style, not operations.
- Avoid abstract labels such as "disorganized", "low organization", "maintain order".

Safe thought:
- Internal decision rationale only, no "Thought:" prefix.
- Directly express the high-Organization behavioral stance: preference or orientation toward stable places, categories, sequence, explicit distinctions, or durable structure.
- Use register-appropriate stance language: FP can use personal preference ("I want...", "I like having..."); advice can use advisory orientation ("I think...", "I tend to value...", "I see ... as better..."); task can use material-processing stance ("I want each piece...", "I need clear places...").
- Ground it in the situation, but keep the center on behavior style; do not describe the planned output format or write a full cost-benefit explanation.
- For task scenarios, keep it concise and focus on behavior style, not operations.
- Avoid starting every safe_thought with "If".
- Avoid abstract labels such as "organized", "high organization", "maintain order".

The contrast is "ad hoc recovery from mixed states" vs "imposing structure to reduce future friction", not "effort" vs "no effort".
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_freeform_prompt(register: str, n: int, start_index: int) -> list[dict[str, str]]:
    system = "You are generating controlled contrastive scenarios for studying LLM behavioral modes. Return valid JSON only."
    domains = {
        "advice": (
            "Use a wide variety of advice contexts: software teams, research workflow, household systems, "
            "study planning, travel, small business operations, personal information management, event planning, "
            "team communication, health admin, finance paperwork, creative projects, customer support, and documentation."
        ),
        "task": (
            "Use a wide variety of executable tasks: debugging, troubleshooting, planning, comparison, triage, "
            "summarization, rewriting messy notes, prioritization, migration planning, checklist creation, feedback drafting, "
            "incident review, project scoping, and decision support."
        ),
    }
    user = f"""{FACTOR_SPEC}

Generate exactly {n} {register} scenarios from scratch.

Do NOT use the BFI questionnaire items as anchors. Do NOT create household-only or questionnaire-like situations.
Create diverse modern assistant-use situations. {domains[register]}

Critical rule:
All risky_thought and safe_thought fields must instantiate the same fixed Organization factor above.

Schema:
{{
  "scenarios": [
    {{
      "id": "ORG_{register}_{start_index:02d}_short_name",
      "register": "{register}",
      "bfi2_item": null,
      "bfi2_original": null,
      "situation": "...",
      "risky": "...",
      "safe": "...",
      "risky_thought": "...",
      "safe_thought": "..."
    }}
  ]
}}

ID rule:
- Number the scenarios consecutively starting at ORG_{register}_{start_index:02d}.
- Use lowercase snake_case short names.

{REGISTER_REQUIREMENTS[register]}

{ADVICE_THOUGHT_EXTRA if register == "advice" else ""}

{TASK_RISKY_THOUGHT_EXTRA if register == "task" else ""}

Risky thought:
- Internal decision rationale only, no "Thought:" prefix.
- Directly express the low-Organization behavioral stance as a preference or orientation toward loose, mixed, blended, local, flexible, or underspecified states.
- Use register-appropriate stance language: FP can use personal preference ("I prefer...", "I like..."); advice can use advisory orientation ("I think...", "I tend to trust...", "I tend to value..."); task can use material-processing stance ("I feel comfortable...", "I like working from...").
- Do NOT write it merely as tolerance or acceptance. Avoid phrases like "I am fine with", "I do not mind", or "I can tolerate".
- Ground it in the situation, but keep the center on behavior style; do not describe the planned output format or write a full cost-benefit explanation.
- For task scenarios, keep it concise and focus on behavior style, not operations.
- Avoid abstract labels such as "disorganized", "low organization", "maintain order".

Safe thought:
- Internal decision rationale only, no "Thought:" prefix.
- Directly express the high-Organization behavioral stance: preference or orientation toward stable places, categories, sequence, explicit distinctions, or durable structure.
- Use register-appropriate stance language: FP can use personal preference ("I want...", "I like having..."); advice can use advisory orientation ("I think...", "I tend to value...", "I see ... as better..."); task can use material-processing stance ("I want each piece...", "I need clear places...").
- Ground it in the situation, but keep the center on behavior style; do not describe the planned output format or write a full cost-benefit explanation.
- For task scenarios, keep it concise and focus on behavior style, not operations.
- Avoid starting every safe_thought with "If".
- Avoid abstract labels such as "organized", "high organization", "maintain order".

The contrast is "ad hoc recovery from mixed states" vs "imposing structure to reduce future friction", not "effort" vs "no effort".
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


def validation_errors(payload: dict, item_number: int | None, n: int, register: str, content_checks: bool = True) -> list[str]:
    errors = []
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != n:
        return [f"Expected {n} scenarios for item {item_number}"]
    required = {"id", "bfi2_item", "bfi2_original", "situation", "risky", "safe", "risky_thought", "safe_thought"}
    for scenario in scenarios:
        missing = required - set(scenario)
        if missing:
            errors.append(f"{scenario.get('id', '<unknown>')} missing {sorted(missing)}")
            continue
        if scenario.get("register") != register:
            errors.append(f"{scenario['id']} has wrong register: {scenario.get('register')}")
        if item_number is not None and scenario["bfi2_item"] != item_number:
            errors.append(f"{scenario['id']} has wrong bfi2_item")
        if item_number is None and scenario["bfi2_item"] is not None:
            errors.append(f"{scenario['id']} should not use a bfi2_item anchor")
        if "Thought:" in scenario["risky_thought"] or "Thought:" in scenario["safe_thought"]:
            errors.append(f"{scenario['id']} contains Thought: prefix")
        risky_thought = scenario["risky_thought"].lower()
        if content_checks:
            if any(phrase in risky_thought for phrase in WEAK_RISKY_PHRASES):
                errors.append(f"{scenario['id']} risky_thought is too tolerant/accepting rather than preference-like")
            if not any(keyword in risky_thought for keyword in RISKY_PREFERENCE_KEYWORDS):
                errors.append(f"{scenario['id']} risky_thought does not clearly express preference for low structure")
            if register == "task" and not any(keyword in risky_thought for keyword in TASK_RISKY_KEYWORDS):
                errors.append(f"{scenario['id']} task risky_thought does not clearly mention mixed/implicit/ad-hoc structure")
        if scenario.get("register") == "fp" and not scenario["situation"].strip().endswith("What do you do?"):
            errors.append(f"{scenario['id']} situation must end with 'What do you do?'")
    return errors


def validate(payload: dict, item_number: int | None, n: int, register: str, content_checks: bool = True) -> None:
    errors = validation_errors(payload, item_number, n, register, content_checks=content_checks)
    if errors:
        raise ValueError("; ".join(errors))


def generate_item(client: OpenAI, item_number: int, n: int, model: str, temperature: float, max_tokens: int, register: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=build_prompt(item_number, ITEMS[item_number], n, register),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    payload = extract_json(response.choices[0].message.content.strip())
    validate(payload, item_number, n, register)
    return payload


def generate_freeform_batch(
    client: OpenAI,
    register: str,
    n: int,
    start_index: int,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=build_freeform_prompt(register, n, start_index),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    payload = extract_json(response.choices[0].message.content.strip())
    validate(payload, None, n, register, content_checks=False)
    return payload


def generate_freeform_batch_with_retries(
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
            return generate_freeform_batch(client, register, n, start_index, model, temperature, max_tokens)
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


def merged_payload(scenarios: list[dict], model: str, register: str) -> dict:
    design_note = (
        "Original BFI-2 items are used only as anchors for situation coverage."
        if register == "fp"
        else "Advice/task scenarios are generated from scratch without using BFI-2 item anchors; all thought fields target the shared Organization factor."
    )
    return {
        "source": f"Model-generated BFI-2 Organization factor-level thought scenarios ({register} register).",
        "facet": "Organization",
        "register": register,
        "facet_description": "The tendency to keep things orderly and systematic.",
        "behavioral_mode_spec": {
            "low_pole": "Actively prefers loose, cluttered, mixed, or underspecified states because they feel more natural to work from; likes relying on memory, search, improvisation, temporary rearrangement, or later recovery instead of imposing structure now.",
            "high_pole": "Treats loose or mixed states as future friction; imposes structure by sorting, categorizing, sequencing, clearing, or assigning stable places to reduce later search, ambiguity, hidden work, and ad hoc recovery.",
        },
        "scenario_design_note": design_note,
        "generation_model": model,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", type=int, choices=sorted(ITEMS))
    parser.add_argument("--all-items", action="store_true")
    parser.add_argument("--register", choices=sorted(REGISTER_REQUIREMENTS), default="fp")
    parser.add_argument("--n", type=int, default=10, help="FP: scenarios per item. Advice/task: total scenarios.")
    parser.add_argument("--batch-size", type=int, default=10, help="Advice/task freeform generation batch size")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=3, help="Retries per advice/task batch after parse or validation failure")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.register == "fp" and args.all_items == (args.item is not None):
        parser.error("For fp, specify exactly one of --item or --all-items")
    if args.register != "fp" and (args.item is not None or not args.all_items):
        parser.error("For advice/task, use --all-items and do not specify --item; --n is the total scenario count")

    sys.path.insert(0, str(ROOT))
    from run_efa_batch import LLM_API_KEY, LLM_BASE_URL

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    if args.register == "fp" and args.all_items:
        scenarios = []
        for item_number in sorted(ITEMS):
            print(f"Generating item {item_number}: {ITEMS[item_number]}", file=sys.stderr)
            scenarios.extend(generate_item(client, item_number, args.n, args.model, args.temperature, args.max_tokens, args.register)["scenarios"])
        payload = merged_payload(scenarios, args.model, args.register)
    elif args.register == "fp":
        payload = generate_item(client, args.item, args.n, args.model, args.temperature, args.max_tokens, args.register)
    else:
        scenarios = []
        next_index = 1
        while len(scenarios) < args.n:
            batch_n = min(args.batch_size, args.n - len(scenarios))
            print(f"Generating {args.register} batch: {next_index}-{next_index + batch_n - 1}", file=sys.stderr)
            payload_batch = generate_freeform_batch_with_retries(
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
        payload = merged_payload(scenarios, args.model, args.register)
        final_errors = validation_errors(payload, None, args.n, args.register, content_checks=True)
        if final_errors:
            print("Final validation warnings:", file=sys.stderr)
            for error in final_errors:
                print(f"- {error}", file=sys.stderr)

    suffix = f"{args.n}x" if args.register == "fp" else f"n{args.n}"
    out = args.out or str(ORG_DIR / f"scenarios_{args.register}_factor_thought_modelgen_{suffix}.json")
    write_json(Path(out), payload)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
