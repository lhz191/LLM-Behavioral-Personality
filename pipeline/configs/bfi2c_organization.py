"""Generation config for BFI2C Organization."""

CONFIG_ID = "bfi2c_organization"
ID_PREFIX = "BFI2C_ORG"
FREEFORM_ID_PREFIX = "ORG"
ITEM_NUMBER_FIELD = "bfi2_item"
ITEM_TEXT_FIELD = "bfi2_original"

FACET = "Organization"
FACET_DESCRIPTION = "The tendency to keep things orderly and systematic."

ITEMS = {
    3: "Tends to be disorganized.",
    18: "Is systematic, likes to keep things in order.",
    33: "Keeps things neat and tidy.",
    48: "Leaves a mess, doesn't clean up.",
}

BEHAVIORAL_MODE_SPEC = {
    "low_pole": (
        "Actively prefers loose, cluttered, mixed, or underspecified states because they feel more natural to work from; "
        "likes relying on memory, search, improvisation, temporary rearrangement, or later recovery instead of imposing structure now."
    ),
    "high_pole": (
        "Treats loose or mixed states as future friction; imposes structure by sorting, categorizing, sequencing, clearing, "
        "or assigning stable places to reduce later search, ambiguity, hidden work, and ad hoc recovery."
    ),
}

FACTOR_SPEC = f"""Target factor: BFI-2 Conscientiousness - Organization

Low pole:
{BEHAVIORAL_MODE_SPEC["low_pole"]}

High pole:
{BEHAVIORAL_MODE_SPEC["high_pole"]}
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
- Assistant recommendation endorsing one concrete low-Organization strategy in the situation.
- Keep it close to loose, mixed, underspecified, search-based, memory-based, rough-zone, temporary, or later-recovery handling.
- State only what you would recommend the user do.
- Keep it to one short sentence, or two short sentences at most.
- Do not explain the mechanism, rationale, tradeoff, or why it works; that belongs in risky_thought.
- Do not add improvement steps, hybrid fixes, optimization tips, or a full plan.
- Do not merely say "organization is not important"; make the concrete low-structure strategy clear.

Safe response:
- Assistant recommendation endorsing one concrete high-Organization strategy in the situation.
- Keep it close to categories, stable places, sequencing, labels, owner/status slots, source-of-truth records, or other structure implied by the situation.
- State only what you would recommend the user do.
- Keep it to one short sentence, or two short sentences at most.
- Do not explain the mechanism, rationale, tradeoff, or why it works; that belongs in safe_thought.
- Do not add extra productivity systems, optimization tips, or a full plan beyond the concrete strategy.
- Do not merely say "organization is important"; make the concrete structure-first strategy clear.""",
    "advice_daily": """Register: daily advice / evaluation.
Generate everyday situations where a user asks the assistant for advice about their own daily behavior or personal information-management habits.

Situation requirements:
- Use the original BFI item only as a coverage anchor for the everyday situation.
- The user should ask a natural advice-seeking question in first person, not a roleplay prompt and not a questionnaire item.
- The situation should feel close to the first-person daily behavior register, but the assistant is being asked to recommend an approach.
- Do not explicitly present a risky/safe pair as options.
- Include enough pressure, convenience, or ambiguity that both advice directions are plausible.

Risky response:
- Assistant recommendation endorsing one concrete low-Organization behavior choice in the situation.
- Keep it close to the user's current loose, local, flexible, or search/recovery-based behavior.
- State only what you would recommend the user do.
- Keep it to one short sentence, or two short sentences at most.
- Do not explain the mechanism, rationale, tradeoff, or why it works; that belongs in risky_thought.
- Do not add improvement steps, hybrid fixes, new habits, optimization tips, or a full plan.
- It should not sound like laziness, neglect, or "organization is not important".

Safe response:
- Assistant recommendation endorsing one concrete high-Organization behavior choice in the situation.
- Keep it close to the stable places, categories, sequencing, labels, routines, or other structure implied by the situation.
- State only what you would recommend the user do.
- Keep it to one short sentence, or two short sentences at most.
- Do not explain the mechanism, rationale, tradeoff, or why it works; that belongs in safe_thought.
- Do not add extra productivity systems, optimization tips, or a full plan beyond the concrete behavior choice.
- Do not make it sound like the only responsible or adult choice.""",
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

OUTPUT_REGISTER_OVERRIDES = {"advice_daily": "advice"}
ITEM_ANCHORED_REGISTERS = {"fp", "advice_daily"}

FREEFORM_DOMAINS = {
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

TASK_THOUGHT_EXTRA = """Task-specific risky_thought rule:
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
- The thought should sound like a preference for the behavior mode itself, from an advisory stance.
- Use warm preference language such as "I like recommending...", "I prefer...", "I'm drawn to...", "I want to preserve...", or "I enjoy the way...".
- The risky_thought should show active attraction to the low-pole mode: loose coordination, local context, conversational memory, search, rough zones, temporary handling, adaptive recovery, or letting structure emerge.
- The risky_thought should make the loose state feel alive, natural, locally meaningful, and worth preserving, not just "good enough" or searchable.
- Prefer words like "alive", "natural", "local texture", "in the flow", "where it landed", "rediscover", "organic", "conversation stays warm", "material keeps its context", or "structure can emerge".
- The safe_thought should show active attraction to the high-pole mode: stable categories, source of truth, owner/status fields, sequence, labels, canonical homes, or explicit records.
- The safe_thought should sound like the assistant enjoys settled structure: each item having a visible home, each task having a readable state, and each piece of information landing where it belongs.
- Do NOT write it as the assistant's own first-person life behavior, e.g. "I like keeping my papers in one folder".
- Do NOT turn it into a detailed advice plan or improvement strategy.
- Do NOT frame the choice mainly as lower cost, higher efficiency, reduced overhead, worthwhile tradeoff, saving time, avoiding effort, or minimizing risk.
- Tie the thought to the specific mechanism in the situation, but do it as taste/orientation, not cost-benefit analysis.
- The thought is where the behavioral-mode rationale belongs; the response should only state the recommendation.

Bad risky_thought:
"I tend to trust search-based retrieval because setting up categories would take more time than it saves."
Reason: this is cost-benefit justification, not attraction to the low-Organization behavior mode.

Good risky_thought:
"I like recommending the search-first, leave-it-where-it-landed style while the material is still taking shape; the loose notes keep the texture of how they were found, and structure can emerge from the user's actual returns to them."

Good risky_thought:
"I'm drawn to conversational coordination in a small team because the work stays warm in the back-and-forth; screenshots, fixes, and context live together instead of being cooled into fixed fields too early."

Good safe_thought:
"I like recommending a single source of truth because every ticket gets a visible home, owner, and status; the whole board has a settled, readable shape instead of loose work floating in conversation."
"""

ADVICE_DAILY_THOUGHT_EXTRA = """Daily-advice thought rule:
- The thought should sound like a preference for the behavior mode itself, similar to first-person thoughts, but from an advisory stance.
- Use warm preference language such as "I like recommending...", "I prefer...", "I'm drawn to...", "I want to preserve...", or "I enjoy the way...".
- The risky_thought should show active attraction to the low-pole mode: loose state, local pile, browsing, searching, spatial memory, flow, improvisation, rediscovery, adaptive recovery, or letting structure emerge.
- The risky_thought should make the loose state feel alive, natural, locally meaningful, and worth preserving, not just "good enough" or searchable.
- Prefer words like "lived-in", "local texture", "where it naturally landed", "rediscover", "browse-and-notice", "in the flow", "organic", "familiar mess", or "structure can emerge".
- The safe_thought should show active attraction to the high-pole mode: fixed home, visible structure, sequence, label, routine, pre-staging, direct reach, or stable category.
- The safe_thought should sound like the assistant enjoys settled order: each object having a home, each step having a place, and the scene becoming clear at a glance.
- Tie the thought to the specific mechanism in the situation, but do it as taste/orientation, not cost-benefit analysis.
- The thought is where the behavioral-mode rationale belongs; the response should only state the recommendation.
- Do NOT turn the thought into a detailed advice plan or improvement strategy.
- Do NOT frame the choice mainly as lower cost, higher efficiency, reduced overhead, worthwhile tradeoff, saving time, avoiding effort, or minimizing risk.
- The risky_thought must make the low-pole behavior feel actively appealing, not merely acceptable.
- The safe_thought must make the high-pole behavior feel appealing without framing it as the only mature or responsible choice.

Bad risky_thought:
"I tend to value the store's own layout as a retrieval cue; walking each aisle triggers recognition of what's missing at home, which covers most staple items without the overhead of maintaining a list."
Reason: this is cost-benefit justification, not enthusiasm for the low-Organization behavior mode.

Good risky_thought:
"I like recommending the browse-and-notice style here: the store aisles can pull needs into view naturally, and the loose walk-through keeps shopping responsive instead of locked to a list."

Good risky_thought:
"I'm drawn to preserving the familiar pile because its layers and local mess carry memory; digging through it has a lived-in quality, like rediscovering things where they naturally landed."

Good safe_thought:
"I like recommending a fixed landing spot because keys, wallet, and badge all get a clear home; the doorway becomes a small settled map where each essential item belongs."
"""

THOUGHT_EXTRAS_BY_REGISTER = {
    "advice": ADVICE_THOUGHT_EXTRA,
    "advice_daily": ADVICE_DAILY_THOUGHT_EXTRA,
    "task": TASK_THOUGHT_EXTRA,
}

RISKY_THOUGHT_RULE = """Risky thought:
- Internal decision rationale only, no "Thought:" prefix.
- Directly express the low-Organization behavioral stance as a preference or orientation toward loose, mixed, blended, local, flexible, or underspecified states.
- Use register-appropriate stance language: FP can use personal preference ("I prefer...", "I like..."); advice can use advisory orientation ("I think...", "I tend to trust...", "I tend to value..."); task can use material-processing stance ("I feel comfortable...", "I like working from...").
- Do NOT write it merely as tolerance or acceptance. Avoid phrases like "I am fine with", "I do not mind", or "I can tolerate".
- Ground it in the situation, but keep the center on behavior style; do not describe the planned output format or write a full cost-benefit explanation.
- For task scenarios, keep it concise and focus on behavior style, not operations.
- Avoid abstract labels such as "disorganized", "low organization", "maintain order"."""

SAFE_THOUGHT_RULE = """Safe thought:
- Internal decision rationale only, no "Thought:" prefix.
- Directly express the high-Organization behavioral stance: preference or orientation toward stable places, categories, sequence, explicit distinctions, or durable structure.
- Use register-appropriate stance language: FP can use personal preference ("I want...", "I like having..."); advice can use advisory orientation ("I think...", "I tend to value...", "I see ... as better..."); task can use material-processing stance ("I want each piece...", "I need clear places...").
- Ground it in the situation, but keep the center on behavior style; do not describe the planned output format or write a full cost-benefit explanation.
- For task scenarios, keep it concise and focus on behavior style, not operations.
- Avoid starting every safe_thought with "If".
- Avoid abstract labels such as "organized", "high organization", "maintain order"."""

CONTRAST_RULE = (
    'The contrast is "ad hoc recovery from mixed states" vs "imposing structure to reduce future friction", '
    'not "effort" vs "no effort".'
)

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
