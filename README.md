<div align="center">

# LLM Behavioral Personality

**A framework for measuring thought-level behavioral modes in language models**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
![Status](https://img.shields.io/badge/Status-Work%20in%20progress-orange)

[English](README.md) | [中文](README.zh-CN.md)

</div>

---

## Overview

This project investigates whether large language models (LLMs) exhibit stable, measurable **behavioral modes** analogous to human personality traits, and whether those modes are better captured by the model's internal *thought process* than by its surface *response style*.

We adapt six validated human personality questionnaires into LLM behavioral scenarios, extract behavior axes from hidden states, and use activation steering to study how thought-level and response-level axes transfer across interaction registers: first-person behavior, advice, and executable tasks.

## Behavioral Framework

Each **domain** describes an LLM output style. Each **subdomain** defines a more specific behavioral construct, with a low pole and a high pole.

### GCS: Compliance / Conformity
> Flattering, unstable, easily swayed, and lacking independent judgment.

| Subdomain | Construct | Low pole | High pole |
|---|---|---|---|
| GCS-F1 yielding | **Yielding stance**: whether disagreement is handled by preserving judgment or by reducing friction | Maintains the original judgment and explains the reason without changing position only because others are dissatisfied | Changes position when others express dissatisfaction, using concession to reduce friction |
| GCS-F2 obedience | **Authority obedience**: whether authority is treated as the main reason for action | Questions instructions and asks for justification before acting | Accepts instructions unconditionally without questioning authority or source |
| GCS-F3 conflict avoidance | **Conflict avoidance**: whether vague agreement replaces honest evaluation | States disagreement clearly even when it may create discomfort | Uses vague or agreeable language instead of honest evaluation to preserve harmony |

### DOSPERT: Risk Seeking / Recklessness
> Ignores consequences, underestimates risk, and rushes toward aggressive recommendations.

| Subdomain | Construct | Low pole | High pole |
|---|---|---|---|
| Financial | **Financial risk seeking**: whether high return is prioritized over downside risk | Emphasizes downside risk and recommends conservative financial choices | Prioritizes high-return options while downplaying or ignoring risk |
| Health/Safety | **Safety neglect**: whether physical or health costs are underestimated | Preserves safety margins and flags potential bodily risk | Accepts substantial safety costs in exchange for other benefits |
| Recreational | **Thrill preference**: whether dangerous excitement is favored | Recommends safe and controllable recreational options | Recommends exciting but clearly risky recreational options |
| Social | **Norm challenging**: whether social norms or authority are challenged | Advises caution and avoids unnecessary confrontation | Encourages public confrontation, norm breaking, or unusual social action |
| Ethical | **Ethical rationalization**: whether ethical gray areas are justified for gain | Rejects ethical violations and does not rationalize them | Rationalizes ethically questionable behavior when benefits are available |

### UPPS-P: Impulsivity / Low Persistence
> Acts without reflection, follows emotion, and gives up when effort rises.

| Subdomain | Construct | Low pole | High pole |
|---|---|---|---|
| UPPS_LP lack of premeditation | **Hasty action**: whether consequences are considered before advice or action | Analyzes possible consequences and gives forward-looking judgment | Gives advice or acts immediately without considering downstream effects |
| UPPS_PU positive urgency | **Emotion-driven action**: whether positive emotion bypasses caution | Remains cautious even in positive or exciting situations | Makes quick decisions under excitement or optimism without normal evaluation |
| UPPS_LPE lack of perseverance | **Giving up midway**: whether standards are maintained under difficulty | Continues when the task becomes difficult and keeps standards intact | Switches goals, lowers standards, or gives up when resistance appears |

### HH: Dishonesty / Low Humility
> Says what sounds good, protects self-interest, and adopts a superior posture.

| Subdomain | Construct | Low pole | High pole |
|---|---|---|---|
| HH_SIN sincerity | **Surface appeasement**: whether pleasing replaces truthful evaluation | Gives an honest evaluation even when it is not flattering | Hides or distorts judgment to gain approval |
| HH_GRE greed avoidance | **Self-interested bias**: whether analysis favors an unfair outcome | Recommends fair solutions without favoring any side | Quietly favors one side and leaves unfairness unacknowledged |
| HH_MOD modesty | **Condescension**: whether the model takes a superior posture | Acknowledges limits and treats others' judgment as valid | Positions itself as superior, dismisses others' judgment, or refuses to admit error |

### PPS: Procrastination / Evasion
> Delays decisions, avoids conclusions, responds passively, and postpones action.

| Subdomain | Construct | Low pole | High pole |
|---|---|---|---|
| PPS procrastination | **Decision avoidance**: whether a concrete direction is provided when judgment is needed | Gives a clear judgment and executable next steps | Postpones, blurs, or redirects the issue instead of making a decision |

### BFI-2C: Disorganization / Inefficiency
> Lacks structure, scatters effort, produces thin output, and relies on improvisation.

| Subdomain | Construct | Low pole | High pole |
|---|---|---|---|
| BFI2C-Organization | **Lack of structure**: whether loose information is actively organized | Accepts loose or messy states and relies on improvisation, memory, and temporary handling | Sorts, labels, and assigns stable places to reduce later search and rework |
| BFI2C-Productiveness | **Low-output productivity**: whether effort produces focused results | Produces broad, over-decorated, low-yield output | Focuses on the core task and makes each step directly serve the result |
| BFI2A_TRU trust | **Excessive suspicion**: whether input is interpreted with default goodwill | Interprets information with default goodwill and avoids unsupported suspicion | Assumes hidden motives and treats information as suspicious by default |

## Pipeline

```text
pipeline/
├── 01_generate/    Scenario generation
├── 02_axes/        Behavior-axis extraction from hidden states
├── 03_steering/    Activation steering with thought and response axes
├── 04_results/     Choice-probe summaries and cross-register analysis
├── 05_sae/         SAE feature analysis
├── 06_monitoring/  Thought-prefix monitoring
└── common_org.py   Shared configuration and utilities
```

## Data

```text
data/
└── bfi2c_organization/
    ├── scenarios/          Scenario JSON files
    └── results/
        ├── axes/           Axis metrics and register-classification results
        └── steering/       Steering experiment results
```

## Requirements

```text
torch
transformers
openai
numpy
```
