# BFI2C Organization Behavior-Mode Pipeline

Scripts are grouped by experiment stage. Shared helpers stay at the pipeline root in `common_org.py`; repository data lives under `data/bfi2c_organization/`.

## 01 Generate

Create or revise scenario JSON files. Axis-building scenarios live under `data/bfi2c_organization/scenarios/axis/`; held-out choice-test scenarios live under `data/bfi2c_organization/scenarios/test/`.

```bash
COUNT=50 MODEL=claude-opus-4-6 bash pipeline/01_generate/generate_choice_suite.sh
```

The axis scenario generator is config-driven:

```bash
python pipeline/01_generate/generate_axis_scenarios.py \
  --config bfi2c_organization \
  --register advice_daily \
  --all-items \
  --n 10 \
  --out data/bfi2c_organization/scenarios/axis/axis_advice_daily.json
```

`choice_advice_daily.json` is generated from the original BFI Organization item anchors in an advice-seeking daily-life register.

Generated files:

- `scenarios/test/choice_fp.json`
- `scenarios/test/choice_task.json`
- `scenarios/test/choice_advice_daily.json`
- `scenarios/test/choice_advice_task.json`

## 02 Axes

Build and inspect behavior axes.

- `axis_diagnostics.py`: one-file-per-axis metrics and PCA diagnostics with thought/response side-by-side.
- `axis_metrics.py`: split-half, pairwise cosine, thought/response axis metrics.
- `axis_pca.py`: centered/uncentered PCA on diff vectors.
- `run_axis_suite.sh`: run combined diagnostics for the maintained axis files.
- `axis_projection_eval.py`: direct projection sanity check.
- `thought_shift_eval.py`: prompt-conditioned thought-shift sanity check.

```bash
LAYER=11 bash pipeline/02_axes/run_axis_suite.sh
```

## 03 Steering

Run dense thought/response activation steering.

```bash
python pipeline/03_steering/steer.py \
  --config bfi2c_organization \
  --axis-source data/bfi2c_organization/scenarios/axis/axis_fp.json \
  --axes both \
  --target task_choice \
  --first-n 50 \
  --coeffs -3:3:1
```

To run the full 4 axis-source by 4 choice-target matrix:

```bash
CUDA_VISIBLE_DEVICES=0 bash pipeline/03_steering/run_choice_transfer_matrix.sh
```

## 04 Results

Post-process steering outputs.

- `regroup_results.py`: regroup old axis/feature outputs by scenario.
- `summarize_choices.py`: summarize A/B choices in one steering result by coefficient.
- `summarize_choice_transfer_matrix.py`: summarize the full transfer matrix as safe-choice rates.

## 05 SAE

SAE feature analysis and feature steering.

- `sae_compare_axes.py`: compare thought vs response axes in SAE feature space and find shared features.
- `sae_steer_single_features.py`: steer with individual SAE decoder features.
- `sae_steer_shared_features.py`: steer with shared SAE feature bundles.

## 06 Monitoring

Thought/behavior-mode projection experiments.

- `org_register_shift_classify.py`: classify a segment against FP/task/advice axes using raw, mean-shift, or last-token shift variants.
- `org_prefix_thought_classify.py`: prefix curve on labeled risky/safe thoughts.
- `org_natural_prefix_register_eval.py`: classify natural coeff-0 thought prefixes into register axes.
- `org_natural_prefix_choice_eval.py`: test whether natural thought prefixes predict final A/B choices.
- `org_coeff_thought_eval.py`: evaluate steered thoughts across coefficients, with pole labels from coefficient sign or parsed model choice.

Example:

```bash
python pipeline/06_monitoring/org_coeff_thought_eval.py \
  --axis-file fp=data/bfi2c_organization/scenarios/axis/axis_fp.json \
  --axis-file task=data/bfi2c_organization/scenarios/axis/axis_task.json \
  --axis-file advice=data/bfi2c_organization/scenarios/axis/axis_advice.json \
  --eval-steering task=data/bfi2c_organization/results/steering/fp_axes_to_task_choice_l11_m3_p3.json \
  --pole-source choice \
  --coeffs=-4,-2,0,2,4 \
  --out data/bfi2c_organization/results/axes/example_choice_pole_eval.json
```

## Removed

- `axis_preference_probe.py` was an early one-off endpoint projection probe. The maintained register/prefix monitoring scripts in `06_monitoring/` cover the same line of analysis more directly.
