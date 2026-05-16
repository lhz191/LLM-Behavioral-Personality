# BFI2C Organization Data

This directory contains the Organization subdomain data used by the behavior-mode pipeline.

## Scenarios

Current maintained axis-building files:

- `scenarios/axis/axis_fp.json`: first-person thought/response pairs for building FP axes.
- `scenarios/axis/axis_task.json`: task-register thought/response pairs for building task axes.
- `scenarios/axis/axis_advice.json`: advice-register thought/response pairs for building advice axes.
- `scenarios/axis/axis_advice_daily.json`: daily advice scenarios generated from the original BFI item anchors.

Generated held-out test choice probes:

- `scenarios/test/choice_fp.json`
- `scenarios/test/choice_task.json`
- `scenarios/test/choice_advice_daily.json`
- `scenarios/test/choice_advice_task.json`

Legacy first-five probes are kept under `scenarios/legacy/` for traceability:

- `scenarios/legacy/fp_choice_first5.json`
- `scenarios/legacy/task_choice_first5.json`
- `scenarios/legacy/fp_natural_choice_first5.json`
- `scenarios/legacy/task_natural_choice_first5.json`

## Results

Dense steering results should be regenerated from the held-out choice probes. The expected matrix filenames are:

- `results/steering/{fp,task,advice,advice_daily}_axes_to_{fp_choice,task_choice,advice_daily_choice,advice_task_choice}_l11_m3_p3.json`

The summary file can be generated as:

```bash
python pipeline/04_results/summarize_choice_transfer_matrix.py \
  data/bfi2c_organization/results/steering/*_l11_m3_p3.json \
  --out data/bfi2c_organization/results/steering/choice_transfer_matrix_summary.json
```
