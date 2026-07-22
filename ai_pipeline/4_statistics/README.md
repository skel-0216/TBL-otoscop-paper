# Statistics: confidence intervals and seed repeats

Uncertainty estimates for the otoscope test-set results in Table 2 of the
paper. Everything in this folder recomputes from the bundled CSVs alone — no
image data or GPU needed (`numpy` only).

## Bootstrap confidence intervals

`predictions/` holds one per-image prediction table per training
configuration (`image,true_label,pred_label`, n = 90), produced by evaluating
the bundled best-epoch weights (`../3_classification/weights/`) on the
otoscope test set.

```
python bootstrap_ci.py            # 10,000 resamples, fixed seed
```

writes `bootstrap_ci_results.csv`. Key rows (top-1 accuracy, percentile 95% CI):

| Configuration | Accuracy | 95% CI |
|---------------|---------:|--------|
| Endoscope (baseline) | 0.6556 | [0.5556, 0.7556] |
| Endoscope + C, S, EB (final) | 0.9556 | [0.9111, 0.9889] |

The final configuration's CI does not overlap the baseline's. CIs of the
top configurations do overlap each other — the ablation's conclusion is the
combined S/C/EB trend, not pairwise separation between the leading rows.

## Seed repeats (N = 3)

The augmentation + training pipeline was repeated three times end-to-end with
different random seeds. `seed_runs/` holds each run's summary table verbatim;
the paper's Table 2 reports the `summary_seed2.csv` run, whose best-epoch
weights are the ones bundled in this repository.

```
python seed_stats.py
```

writes `seed_stats_results.csv` (mean ± sample SD across seeds):

| Configuration | Oto accuracy (N = 3) |
|---------------|---------------------:|
| Endoscope + S | 0.874 ± 0.026 |
| Endoscope + C, S | 0.893 ± 0.032 |
| Endoscope + S, EB | 0.915 ± 0.034 |
| Endoscope + C, S, EB | **0.933 ± 0.019** |

The full combination is the best configuration in the mean across seeds as
well as in the reported run. The four configurations without style transfer
(Endoscope, +C, +EB, +C,EB) do not involve the style-transfer randomness and
were trained once, shared across the repeats — they appear as single-run
values without an SD.

## Notes

- Regenerating `predictions/` itself requires the otoscope test set, which is
  not distributable (see the paper's Data availability section); the tables
  here were produced with the bundled weights and are sufficient to reproduce
  every number in this folder.
- `bootstrap_ci.py --seed`/`--n-iter` change the resampling; the shipped
  results use seed 0 and 10,000 iterations.
