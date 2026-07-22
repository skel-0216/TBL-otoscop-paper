"""Aggregate the three seed-repeat training runs into mean +/- SD.

The full pipeline (augmentation + training) was repeated three times with
different random seeds; seed_runs/ holds each run's summary table verbatim.
The paper's Table 2 reports the run whose summary is summary_seed2.csv.
The four configurations without style transfer (Endoscope, +C, +EB, +C,EB)
were trained once and shared across the repeats, so they are reported as
single-run values.

Usage:
    python seed_stats.py
"""
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = ["summary_seed1.csv", "summary_seed2.csv", "summary_seed3.csv"]
METRICS = ["Oto Acc", "Oto BalAcc", "Oto MacroF1"]


def load(run_file):
    with open(HERE / "seed_runs" / run_file, newline="", encoding="utf-8-sig") as f:
        return {row["Model"]: {m: float(row[m]) for m in METRICS} for row in csv.DictReader(f)}


def main():
    tables = [load(f) for f in RUNS]
    models = list(tables[0].keys())

    out_rows = ["Model," + ",".join(f"{m} mean,{m} SD" for m in METRICS) + ","
                + ",".join(f"{m} (seed{i+1})" for m in METRICS for i in range(len(RUNS)))]

    print(f"{'configuration':<22}" + "".join(f" {m:>20}" for m in METRICS) + "   (mean +/- SD, N=3 seeds)")
    print("-" * 90)
    for model in models:
        vals = {m: np.array([t[model][m] for t in tables]) for m in METRICS}
        shared = all(np.ptp(v) == 0 for v in vals.values())  # identical across runs -> shared model
        cells = [(f"{vals[m].mean():.4f} (single)" if shared
                  else f"{vals[m].mean():.4f} +/- {vals[m].std(ddof=1):.4f}") for m in METRICS]
        print(f"{model:<22}" + "".join(f" {c:>20}" for c in cells))
        row = [f"\"{model}\""]
        for m in METRICS:
            row += [f"{vals[m].mean():.4f}", f"{vals[m].std(ddof=1):.4f}"]
        for m in METRICS:
            row += [f"{t[model][m]:.4f}" for t in tables]
        out_rows.append(",".join(row))

    out = HERE / "seed_stats_results.csv"
    out.write_text("\n".join(out_rows) + "\n", encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
