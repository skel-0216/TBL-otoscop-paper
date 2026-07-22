"""Bootstrap 95% confidence intervals for the otoscope test-set metrics.

Resamples the 90 test images with replacement (default 10,000 iterations,
fixed RNG seed) and reports percentile CIs for top-1 accuracy and macro-F1.
Inputs are the per-image prediction tables under predictions/, produced by
evaluating the bundled best-epoch weights (../3_classification/weights/) on
the otoscope test set — so the CIs can be recomputed from this folder alone,
without the image data.

Usage:
    python bootstrap_ci.py                 # all eight configurations
    python bootstrap_ci.py --n-iter 20000 --seed 1
"""
import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# prediction table -> label used in the paper's Table 2
CONFIGS = [
    ("dataset_p_f2_raw_otoscope_test.csv",    "Endoscope"),
    ("dataset_p_f2_crop_otoscope_test.csv",   "Endoscope + C"),
    ("dataset_pb_f2_raw_otoscope_test.csv",   "Endoscope + EB"),
    ("dataset_pb_f2_crop_otoscope_test.csv",  "Endoscope + C, EB"),
    ("dataset_sp_f2_raw_otoscope_test.csv",   "Endoscope + S"),
    ("dataset_sp_f2_crop_otoscope_test.csv",  "Endoscope + C, S"),
    ("dataset_spb_f2_raw_otoscope_test.csv",  "Endoscope + S, EB"),
    ("dataset_spb_f2_crop_otoscope_test.csv", "Endoscope + C, S, EB"),
]


def load_predictions(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["true_label"]) for r in rows])
    p = np.array([int(r["pred_label"]) for r in rows])
    return y, p


def macro_f1(y, p, n_cls=4):
    f1s = []
    for c in range(n_cls):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def bootstrap(y, p, n_iter, rng):
    n = len(y)
    accs = np.empty(n_iter)
    f1s = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        accs[i] = np.mean(y[idx] == p[idx])
        f1s[i] = macro_f1(y[idx], p[idx])
    return accs, f1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "bootstrap_ci_results.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = ["Model,Oto Acc,Acc CI95 low,Acc CI95 high,Oto MacroF1,F1 CI95 low,F1 CI95 high"]
    print(f"{args.n_iter} bootstrap iterations, seed={args.seed}\n")
    print(f"{'configuration':<22} {'Acc':>7}  {'95% CI':>17} {'MacroF1':>8}  {'95% CI':>17}")
    print("-" * 78)
    for fname, label in CONFIGS:
        y, p = load_predictions(HERE / "predictions" / fname)
        acc = float(np.mean(y == p))
        f1 = macro_f1(y, p)
        accs, f1s = bootstrap(y, p, args.n_iter, rng)
        alo, ahi = np.percentile(accs, [2.5, 97.5])
        flo, fhi = np.percentile(f1s, [2.5, 97.5])
        print(f"{label:<22} {acc:>7.4f}  [{alo:.4f}, {ahi:.4f}] {f1:>8.4f}  [{flo:.4f}, {fhi:.4f}]")
        rows.append(f"\"{label}\",{acc:.4f},{alo:.4f},{ahi:.4f},{f1:.4f},{flo:.4f},{fhi:.4f}")

    Path(args.out).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
