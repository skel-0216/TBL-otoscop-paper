# Classification (ResNet-18)

This folder trains and evaluates the ResNet-18 classifier reported in the paper.
The classifier is trained once per (augmentation version × content variant)
combination — eight runs in total — over four classes: `NORMAL`, `PERFORATION`,
`RETRACTION`, `TYMPANOSCLEROSIS`.

## Files

```
3_classification/
├── train.ipynb            # entry point: runs train.py, then evaluates
├── train.py               # the paper's trainer (one run = one dataset + trial)
├── eval_lib.py            # load a checkpoint, predict, metrics + confusion matrix
├── model.py               # ResNet (BasicBlock / Bottleneck)
├── configs.py             # ResNet18/34/50/101/152 configs
├── dataset.py             # FolderClassDataset (folder-per-class loader)
├── functions.py           # train() / evaluate() top-k loops
├── requirements.txt
└── weights/               # eight best-epoch checkpoints (see weights/README.md)
```

`train.py` is the paper's original command-line trainer (comments translated to
English; logic unchanged). The notebook calls it; you can also run it directly:

```bash
# one run, pointing at your data root
OTOSCOPE_DATA_ROOT=/path/to/__dataset/__for_paper \
  python train.py --dataset dataset_sp_f2_crop --trial 1
```

## Data layout

Each training dataset is a folder of PNGs grouped into class subfolders prefixed
`00_…`, `01_…`, … so they sort into a canonical label order:

```
<dataset>/
├── 00_NORMAL/*.png
├── 01_PERFORATION/*.png
├── 02_RETRACTION/*.png
└── 03_TYMPANOSCLEROSIS/*.png
```

Validation folders (`roi_test`, `raw_test`, `otoscope`, …) use the same layout.
The eight training folders follow `dataset_{ver}_f2_{raw|crop}/`, produced by the
companion `../2_wct2_augmentation/`.

## Running

```bash
pip install -r requirements.txt
```

Open `train.ipynb` and set `MODE` in the first cell:

- **`demo`** — no training (the bundled sample is a few images per class, too
  small to train on). It runs section 4 only: the bundled final model evaluated
  on the small otoscope sample under `../demo_data/classification/otoscope_sample/`
  (CPU, seconds).
- **`real`** — trains the full run (100 epochs). Set `OTOSCOPE_DATA_ROOT` (or
  edit the path) to your copy of `__dataset/__for_paper`. A CUDA GPU is expected.

`DATASET` selects which augmented dataset to train on. In `real` mode the
notebook then:

1. trains ResNet-18 and writes checkpoints to `./__checkpoints/<dataset>/`
   (it first checks that the augmented dataset exists and says where to get it
   if not);
2. evaluates the best checkpoint (top-1 / top-2 accuracy, per-class report,
   confusion matrix);
3. loads the bundled final weights (`weights/dataset_spb_f2_crop_best.pt`,
   the paper's S + C + EB model) and evaluates them on the full otoscope test
   set (`../real_data/otoscope`, n = 90) — **no training required**,
   reproducing the paper's 95.5% cross-device result. Any of the eight
   bundled weights can be substituted (see `weights/README.md`).

Step 3 runs in `demo` mode too, over the small bundled `otoscope_sample/`
(steps 1–2 are skipped there).

Defaults match the paper: ResNet-18, dropout 0.3, Adam @ lr 5e-5, weight decay
1e-4, batch 64, 100 epochs, early-stopping patience 20, image size 256×256,
top-2 accuracy alongside top-1. The validation set that drives best-model
selection is `Test`, auto-paired by dataset name (`*_raw` → `raw_test`,
`*_crop` → `roi_test`).

## The eight runs and best epochs

The paper's ablation (Table 2) labels the three components S (WCT2 style
transfer), C (ROI cropping), and EB (edge-blurring). In the dataset names here,
`s` = S, `b` = EB, and the `crop`/`raw` variant = with/without C; `p` (the
photometric/geometric augmentation) is part of every configuration, including
the baseline.

| Run | Paper condition | Best epoch | Oto accuracy |
|-----|-----------------|-----------:|-------------:|
| `dataset_p_f2_raw`   | baseline    |  9 | 0.656 |
| `dataset_p_f2_crop`  | + C         | 17 | 0.656 |
| `dataset_pb_f2_raw`  | + EB        | 11 | 0.800 |
| `dataset_pb_f2_crop` | + C, EB     | 11 | 0.856 |
| `dataset_sp_f2_raw`  | + S         | 23 | 0.844 |
| `dataset_sp_f2_crop` | + C, S      | 37 | 0.911 |
| `dataset_spb_f2_raw` | + S, EB     | 23 | 0.878 |
| `dataset_spb_f2_crop`| + C, S, EB  | 37 | **0.956** |

Best epochs are one-based (matching the epoch numbers printed during training)
and are the epochs chosen on the test set for the paper, not the training-time
validation-loss checkpoint. All eight runs use trial 1; these checkpoints are
bundled in `weights/` (see `weights/README.md`).

## Scope

This notebook covers training and basic metrics (accuracy and a single
confusion matrix). The full per-subset evaluation across all five validation
sets and the Figure-5 confusion-matrix panels are produced separately.
