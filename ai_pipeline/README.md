# AI pipeline: preprocessing, augmentation, and classification

This section contains the code that preprocesses the endoscope frames, produces
the augmented training datasets, and trains the ResNet-18 classifier reported in
the paper. It has three stages, each driven by a single notebook:

| Folder | Notebook | What it does |
|--------|----------|--------------|
| `1_roi_crop_sam2/`     | `roi_crop_sam2.ipynb` | Crops each endoscope frame to the eardrum ROI with SAM 2 (**C**), producing the ROI content split. Needs SAM 2 and its checkpoint. |
| `2_wct2_augmentation/` | `augment.ipynb` | Builds the augmented datasets: WCT2 style transfer (**S**) + photometric/geometric ops (**P**) + border vignette (**B**). |
| `3_classification/`    | `train.ipynb`   | Trains ResNet-18 on an augmented dataset and reports top-1 / top-2 accuracy and a confusion matrix. |
| `4_statistics/`        | `*.py`          | Bootstrap 95% confidence intervals and the three-seed repeat statistics behind Table 2. Recompute from bundled CSVs; no image data needed. |

The reusable logic lives in `.py` modules next to each notebook; the notebooks
only set paths and call it.

The paper's ablation labels the components S (style transfer), C (ROI
cropping), and EB (edge-blurring): **C** is this repository's `roi`/`crop`
content variant, **B** is EB, and **P** (photometric/geometric) is part of
every configuration, including the baseline.

## Two run modes

Both notebooks have a `MODE` switch in their first configuration cell.

- **`demo`** — runs on the small sample under `demo_data/` (CPU). The
  augmentation notebook produces a few style-transferred images; the
  classification notebook skips training (the sample is too small) and instead
  evaluates the bundled final model on a few otoscope images. A few minutes end
  to end.
- **`real`** — uses the full dataset. The bundled `real_data/` holds the
  four-class **test sets**, so in `real` mode the "evaluate provided weights"
  path runs on them directly. Reproducing augmentation and training on the full
  data needs the complete dataset: point the `OTOSCOPE_DATA_ROOT` environment
  variable at your own copy of `__for_paper`. A CUDA GPU is expected.

## Order of use

1. `1_roi_crop_sam2/roi_crop_sam2.ipynb` crops the endoscope frames to the eardrum
   ROI with SAM 2, producing the ROI content split. The uncropped frames serve as
   the raw content split.
2. `2_wct2_augmentation/augment.ipynb` writes the augmented datasets
   (`dataset_{p,sp,pb,spb}_f2_{crop,raw}/`). `real` mode reads the endoscope
   content splits and the otoscope style pool from `OTOSCOPE_DATA_ROOT`.
3. `3_classification/train.ipynb` trains on one of those datasets and evaluates it.
   Training is run by the paper's own `3_classification/train.py`; the notebook
   invokes it and then evaluates the result. The notebook can also load a bundled
   best-epoch weight and evaluate it without training — this runs on the bundled
   `real_data/` test sets directly.

Each folder has its own `README.md` with the parameter tables, the run list, and
the data layout it expects. Install per-folder dependencies from the matching
`requirements.txt`; exact paper versions are in the repository `environment.yml`.

## demo_data/

A small de-identified subset of the study images, used only by `MODE = "demo"`:

```
demo_data/
├── augmentation/
│   ├── content_raw/       # one endoscope frame per class (raw split), class in filename
│   ├── content_roi/       # the same ids, ROI-cropped split
│   └── styles/            # a few otoscope style images (flat, class-agnostic pool)
└── classification/
    └── otoscope_sample/   # a few otoscope test images per class (00_/01_/02_/03_),
                           # evaluated by train.ipynb section 4 with the bundled model
```

It exists to exercise the pipeline, not to train a usable model — the sample is
far too small for that (`MODE = "demo"` in `train.ipynb` evaluates the bundled
weights instead of training).

## real_data/

The four-class **test sets** used by `MODE = "real"` for the "evaluate provided
weights" path:

```
real_data/
├── roi_test/    # endoscope, cropped ROI   (00_/01_/02_/03_ class subfolders)
├── raw_test/    # endoscope, full frame
└── otoscope/    # external otoscope test set
```

Full augmentation/training reproduction is not driven from here; it uses the
complete `__for_paper` dataset via `OTOSCOPE_DATA_ROOT` (see `real_data/README.md`).

## License and attribution

`2_wct2_augmentation/` bundles the WCT2 model and utilities from the reference
implementation; see `2_wct2_augmentation/README.md` and the `LICENSE-*` files there
for the WCT2 (MIT) and FastPhotoStyle (CC BY-NC-SA 4.0) terms. The classification
code and the trained ResNet-18 weights are released with the paper.
