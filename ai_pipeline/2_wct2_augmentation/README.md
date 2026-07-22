# WCT2 style-augmentation

This folder builds the augmented training datasets reported in the paper. Each
training sample passes through three independent operators:

| Tag | Operator | Implementation |
|-----|----------|----------------|
| **S** | Style transfer | WCT2 (Yoo et al., 2019) — decoder-only transfer, `option_unpool=cat5`, 512 px |
| **P** | Photometric / geometric | Albumentations `ReplayCompose` (HFlip, ±180° Affine, GaussianBlur, MotionBlur, RGBShift, RandomBrightness) |
| **B** | Border / edge multiply | Ellipse-gradient vignette, Gaussian-sampled strength ∈ [0.6, 1.0] and gamma ∈ [0.8, 1.2] |

Operators are sampled independently per job (each a Bernoulli with the
configured probability) and applied in the order **S → P → resize(256×256) → B**.
Four dataset versions — `p`, `sp`, `pb`, `spb` — come from changing which
operators are allowed.

In the paper's notation, **S** is the style-transfer operator (S), **B** is the
edge-blurring operator (EB), and the `roi`/`raw` content variant corresponds to
the ROI-cropping operator (C, applied upstream in `../1_roi_crop_sam2/`); **P** is
part of every configuration, including the baseline.

## Files

```
2_wct2_augmentation/
├── augment.ipynb            # entry point: set paths, run
├── pipeline.py              # all reusable pipeline code (WCT2 engine, edge, replay, runner)
├── model.py                 # WCT2 wavelet encoder/decoder (NAVER 2019)
├── utils/
│   ├── core.py              # WCT feature transform
│   └── io.py                # image / segment IO helpers
├── model_checkpoints/
│   ├── wave_encoder_cat5_l4.pth
│   └── wave_decoder_cat5_l4.pth
├── requirements.txt
├── LICENSE-WCT2-MIT.txt
└── THIRD_PARTY_NOTICES.md
```

`model.py`, `utils/`, and the checkpoints come from the WCT2 reference
implementation; see `THIRD_PARTY_NOTICES.md` for licenses (WCT2 = MIT,
FastPhotoStyle-derived utils = CC BY-NC-SA 4.0).

## Running

Install dependencies (the pinned versions match the paper environment):

```bash
pip install -r requirements.txt
```

Open `augment.ipynb` and set `MODE` in the first cell:

- **`demo`** — runs on the small sample under `../demo_data/augmentation/`
  (CPU, a few minutes); output goes to `./demo_output/`.
- **`real`** — reproduces the paper datasets. Set `OTOSCOPE_DATA_ROOT` (or edit
  the path) to your copy of `__dataset/__for_paper`. A CUDA GPU is expected.

`CONTENT_VARIANT` selects `roi` (256-friendly crops) or `raw` (full frames).
Run the cells top to bottom. Section 4 precomputes the per-image ellipse-gradient
parameters once into `cache_<variant>/`; Section 5 defines the four version
recipes and runs them. Each `run_version` writes
`<OUT_ROOT>/<tag>_<suffix>/<class>/*.jpg` plus a metadata CSV. In `real` mode
each version produces ~80 000 images (20 000 per class).

## Input layout (real mode)

- Content split: `f2_<variant>/train/` — one flat directory of PNGs whose file
  names contain the class label (`NORMAL`, `PERFORATION`, `RETRACTION`,
  `TYMPANOSCLEROSIS`). The `roi` split is produced by `../1_roi_crop_sam2/`; the
  `raw` split is the uncropped frames.
- Style pool: `style_pool/` — the otoscope images used as style targets. It
  must be **disjoint from the otoscope test set**: style transfer copies pool
  appearance into the training data, so any image shared with the test set
  leaks test-set appearance into training. Section 3.5 of the notebook verifies
  disjointness by content hash before generating data and stops on any overlap.
  The metadata CSVs record every sampled style in their `style_path` column
  (collected recursively, so class subfolders are fine).

## Reproducibility

Every per-image decision is recorded in the metadata CSV written by
`run_augmentation_random_ops`: `combo_tag`, `apply_style/proc/border`,
`style_path`, the JSON-encoded Albumentations replay, the ellipse-fit
parameters, the sampled vignette `edge_strength`/`edge_gamma`, and the WCT2
engine settings. With the fixed `seed=1337` and the on-disk edge cache, the
pipeline is deterministic for fixed inputs and a fixed dependency set.

## Hyperparameters (paper run)

| Field | Value |
|-------|-------|
| `target_per_class` | 20 000 |
| `final_h`, `final_w` | 256, 256 |
| `seed` | 1337 |
| WCT2 `transfer_at` | `{decoder}` |
| WCT2 `option_unpool` | `cat5` |
| WCT2 `image_size` | 512 |
| WCT2 `alpha` | 1.0 |
| `prob_hflip` | 0.5 |
| `prob_vflip` | 0.0 |
| `prob_rotate` | 0.9 effective (limit ±180°) — see note below |
| `gblur` | kernel 7, p = 0.75 |
| `mblur` | kernel ≤ 15, p = 0.25 |
| `rgbshift` | (±15, ±15, ±15), p = 0.35 |
| `rbrightness` | ±0.15, p = 0.85 |
| edge `thresh1` | 30 |
| edge `morph_kernel` | 15 |
| edge `fade_ratio` | 0.20 |
| edge `strength_range` | (0.6, 1.0) truncated Gaussian |
| edge `gamma_range` | (0.8, 1.2) truncated Gaussian |
| `op_probs` per version | `p` / `p,s=0.9` / `p,b=0.85` / `p,s=0.9,b=0.85` |

Note on `prob_rotate`: the run scripts pass `prob_rotate=0.99`, but
`run_augmentation_random_ops` does not forward the flip/rotate kwargs to the
per-job replay builder, so the builder defaults apply — HFlip 0.5, Rotate 0.9
(±180°). The paper's metadata CSVs record exactly these effective values
(`"name": "Affine", "p": 0.9`). The kwargs are kept verbatim for fidelity with
the original scripts.
