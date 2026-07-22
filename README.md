# AI-enabled Smart Otoscope Platform for Ear Disease Diagnosis Using Style-Based Domain Adaptation

Code release for the paper. It contains the two software components reported
there: the device software for the custom wireless digital otoscope and its
smart-device clients, and the AI pipeline that builds the style-adapted
training data and trains the diagnostic classifier.

## System overview

The otoscope is built around a Raspberry Pi Zero 2 W with a Pi Camera Module 3
behind a commercial otoscope lens. It runs its own Wi-Fi hotspot and streams
H.264 video over TCP; an Android smartphone or Google Glass client shows the
live view, controls the LED, captures frames, and classifies the captured
eardrum image on-device (PyTorch Mobile, four classes: normal, perforation,
retraction, tympanosclerosis).

The classifier is trained on hospital endoscope images adapted to the otoscope
domain. Three components bridge the device gap, labeled S, C, and EB in the
paper: WCT2 photorealistic style transfer (S), SAM 2 eardrum ROI cropping (C),
and edge-blurring that reproduces the otoscope's optical falloff (EB). A
ResNet-18 is trained per configuration; the full combination raises accuracy
on the otoscope test set from 65.6% (endoscope-trained baseline) to 95.5%.

## Repository layout

```
device_software/            # Section 1 — the device and its clients
├── raspberrypi_server/     #   Pi camera server: H.264 stream + command/LED port
├── phone_otoview/          #   Android phone client (Kotlin/Compose, on-device AI)
└── glass_otoscope/         #   Google Glass client (gestures + voice commands)

ai_pipeline/                # Section 2 — data preparation and training
├── 1_roi_crop_sam2/        #   stage 1: SAM 2 eardrum ROI cropping (C)
├── 2_wct2_augmentation/    #   stage 2: WCT2 style transfer + augmentation (S, EB)
├── 3_classification/       #   stage 3: ResNet-18 training and evaluation
├── 4_statistics/           #   bootstrap CIs and seed-repeat statistics
└── demo_data/              #   small de-identified sample driving the demo mode

environment.yml             # full conda environment of the paper (py39)
```

Each folder has its own `README.md` with parameters, data layouts, and run
instructions.

## Quick start

The three `ai_pipeline` notebooks each have a `MODE` switch in their first
cell. `MODE = "demo"` runs end-to-end on the bundled `demo_data/` in a few
minutes on CPU — no dataset or GPU required:

1. `1_roi_crop_sam2/roi_crop_sam2.ipynb` — crop frames to the eardrum ROI
   (needs SAM 2 and its checkpoint; see `1_roi_crop_sam2/README.md`)
2. `2_wct2_augmentation/augment.ipynb` — build the augmented datasets
3. `3_classification/train.ipynb` — train ResNet-18 and report metrics

To evaluate the paper's trained models without training anything, open
`3_classification/train.ipynb`, section 4 ("Final model on the otoscope test
set"): it loads one of the eight bundled checkpoints
(`3_classification/weights/`) and evaluates it. In `demo` mode this runs on the
small bundled `demo_data/` sample; in `real` mode it runs on the full otoscope
test set you provide.

The clinical image data is not distributed (see Data availability). Reproducing
augmentation or the full training runs needs your own copy: point the
`OTOSCOPE_DATA_ROOT` environment variable at it and set `MODE = "real"`
(see `ai_pipeline/README.md`).

For the device software, `raspberrypi_server/README.md` covers the Pi setup
and the phone/Glass apps build with Android Studio (`device_software/README.md`).

## Environment

`environment.yml` is the exact conda environment used for the paper. Each
`ai_pipeline` folder also has a minimal `requirements.txt`; note that
`wct2_augmentation` pins `albumentations==1.3.1` (the paper version — 1.4+
changes the Affine/ReplayCompose behaviour the pipeline depends on) and
`roi_crop_sam2` needs Python ≥ 3.10 for SAM 2.

## Data and trained models

- `demo_data/` — a small de-identified subset of the study images (a few per
  class), included only to exercise the pipeline end to end in demo mode.
- `3_classification/weights/` — the eight best-epoch ResNet-18 checkpoints, one
  per training configuration (~45 MB each, weights only).
- The phone and Glass apps bundle the deployed classifier as `model.ptl`.

The full clinical dataset was collected under IRB approval
(AJIRB-MED-OBS-21-409) and is available under the conditions stated in the
paper's Data availability section; the `demo_data/` subset above is the only
image data distributed with the code.

## Results

Accuracy on the otoscope test set (n = 90), by training configuration
(Table 2 of the paper), with 95% bootstrap confidence intervals from
`ai_pipeline/4_statistics/`:

| Configuration | Accuracy | 95% CI |
|---------------|---------:|:------:|
| Endoscope baseline | 0.656 | [0.556, 0.756] |
| + C | 0.656 | [0.556, 0.756] |
| + EB | 0.800 | [0.711, 0.878] |
| + C, EB | 0.856 | [0.778, 0.922] |
| + S | 0.844 | [0.767, 0.911] |
| + C, S | 0.911 | [0.844, 0.967] |
| + S, EB | 0.878 | [0.800, 0.944] |
| + C, S, EB | **0.956** | [0.911, 0.989] |

The full S + C + EB configuration is the paper's final model (95.5%). Repeating
the whole augmentation-and-training pipeline over three random seeds gives
0.933 ± 0.019 for that configuration, still the best of the eight; see
`ai_pipeline/4_statistics/`. `3_classification/README.md` maps the
configurations to the eight bundled runs and weights.

## License and third-party code

| Component | License |
|-----------|---------|
| Code in this repository (unless noted below) | MIT (see `LICENSE`) |
| `ai_pipeline/2_wct2_augmentation/model.py` + WCT2 checkpoints (NAVER Corp.) | MIT |
| `ai_pipeline/2_wct2_augmentation/utils/core.py`, `io.py` (derived from NVIDIA FastPhotoStyle) | CC BY-NC-SA 4.0 |
| `device_software/glass_otoscope/.../GlassGestureDetector.kt` (derived from Google glass-enterprise-samples) | Apache 2.0 |
| PyTorch Mobile runtime (bundled by the Android apps) | BSD-3-Clause |

Details and full texts: `ai_pipeline/2_wct2_augmentation/THIRD_PARTY_NOTICES.md`,
`ai_pipeline/2_wct2_augmentation/LICENSE-WCT2-MIT.txt`,
`device_software/glass_otoscope/LICENSE-Apache-2.0.txt`. Note that the two
FastPhotoStyle-derived files are non-commercial / share-alike.
