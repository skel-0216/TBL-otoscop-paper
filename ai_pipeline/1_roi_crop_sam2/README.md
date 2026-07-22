# ROI cropping with SAM 2 (the **C** operator)

Generates the *ROI content split* (eardrum-centered zoom crops) that
`../2_wct2_augmentation/augment.ipynb` reads with `CONTENT_VARIANT="roi"`. This is
the first, cropping step of the preprocessing pipeline; style transfer (S),
photometric/geometric augmentation (P), and edge multiply (B) follow in
`../2_wct2_augmentation/`.

## Method

For each endoscope frame:

1. **Segment** the eardrum with SAM 2 (single foreground point at the frame
   centre; the highest-scoring multi-mask output is kept).
2. **Box + buffer.** Take the eardrum bounding box, square it to `max(w, h)`,
   and extend all four sides by `buffer = 0.20` (a fraction of the squared box
   side) so surrounding ear-canal context stays visible.
3. **Keep it black-free.** Shrink the crop if the buffered box would reach past
   the image edge or the circular endoscope field of view, so the visible ROI
   sits entirely inside both — no black padding and no curved FOV edge appear.
4. **Finalize.** Resize to 256×256 and apply an inscribed circular mask.

**Manual review.** Every crop was inspected by hand; whenever the eardrum was
not clearly visible, the image was re-cropped manually. Use `roi_crop_sam2.ipynb`
(the `MANUAL` dict) or `--manual-json` to supply those overrides. The output
manifest flags which images were manual and which were shrunk to fit the FOV.

## Files

| File | Purpose |
|------|---------|
| `roi_crop_sam2.py`      | Shared implementation + command-line batch runner |
| `roi_crop_sam2.ipynb`   | Cell-by-cell workflow: review grid, single-image inspector, manual overrides, save |
| `requirements-sam2.txt` | Extra dependencies (SAM 2; Python ≥ 3.10) |

## Environment

SAM 2 needs Python ≥ 3.10, so its dependencies are kept separate from
`../2_wct2_augmentation/requirements.txt` (Python ≥ 3.9).

```bash
pip install "git+https://github.com/facebookresearch/sam2.git"
pip install -r requirements-sam2.txt
# checkpoint: sam2.1_hiera_large.pt   cfg: configs/sam2.1/sam2.1_hiera_l.yaml
```

## Run

**Notebook** (recommended for review). Open `roi_crop_sam2.ipynb` and set `MODE`
in the first cell:

- **`demo`** — crops the few raw frames under `../demo_data/augmentation/content_raw/`
  and writes to `./demo_output_roi/`.
- **`real`** — reads the raw content split (`f2_raw/train`, the uncropped
  endoscope frames) under `OTOSCOPE_DATA_ROOT` and writes the ROI content split
  (`f2_roi/train`) next to it. A CUDA GPU is expected for the full set.

Set `SAM2_CKPT` to the downloaded checkpoint, then run top to bottom; edit the
`MANUAL` dict after reviewing.

**Command line** (full set):

```bash
python roi_crop_sam2.py \
    --input  <endoscope_frames_dir> \
    --output <roi_content_split_dir> \
    --crop-mode box --buffer 0.20 --size 256 \
    --manifest <roi_content_split_dir>/roi_crop_manifest.csv
# optional: --manual-json overrides.json   --save-mask-dir <dir>   --skip-existing
```

Input is a flat directory of frames whose file names carry the class label
(`{idx}_{CLASS}_{sub}.png`); output is `<stem>_roi.png` (256×256, circular). The
script refuses to write into its input directory.

## Parameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--crop-mode` | `box` | `box` = eardrum bbox + buffer; `circle` = min-enclosing-circle / `--eardrum-frac` |
| `--buffer` | `0.20` | per-side extension as a fraction of the squared eardrum box |
| `--size` | `256` | output side length |
| `--fov-margin` | `0.98` | fraction of the detected FOV radius the ROI may reach (`--no-fit-fov` disables) |
| `--manual-json` | – | `{stem: {point:[x,y]} \| {box:[cx,cy,half]}}` overrides for hand re-cropping |

## Calibrating the buffer

`buffer` is the only free parameter. Because the output frame *is* the crop box,
the eardrum bbox spans `1/(1 + 2·buffer)` of it, so it can be measured back from
any existing crop set:

```bash
python roi_crop_sam2.py --calibrate <existing_roi_crops> [--calibrate-raw <raw_frames>]
```

## Reference

> Ravi, N. et al. (2024). *SAM 2: Segment Anything in Images and Videos.*
> https://github.com/facebookresearch/sam2
