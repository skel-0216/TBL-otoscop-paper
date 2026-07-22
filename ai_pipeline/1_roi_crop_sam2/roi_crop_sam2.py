#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
roi_crop_sam2.py -- Automated ROI Cropping (the C operator of the preprocessing
pipeline; "Zoomed-Cropping").

This is the ROI-generation step that produces the *ROI content split* consumed
by ``data_organizer_roi.ipynb``. Each endoscope frame is cropped to a zoomed,
eardrum-centered ROI:

    1. Segment the eardrum with SAM 2 (single foreground point at the frame
       center; the highest-scoring of the multi-mask outputs is kept).
    2. Take the axis-aligned bounding box of the eardrum mask and square it to
       ``max(w, h)``.
    3. Extend all four sides by ``buffer`` (a fraction of the squared box side)
       so that surrounding ear-canal context is retained.
    4. Shrink the crop, if necessary, so the visible circular ROI stays entirely
       inside BOTH the image rectangle AND the circular endoscope field of view
       -- i.e. no black padding and no curved FOV edge appear in the output
       (see ``fit_half_no_black`` / ``--fit-fov``).
    5. Resize to 256x256 and apply an inscribed circular mask (the endoscope
       field of view is circular).

Manual review (part of the paper procedure). Every crop was inspected by hand;
whenever the eardrum was not clearly visible the image was re-cropped manually.
That step is preserved here: ``roi_crop_sam2.ipynb`` shows each crop for review,
and per-image manual overrides (a better SAM point, or an explicit crop window)
are supplied through ``--manual-json`` / the notebook's ``MANUAL`` dict and are
flagged in the output manifest.

Input  : one flat directory of endoscope frames, file names containing the
         class label (e.g. ``0007_NORMAL_0007.png``).
Output : one flat directory of ``<stem><suffix>.png`` crops (256x256, circular).

Hyper-parameters for the paper run: ``--crop-mode box``, ``--buffer 0.20``,
``--size 256``, FOV fitting on.

The buffer can be re-derived from an existing set of ROI crops with
``--calibrate``: the output frame *is* the crop box, so the eardrum bbox occupies
``1/(1 + 2*buffer)`` of the frame, giving
``buffer = (frame / eardrum_bbox_px - 1) / 2``; ``--calibrate-raw`` additionally
cross-checks the crop scale against the raw frames.

Segmentation model:
    Ravi et al. (2024), "SAM 2: Segment Anything in Images and Videos."
    Checkpoint ``sam2.1_hiera_large.pt`` with cfg
    ``configs/sam2.1/sam2.1_hiera_l.yaml``.
    Requires Python >= 3.10 and the ``sam2`` package
    (``pip install "git+https://github.com/facebookresearch/sam2.git"``);
    see requirements-sam2.txt.

A ``--crop-mode circle`` variant (min-enclosing-circle scaled by
``--eardrum-frac``) is also provided for reference.
"""
import os
import sys
import csv
import json
import glob
import time
import argparse

import numpy as np
import cv2
import torch


# --------------------------------------------------------------------------- #
# I/O (unicode-path safe)
# --------------------------------------------------------------------------- #
def imread_rgb(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path, img_rgb):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1].lower() or ".png"
    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(path)


def imwrite_gray(path, mask_u8):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(path)[1].lower() or ".png", mask_u8)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(path)


# --------------------------------------------------------------------------- #
# SAM2 predictor (lazy singleton)
# --------------------------------------------------------------------------- #
_SAM2_PREDICTOR = None


def load_sam2_predictor(ckpt, cfg, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(cfg, str(ckpt), device=str(device))
    return SAM2ImagePredictor(model)


def init_predictor(ckpt, cfg, device):
    global _SAM2_PREDICTOR
    if _SAM2_PREDICTOR is None:
        _SAM2_PREDICTOR = load_sam2_predictor(ckpt, cfg, device)
    return _SAM2_PREDICTOR


def sam2_roi_mask(img_rgb, point_xy=None):
    """Estimate eardrum ROI mask (bool HxW) with a single foreground point."""
    h, w = img_rgb.shape[:2]
    if point_xy is None:
        point_xy = (w / 2.0, h / 2.0)          # eardrum sits centrally -> center prompt
    _SAM2_PREDICTOR.set_image(img_rgb)
    masks, scores, _ = _SAM2_PREDICTOR.predict(
        point_coords=np.array([point_xy], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    return masks[int(np.argmax(scores))].astype(bool)


# --------------------------------------------------------------------------- #
# Mask geometry
# --------------------------------------------------------------------------- #
def eardrum_bbox(mask_bool):
    """Axis-aligned bounding box (x, y, w, h) of the largest mask blob."""
    m = (mask_bool.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = mask_bool.shape[:2]
        return int(w * 0.25), int(h * 0.25), int(w * 0.5), int(h * 0.5)
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return int(x), int(y), int(w), int(h)


def eardrum_circle(mask_bool):
    """Min enclosing circle (cx, cy, r) of the largest mask blob."""
    m = (mask_bool.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = mask_bool.shape[:2]
        return w / 2.0, h / 2.0, min(w, h) / 4.0
    (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return float(cx), float(cy), float(r)


def crop_window_box(mask_bool, buffer):
    """box mode: square around the eardrum bbox, extended each side by
    `buffer` (fraction of the squared box side). Returns (cx, cy, half_side)."""
    x, y, w, h = eardrum_bbox(mask_bool)
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h)                                  # square the eardrum box
    half = (side / 2.0) * (1.0 + 2.0 * float(buffer))  # extend all four sides
    return cx, cy, half


def crop_window_circle(mask_bool, eardrum_frac):
    """circle mode: crop circle s.t. eardrum diameter = eardrum_frac * crop
    diameter. Returns (cx, cy, half_side) with half_side = R = r/frac."""
    cx, cy, r = eardrum_circle(mask_bool)
    R = r / max(float(eardrum_frac), 1e-3)
    return cx, cy, R


def crop_is_zoom_in(img_rgb, cx, cy, half):
    """True if the crop square fits inside the image (a genuine zoom-in)."""
    h, w = img_rgb.shape[:2]
    return half <= min(cx, cy, w - cx, h - cy) + 0.5


def fov_disk(img_rgb, thresh=10, morph=15):
    """Estimate the circular endoscope field of view (fx, fy, R) as the min
    enclosing circle of the non-black region."""
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return w / 2.0, h / 2.0, min(w, h) / 2.0
    (fx, fy), R = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return float(fx), float(fy), float(R)


def fit_half_no_black(img_rgb, cx, cy, half, fov_margin=0.98, fov=None):
    """Shrink `half` so the visible circular ROI (centre (cx,cy), radius half)
    stays inside BOTH the image rectangle and the circular FOV -- guaranteeing
    no black padding and no curved FOV edge in the output. Centre is kept on the
    eardrum. Returns (half_fitted, (fx, fy, R))."""
    h, w = img_rgb.shape[:2]
    half = min(half, cx, cy, w - cx, h - cy)          # inside image rectangle
    fx, fy, R = fov if fov is not None else fov_disk(img_rgb)
    d = float(np.hypot(cx - fx, cy - fy))
    half = min(half, max(1.0, R * float(fov_margin) - d))  # inside circular FOV
    return float(half), (fx, fy, R)


def square_crop(img_rgb, cx, cy, r):
    """Square crop centered at (cx,cy), half-side r; out-of-frame -> black pad."""
    h, w = img_rgb.shape[:2]
    x0, y0 = int(round(cx - r)), int(round(cy - r))
    x1, y1 = int(round(cx + r)), int(round(cy + r))
    side = max(x1 - x0, y1 - y0)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    sx, sy = -x0, -y0
    ix0, iy0 = max(x0, 0), max(y0, 0)
    ix1, iy1 = min(x1, w), min(y1, h)
    canvas[iy0 + sy:iy1 + sy, ix0 + sx:ix1 + sx] = img_rgb[iy0:iy1, ix0:ix1]
    return canvas


def inscribed_circle_mask(side, feather_px=0):
    cc = side / 2.0
    m = np.zeros((side, side), dtype=np.float32)
    cv2.circle(m, (int(round(cc)), int(round(cc))), int(round(cc)), 1.0, -1, lineType=cv2.LINE_AA)
    if feather_px and feather_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather_px)
    return m


def crop_roi_circular(img_rgb, cx, cy, r, feather_px=0):
    canvas = square_crop(img_rgb, cx, cy, r)
    m = inscribed_circle_mask(canvas.shape[0], feather_px)
    return (canvas.astype(np.float32) * m[..., None]).clip(0, 255).astype(np.uint8)


def resize_sq(img, size):
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if size < max(h, w) else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


# --------------------------------------------------------------------------- #
# Per-image C step
# --------------------------------------------------------------------------- #
def make_roi_crop(img_rgb, mode, buffer, eardrum_frac, size,
                  circular_mask=True, feather_px=0, point_xy=None,
                  fit_fov=True, fov_margin=0.98, return_extra=False):
    mask = sam2_roi_mask(img_rgb, point_xy)            # SAM2 eardrum ROI
    if mode == "circle":
        cx, cy, half = crop_window_circle(mask, eardrum_frac)
    else:
        cx, cy, half = crop_window_box(mask, buffer)
    half_req = half
    fov = None
    if fit_fov:
        half, fov = fit_half_no_black(img_rgb, cx, cy, half, fov_margin)
    clamped = half < half_req - 0.5
    if circular_mask:
        out = resize_sq(crop_roi_circular(img_rgb, cx, cy, half, feather_px), size)
    else:
        out = resize_sq(square_crop(img_rgb, cx, cy, half), size)
    meta = dict(cx=cx, cy=cy, half=half, half_req=half_req, clamped=int(clamped),
                fov_R=("" if fov is None else round(fov[2], 1)))
    if return_extra:
        return out, mask, meta
    return out, None, meta


# --------------------------------------------------------------------------- #
# Calibration: recover the buffer from an existing set of ROI crops
# --------------------------------------------------------------------------- #
def _measure_eardrum(path):
    """Return (frame_side, bbox_w, bbox_h, circle_r) by running SAM2 once on `path`."""
    img = imread_rgb(path)
    frame = min(img.shape[:2])
    mask = sam2_roi_mask(img)                 # one SAM2 pass; reuse for bbox + circle
    _, _, bw, bh = eardrum_bbox(mask)
    _, _, r = eardrum_circle(mask)
    return frame, bw, bh, r


def calibrate(out_dir, raw_dir, n):
    """Recover the buffer (box mode) and eardrum_frac (circle mode) by measuring
    an existing set of ROI crops. If raw_dir is given, cross-check crop scale
    on the same stems (raw eardrum px * scale should match output eardrum px)."""
    outs = sorted(glob.glob(os.path.join(out_dir, "*.png")) +
                  glob.glob(os.path.join(out_dir, "*.jpg")))
    if not outs:
        print(f"[calibrate] no images in {out_dir}", file=sys.stderr)
        return
    step = max(1, len(outs) // n)
    sample = outs[::step][:n]

    buffers, fracs, scales = [], [], []
    for i, po in enumerate(sample):
        try:
            fo, bw_o, bh_o, r_o = _measure_eardrum(po)
            eside_o = max(bw_o, bh_o)                       # eardrum square side in output px
            if eside_o <= 1:
                continue
            buffers.append((fo / eside_o - 1.0) / 2.0)      # box-mode buffer
            fracs.append((2.0 * r_o) / fo)                  # circle-mode eardrum_frac
            if raw_dir:
                stem = os.path.splitext(os.path.basename(po))[0]
                for cand_stem in (stem, stem[:-4] if stem.endswith("_roi") else stem):
                    hits = glob.glob(os.path.join(raw_dir, cand_stem + ".*"))
                    if hits:
                        fr, bw_r, bh_r, _ = _measure_eardrum(hits[0])
                        eside_r = max(bw_r, bh_r)
                        if eside_r > 1:
                            crop_side_raw = eside_r * (fo / eside_o)
                            scales.append(crop_side_raw / fr)
                        break
            if (i + 1) % 10 == 0 or i + 1 == len(sample):
                print(f"[calibrate] {i+1}/{len(sample)} "
                      f"buffer~{np.median(buffers):.3f} frac~{np.median(fracs):.3f}")
        except Exception as e:
            print(f"[calibrate] skip {os.path.basename(po)}: {e}", file=sys.stderr)

    def stats(a):
        a = np.array(a)
        return (f"median={np.median(a):.3f} mean={a.mean():.3f} "
                f"p25={np.percentile(a,25):.3f} p75={np.percentile(a,75):.3f} n={len(a)}")

    print("\n===== calibration =====")
    if buffers:
        print(f"box buffer      : {stats(buffers)}")
        print(f"  -> --crop-mode box  --buffer {np.median(buffers):.2f}")
    if fracs:
        print(f"circle eardrum_frac: {stats(fracs)}")
        print(f"  -> --crop-mode circle --eardrum-frac {np.median(fracs):.2f}")
    if scales:
        print(f"crop/raw side frac : {stats(scales)}   "
              f"(sanity: crop window covers this fraction of the raw)")
    print("Heavy-tailed spread => outliers/hand-adjusted crops; prefer the median.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="SAM2 ROI zoom-crop (C operator).")
    ap.add_argument("--input",  help="input folder of endoscope images")
    ap.add_argument("--output", help="output folder for cropped images")
    ap.add_argument("--sam2-ckpt", default=os.path.expanduser(
        "~/sam2_checkpoints/sam2.1_hiera_large.pt"))
    ap.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    # crop geometry
    ap.add_argument("--crop-mode", choices=["box", "circle"], default="box",
                    help="box = eardrum bbox + buffer (default); "
                         "circle = min-enclosing-circle / eardrum_frac")
    ap.add_argument("--buffer", type=float, default=0.20,
                    help="box mode: extend each side by this fraction of the squared "
                         "eardrum box (paper value 0.20)")
    ap.add_argument("--eardrum-frac", type=float, default=0.50,
                    help="circle mode: eardrum diameter / crop diameter")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--suffix", default="_roi", help="{stem}{suffix}.png")
    ap.add_argument("--no-circular-mask", action="store_true",
                    help="write a plain square crop instead of a circular ROI")
    ap.add_argument("--feather-px", type=float, default=0.0)
    # black-free fitting
    ap.add_argument("--no-fit-fov", action="store_true",
                    help="disable shrinking the crop to stay inside the image and the "
                         "endoscope FOV (by default the crop is kept black-free)")
    ap.add_argument("--fov-margin", type=float, default=0.98,
                    help="fraction of the detected FOV radius the ROI may reach "
                         "(<1 hides the curved FOV edge)")
    ap.add_argument("--ext", default=".png .jpg .jpeg .tif .tiff")
    ap.add_argument("--manual-json", default=None,
                    help="JSON {stem: {point:[x,y]} | {box:[cx,cy,half]}} of manual "
                         "overrides for images re-cropped by hand")
    ap.add_argument("--save-mask-dir", default=None,
                    help="also save the raw SAM2 eardrum mask (0/255 png)")
    ap.add_argument("--manifest", default=None,
                    help="write a CSV of the crop window used per image")
    ap.add_argument("--skip-existing", action="store_true", help="resume")
    ap.add_argument("--limit", type=int, default=0, help="process first N only (testing)")
    ap.add_argument("--device", default="cuda:0")
    # calibration
    ap.add_argument("--calibrate", metavar="ROI_CROP_DIR", default=None,
                    help="recover buffer/frac from an existing crop set, then exit")
    ap.add_argument("--calibrate-raw", metavar="RAW_DIR", default=None,
                    help="optional: also cross-check crop scale against raw inputs")
    ap.add_argument("--calibrate-n", type=int, default=200)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}  ckpt={args.sam2_ckpt}  cfg={args.sam2_cfg}")
    init_predictor(args.sam2_ckpt, args.sam2_cfg, device)

    if args.calibrate:
        calibrate(args.calibrate, args.calibrate_raw, args.calibrate_n)
        return

    if not args.input or not args.output:
        ap.error("--input and --output are required (unless --calibrate)")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        ap.error("refusing to write into the input folder (never overwrite originals)")

    manual = {}
    if args.manual_json:
        manual = json.load(open(args.manual_json, encoding="utf-8"))
        print(f"[manual] {len(manual)} override entries loaded")

    fit_fov = not args.no_fit_fov
    exts = tuple(e if e.startswith(".") else "." + e for e in args.ext.split())
    files = sorted(p for p in glob.glob(os.path.join(args.input, "*"))
                   if os.path.splitext(p)[1].lower() in exts)
    if args.limit:
        files = files[:args.limit]
    print(f"[scan] {len(files)} input images from {args.input}")
    print(f"[cfg ] mode={args.crop_mode} buffer={args.buffer} frac={args.eardrum_frac} "
          f"size={args.size} suffix='{args.suffix}' circular={not args.no_circular_mask} "
          f"fit_fov={fit_fov} fov_margin={args.fov_margin}")

    os.makedirs(args.output, exist_ok=True)
    mrows = []
    done = skipped = failed = clamped_n = manual_n = 0
    t0 = time.time()
    for i, p in enumerate(files):
        stem = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(args.output, f"{stem}{args.suffix}.png")
        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue
        try:
            img = imread_rgb(p)
            ov = manual.get(stem, {})
            if "box" in ov:                            # full manual crop window
                cx, cy, half = ov["box"]
                out = resize_sq(
                    square_crop(img, cx, cy, half) if args.no_circular_mask
                    else crop_roi_circular(img, cx, cy, half, args.feather_px), args.size)
                mask = None
                meta = dict(cx=cx, cy=cy, half=half, half_req=half, clamped=0, fov_R="")
            else:                                      # SAM (optionally manual point)
                out, mask, meta = make_roi_crop(
                    img, args.crop_mode, args.buffer, args.eardrum_frac, args.size,
                    circular_mask=not args.no_circular_mask, feather_px=args.feather_px,
                    point_xy=tuple(ov["point"]) if "point" in ov else None,
                    fit_fov=fit_fov, fov_margin=args.fov_margin,
                    return_extra=args.save_mask_dir is not None)
            imwrite_rgb(out_path, out)
            if args.save_mask_dir is not None and mask is not None:
                imwrite_gray(os.path.join(args.save_mask_dir, f"{stem}.png"),
                             (mask.astype(np.uint8) * 255))
            is_manual = int(stem in manual)
            if args.manifest:
                mrows.append([stem, round(meta["cx"], 1), round(meta["cy"], 1),
                              round(meta["half"], 1), round(meta["half_req"], 1),
                              meta["clamped"], meta["fov_R"], is_manual])
            done += 1
            clamped_n += int(meta["clamped"])
            manual_n += is_manual
        except Exception as e:
            failed += 1
            print(f"[fail] {os.path.basename(p)}: {e}", file=sys.stderr)
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-6)
            eta = (len(files) - (i + 1)) / max(rate, 1e-6)
            print(f"[{i+1}/{len(files)}] done={done} skip={skipped} fail={failed} "
                  f"clamped={clamped_n} manual={manual_n} | {rate:.2f} img/s | ETA {eta/60:.1f} min")

    if args.manifest and mrows:
        with open(args.manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stem", "cx", "cy", "half", "half_requested",
                        "clamped_for_fov", "fov_radius", "manual"])
            w.writerows(mrows)
        print(f"[manifest] {len(mrows)} rows -> {args.manifest}")

    print(f"\n[done] wrote={done} skipped={skipped} failed={failed} "
          f"clamped={clamped_n} manual={manual_n} -> {args.output}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
