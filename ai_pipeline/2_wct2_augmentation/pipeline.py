"""
Style-augmentation pipeline for otoscope/endoscope data augmentation.

The pipeline composes three independent operators per sample, controlled by a
combo string drawn per job:

    S  -- WCT2 photorealistic style transfer (Yoo et al., 2019)
    P  -- Albumentations photometric/geometric augmentation (Replay)
    B  -- Ellipse-gradient edge multiply (vignette) with Gaussian-sampled
          strength and gamma; ellipse parameters are precomputed per content
          image and cached on disk for reproducibility.

Pipeline order is fixed: S -> P -> RESIZE to final_size -> B.

Per-class balancing samples sources from `list_of_lists` until each class
reaches `target_per_class`; per-image metadata (operator flags, replay summary,
edge parameters) is written to CSV for full reproducibility.

Reference (WCT2):
    Yoo, J., Uh, Y., Chun, S., Kang, B., & Ha, J. (2019). Photorealistic Style
    Transfer via Wavelet Transforms. ICCV.  (NAVER Corp implementation)
"""

import json
import os
import random
from collections import OrderedDict
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from model import WaveEncoder, WaveDecoder
from utils.core import feature_wct
from utils.io import compute_label_info, open_image


# ============================================================================
# Default Albumentations setup (P operator). Override by passing setup_args.
# ============================================================================
DEFAULT_SETUP_ARGS = {
    "gblur":       (7, 7, 0.75),                  # (kmin, kmax, p)
    "mblur":       (15, 0.25),                    # (ksize, p)
    "rgbshift":    ((15, 15, 15), 0.35),          # ((r,g,b)_lim, p)
    "rbrightness": (0.15, 0.85),                  # (brightness_limit, p)
}


# ============================================================================
# Unicode-path-safe image IO
# ============================================================================
def imread_rgb(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: str, img_rgb: np.ndarray, quality_jpg: int = 95, png_level: int = 3):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    params = []
    if ext in [".jpg", ".jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality_jpg)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_level)]
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(path)


def resize_to(img: np.ndarray, target_hw: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    th, tw = int(target_hw[0]), int(target_hw[1])
    h, w = img.shape[:2]
    if h == th and w == tw:
        return img.astype(np.float32) if is_mask and img.dtype != np.float32 else img
    interp = cv2.INTER_AREA if (th < h or tw < w) else cv2.INTER_CUBIC
    out = cv2.resize(img.astype(np.float32) if is_mask else img, (tw, th), interpolation=interp)
    return out.astype(np.float32) if is_mask else out


# ============================================================================
# WCT2 (S operator)
# ============================================================================
class WCT2:
    """Wavelet-based photorealistic style transfer (Yoo et al., 2019).

    transfer_at: subset of {"encoder", "decoder", "skip"} indicating which
        feature locations to apply whitening-coloring at. In this study only
        "decoder" is used.
    option_unpool: "cat5" (concatenation; default) or "sum".
    """

    def __init__(
        self,
        model_path: str = "./model_checkpoints",
        transfer_at: Set[str] = frozenset({"decoder"}),
        option_unpool: str = "cat5",
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
    ):
        self.transfer_at = set(transfer_at)
        self.device = device
        self.verbose = verbose

        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)

        enc_path = os.path.join(model_path, f"wave_encoder_{option_unpool}_l4.pth")
        dec_path = os.path.join(model_path, f"wave_decoder_{option_unpool}_l4.pth")
        assert os.path.exists(enc_path), f"Missing: {enc_path}"
        assert os.path.exists(dec_path), f"Missing: {dec_path}"

        try:
            enc_sd = torch.load(enc_path, map_location=self.device, weights_only=True)
        except TypeError:
            enc_sd = torch.load(enc_path, map_location=self.device)
        try:
            dec_sd = torch.load(dec_path, map_location=self.device, weights_only=True)
        except TypeError:
            dec_sd = torch.load(dec_path, map_location=self.device)

        self.encoder.load_state_dict(enc_sd)
        self.decoder.load_state_dict(dec_sd)

        self.wct2_enc_level = [1, 2, 3, 4]
        self.wct2_dec_level = [1, 2, 3, 4]
        self.wct2_skip_level = ["pool1", "pool2", "pool3"]

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def _get_style_features(self, style):
        skips, feats = {}, {"encoder": {}, "decoder": {}}
        x = style
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if "encoder" in self.transfer_at and level in self.wct2_enc_level:
                feats["encoder"][level] = x
        if "encoder" not in self.transfer_at:
            feats["decoder"][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if "decoder" in self.transfer_at:
                feats["decoder"][level - 1] = x
        return feats, skips

    @torch.no_grad()
    def transfer(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        content_segment: Optional[torch.Tensor] = None,
        style_segment: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        if content_segment is None:
            content_segment = torch.zeros(1, content.shape[-2], content.shape[-1]).long().to(self.device)
        if style_segment is None:
            style_segment = torch.zeros(1, style.shape[-2], style.shape[-1]).long().to(self.device)

        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        style_feats, style_skips = self._get_style_features(style)

        content_feat = content
        content_skips: Dict = {}
        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if "encoder" in self.transfer_at and level in self.wct2_enc_level:
                content_feat = feature_wct(
                    content_feat, style_feats["encoder"][level],
                    content_segment, style_segment, label_set, label_indicator,
                    alpha=alpha, device=self.device,
                )
        if "skip" in self.transfer_at:
            for skip_level in self.wct2_skip_level:
                for comp in [0, 1, 2]:
                    content_skips[skip_level][comp] = feature_wct(
                        content_skips[skip_level][comp], style_skips[skip_level][comp],
                        content_segment, style_segment, label_set, label_indicator,
                        alpha=alpha, device=self.device,
                    )
        for level in [4, 3, 2, 1]:
            if "decoder" in self.transfer_at and level in style_feats["decoder"] and level in self.wct2_dec_level:
                content_feat = feature_wct(
                    content_feat, style_feats["decoder"][level],
                    content_segment, style_segment, label_set, label_indicator,
                    alpha=alpha, device=self.device,
                )
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat.clamp(0, 1)


class StyleTransferEngine:
    """Thin path-based wrapper around WCT2 that returns an RGB uint8 image."""

    def __init__(
        self,
        mode: str = "wct2",
        device: str = "cuda",
        model_path: str = "./model_checkpoints",
        option_unpool: str = "cat5",
        transfer_at: Set[str] = frozenset({"decoder"}),
        image_size: int = 512,
        alpha: float = 1.0,
    ):
        self.mode = mode
        use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.model_path = model_path
        self.option_unpool = option_unpool
        self.transfer_at = set(transfer_at)
        self.image_size = int(image_size)
        self.alpha = float(alpha)

        self.initialized = (self.mode == "wct2")
        if self.initialized:
            self.wct2 = WCT2(
                model_path=self.model_path,
                transfer_at=self.transfer_at,
                option_unpool=self.option_unpool,
                device=self.device,
            )

    @torch.no_grad()
    def stylize_paths(self, content_path: str, style_path: Optional[str]) -> np.ndarray:
        if self.mode != "wct2" or not self.initialized or not style_path:
            return imread_rgb(content_path)
        content = open_image(content_path, self.image_size).to(self.device)
        style = open_image(style_path, self.image_size).to(self.device)
        out = self.wct2.transfer(content, style, None, None, alpha=self.alpha)
        out_np = (out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return out_np

    def meta_config(self) -> Dict[str, Union[str, int, float]]:
        return {
            "engine": self.mode,
            "wct2_model_path": self.model_path if self.mode == "wct2" else "",
            "wct2_option_unpool": self.option_unpool if self.mode == "wct2" else "",
            "wct2_transfer_at": "+".join(sorted(self.transfer_at)) if self.mode == "wct2" else "",
            "wct2_image_size": self.image_size if self.mode == "wct2" else "",
            "wct2_alpha": self.alpha if self.mode == "wct2" else "",
        }


# ============================================================================
# Ellipse-gradient edge mask (B operator), with on-disk + RAM caching
# ============================================================================
def render_ellipse_mask_from_params(
    h: int, w: int,
    cx: float, cy: float, MA: float, ma: float, angle_deg: float,
    fade_ratio: float = 0.2,
) -> np.ndarray:
    a, b = MA / 2.0, ma / 2.0
    theta = np.deg2rad(-angle_deg)
    y, x = np.ogrid[:h, :w]
    dx = x - cx
    dy = y - cy
    x_rot = dx * np.cos(theta) - dy * np.sin(theta)
    y_rot = dx * np.sin(theta) + dy * np.cos(theta)
    a = max(a, 1e-6); b = max(b, 1e-6)
    norm = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
    inner = 1.0 - float(fade_ratio)
    mask = np.zeros_like(norm, dtype=np.float32)
    mask[norm <= inner] = 1.0
    trans = (norm > inner) & (norm <= 1.0)
    if fade_ratio > 0:
        mask[trans] = (1.0 - (norm[trans] - inner) / float(fade_ratio))
    return mask


def compute_ellipse_params_once(
    img_rgb: np.ndarray,
    thresh1: int = 30,
    morph_kernel_size: int = 15,
    fade_ratio: float = 0.2,
    min_contour_points: int = 5,
    min_area_px: int = 50,
) -> Tuple[dict, np.ndarray]:
    """Fit an ellipse to the largest bright connected component and return
    its parameters plus the rendered fade mask. Falls back to a centered
    near-full-image ellipse when fitting fails."""
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask_bg = cv2.threshold(gray, thresh1, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask_clean = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean)

    ellipse_ok = False
    cx, cy = w / 2.0, h / 2.0
    MA = ma = min(w, h) * 0.9
    angle = 0.0

    if num_labels > 1:
        max_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        if stats[max_idx, cv2.CC_STAT_AREA] >= max(min_area_px, 1):
            region = (labels == max_idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                if len(cnt) >= int(min_contour_points) and cv2.contourArea(cnt) >= max(min_area_px, 1):
                    try:
                        ell = cv2.fitEllipse(cnt)
                        (cx, cy), (MA, ma), angle = ell
                        ellipse_ok = True
                    except Exception:
                        ellipse_ok = False

    params = {
        "h": int(h), "w": int(w),
        "cx": float(cx), "cy": float(cy),
        "MA": float(MA), "ma": float(ma),
        "angle": float(angle),
        "fade_ratio": float(fade_ratio),
        "thresh1": int(thresh1),
        "morph_kernel_size": int(morph_kernel_size),
        "ellipse_ok": int(ellipse_ok),
    }
    mask0 = render_ellipse_mask_from_params(h, w, cx, cy, MA, ma, angle, fade_ratio).astype(np.float32)
    return params, mask0


def precompute_edge_cache(
    list_of_lists: List[List[str]],
    cache_json_path: str,
    save_masks_dir: Optional[str] = None,
    thresh1: int = 30,
    morph_kernel_size: int = 15,
    fade_ratio: float = 0.2,
    min_contour_points: int = 5,
    min_area_px: int = 50,
) -> dict:
    """Pre-compute ellipse parameters for every image once and store them in
    JSON. Optionally also dump the rendered masks as .npy for fastest reuse."""
    all_paths = [p for sub in list_of_lists for p in sub]
    cache: Dict[str, dict] = {}

    masks_dir = Path(save_masks_dir) if save_masks_dir else None
    if masks_dir:
        masks_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(all_paths, desc="Precomputing edge params"):
        try:
            img = imread_rgb(p)
            params, mask0 = compute_ellipse_params_once(
                img, thresh1=thresh1, morph_kernel_size=morph_kernel_size, fade_ratio=fade_ratio,
                min_contour_points=min_contour_points, min_area_px=min_area_px,
            )
            cache[p] = params
            if masks_dir:
                key = sha1(p.encode("utf-8")).hexdigest()
                np.save(masks_dir / f"{key}.npy", mask0)
        except Exception as e:
            print(f"[PRECOMP ERR] {p}: {e}")

    Path(cache_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_json_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved edge cache: {cache_json_path} ({len(cache)} items)")
    if masks_dir:
        print(f"[INFO] Saved masks to: {masks_dir}")
    return cache


class EdgeMaskRAMCache:
    """LRU-style RAM cache that lazily loads ellipse masks from .npy or
    re-renders them from JSON parameters. Falls back to on-the-fly fitting
    when neither is available."""

    def __init__(
        self,
        edge_cache_params: Optional[Dict[str, dict]] = None,
        edge_masks_dir: Optional[str] = None,
        preload: bool = False,
        max_items: Optional[int] = None,
        fallback_thresh1: int = 30,
        fallback_morph_kernel: int = 15,
        fallback_fade_ratio: float = 0.20,
        min_contour_points: int = 5,
        min_area_px: int = 50,
    ):
        self.params = edge_cache_params or {}
        self.masks_dir = Path(edge_masks_dir) if edge_masks_dir else None
        self.max_items = max_items
        self.cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.fallback_thresh1 = int(fallback_thresh1)
        self.fallback_morph_kernel = int(fallback_morph_kernel)
        self.fallback_fade_ratio = float(fallback_fade_ratio)
        self.min_contour_points = int(min_contour_points)
        self.min_area_px = int(min_area_px)
        if preload:
            self._preload_all()

    def _touch(self, key: str):
        if self.max_items is None:
            return
        try:
            self.cache.move_to_end(key, last=True)
        except KeyError:
            pass
        while self.max_items is not None and len(self.cache) > self.max_items:
            self.cache.popitem(last=False)

    def _preload_all(self):
        for path, par in tqdm(self.params.items(), desc="Preloading edge masks into RAM"):
            try:
                mask = None
                if self.masks_dir is not None:
                    key = sha1(path.encode("utf-8")).hexdigest()
                    npy_path = self.masks_dir / f"{key}.npy"
                    if npy_path.exists():
                        mask = np.load(npy_path)
                if mask is None:
                    h0, w0 = int(par["h"]), int(par["w"])
                    mask = render_ellipse_mask_from_params(
                        h0, w0, par["cx"], par["cy"], par["MA"], par["ma"], par["angle"],
                        par.get("fade_ratio", self.fallback_fade_ratio),
                    ).astype(np.float32)
                self.cache[path] = mask
                self._touch(path)
            except Exception as e:
                print(f"[RAM PRELOAD WARN] {path}: {e}")

    def get_mask_and_params(self, content_path: str) -> Tuple[np.ndarray, Optional[dict]]:
        if content_path in self.cache:
            self._touch(content_path)
            return self.cache[content_path], self.params.get(content_path, None)

        mask = None
        par = self.params.get(content_path, None)

        if self.masks_dir is not None:
            try:
                key = sha1(content_path.encode("utf-8")).hexdigest()
                npy_path = self.masks_dir / f"{key}.npy"
                if npy_path.exists():
                    mask = np.load(npy_path)
            except Exception:
                mask = None

        if mask is None and par is not None:
            try:
                h0, w0 = int(par["h"]), int(par["w"])
                mask = render_ellipse_mask_from_params(
                    h0, w0, par["cx"], par["cy"], par["MA"], par["ma"], par["angle"],
                    par.get("fade_ratio", self.fallback_fade_ratio),
                ).astype(np.float32)
            except Exception:
                mask = None

        if mask is None:
            img0 = imread_rgb(content_path)
            par, mask = compute_ellipse_params_once(
                img0,
                thresh1=self.fallback_thresh1,
                morph_kernel_size=self.fallback_morph_kernel,
                fade_ratio=self.fallback_fade_ratio,
                min_contour_points=self.min_contour_points,
                min_area_px=self.min_area_px,
            )
            self.params[content_path] = par

        self.cache[content_path] = mask
        self._touch(content_path)
        return mask, par


# ============================================================================
# Albumentations Replay pipeline (P operator)
# ============================================================================
def _ensure_odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else (k + 1)


def build_replay_pipeline(
    setup_args: Optional[dict] = None,
    prob_hflip: float = 0.5,
    prob_vflip: float = 0.0,
    prob_rotate: float = 0.9,
    rotate_limit_deg: int = 180,
):
    """Return an `A.ReplayCompose` covering the spatial/photometric ops in P.

    setup_args keys (all optional):
        gblur:       (kmin, kmax, p)       -- GaussianBlur kernel range (odd)
        mblur:       (ksize, p)            -- MotionBlur kernel (odd)
        noise:       (var_min, var_max, mean, p)
        cjitter:     (brightness_limit, contrast_limit, p)
        equalize:    (mode, by_channels, p)
        rgbshift:    ((r_lim, g_lim, b_lim), p)
        rbrightness: (brightness_limit, p) -- brightness-only RandomBrightness
    No RandomCrop is applied; final size is enforced by RESIZE downstream.
    """
    sa = setup_args or {}

    g_tuple = sa.get("gblur")
    if g_tuple is not None:
        gmin, gmax, p_gblur = g_tuple
        gmin, gmax = _ensure_odd(gmin), _ensure_odd(gmax)
        if gmax < gmin:
            gmin, gmax = gmax, gmin
    else:
        p_gblur, gmin, gmax = 0.0, 0, 0

    m_tuple = sa.get("mblur")
    if m_tuple is not None:
        msize, p_mblur = m_tuple
        msize = _ensure_odd(msize)
    else:
        p_mblur, msize = 0.0, 0

    n_tuple = sa.get("noise")
    if n_tuple is not None:
        nmin, nmax, nmean, p_noise = n_tuple
        var_limit = (max(1, int(nmin)), max(1, int(nmax)))
        noise_mean = float(nmean)
    else:
        p_noise, var_limit, noise_mean = 0.0, (0, 0), 0.0

    cj_tuple = sa.get("cjitter")
    if cj_tuple is not None:
        cj_bright_lim, cj_cont_lim, p_cj = cj_tuple
    else:
        p_cj, cj_bright_lim, cj_cont_lim = 0.0, 0.0, 0.0

    eq_tuple = sa.get("equalize")
    rs_tuple = sa.get("rgbshift")
    rb_tuple = sa.get("rbrightness")

    tfs = []

    if prob_hflip > 0:
        tfs.append(A.HorizontalFlip(p=float(prob_hflip)))
    if prob_vflip > 0:
        tfs.append(A.VerticalFlip(p=float(prob_vflip)))
    if prob_rotate > 0 and rotate_limit_deg > 0:
        tfs.append(
            A.Affine(
                scale=None, translate_percent=None, translate_px=None,
                rotate=(-int(rotate_limit_deg), int(rotate_limit_deg)),
                shear=None,
                interpolation=cv2.INTER_LINEAR,
                cval=(0, 0, 0), cval_mask=0,
                mode=cv2.BORDER_CONSTANT,
                fit_output=False,
                p=float(prob_rotate),
            )
        )

    if eq_tuple is not None:
        mode, by_ch, p_eq = eq_tuple
        tfs.append(A.Equalize(mode=str(mode), by_channels=bool(by_ch), p=float(p_eq)))

    if p_gblur > 0 and gmax > 0:
        tfs.append(A.GaussianBlur(blur_limit=(int(gmin), int(gmax)), sigma_limit=0, p=float(p_gblur)))

    if p_mblur > 0 and msize > 0:
        tfs.append(A.MotionBlur(blur_limit=int(msize), p=float(p_mblur)))

    if rs_tuple is not None:
        (r_lim, g_lim, b_lim), p_rs = rs_tuple
        tfs.append(A.RGBShift(
            r_shift_limit=int(r_lim), g_shift_limit=int(g_lim), b_shift_limit=int(b_lim),
            p=float(p_rs),
        ))

    if rb_tuple is not None:
        bright_lim, p_rb = rb_tuple
        tfs.append(A.RandomBrightnessContrast(
            brightness_limit=float(bright_lim), contrast_limit=0.0, p=float(p_rb),
        ))

    if p_noise > 0 and var_limit[1] > 0:
        tfs.append(A.GaussNoise(var_limit=tuple(var_limit), mean=noise_mean, p=float(p_noise)))
    if p_cj > 0 and (cj_bright_lim > 0 or cj_cont_lim > 0):
        tfs.append(A.RandomBrightnessContrast(
            brightness_limit=float(cj_bright_lim), contrast_limit=float(cj_cont_lim), p=float(p_cj),
        ))

    return A.ReplayCompose(tfs, p=1.0)


def apply_replay_to_mask(replay_params: dict, mask_0to1: np.ndarray) -> np.ndarray:
    """Replay the same geometric transformations recorded for an image onto
    the corresponding edge mask so the mask remains spatially aligned."""
    H, W = mask_0to1.shape
    dummy_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    transformed = A.ReplayCompose.replay(replay_params, image=dummy_rgb, mask=mask_0to1)
    out_mask = transformed["mask"]
    return np.clip(out_mask.astype(np.float32), 0, 1)


def _to_jsonable(x):
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        if x.size > 32:
            return {"ndarray": True, "shape": list(x.shape), "dtype": str(x.dtype)}
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            try:
                out[str(k)] = _to_jsonable(v)
            except Exception:
                out[str(k)] = f"<unserializable:{type(v).__name__}>"
        return out
    return f"<obj:{type(x).__name__}>"


def summarize_replay_transforms(replay: Optional[dict]) -> dict:
    if replay is None:
        return {"sequence": []}
    seq = []
    for t in replay.get("transforms", []):
        name = t.get("__class_fullname__", "unknown").split(".")[-1]
        applied = bool(t.get("applied", False))
        params_raw = {k: v for k, v in t.items() if k not in ["__class_fullname__", "applied", "lambda", "replay"]}
        seq.append({"name": name, "applied": applied, "params": _to_jsonable(params_raw)})
    return {"sequence": seq}


def extract_quick_fields_from_replay(replay: Optional[dict]) -> dict:
    out = {"p_hflip": 0, "p_vflip": 0, "p_rot_deg": "", "p_gblur_ksize": "", "p_mblur_ksize": ""}
    if replay is None:
        return out
    for t in replay.get("transforms", []):
        name = t.get("__class_fullname__", "").split(".")[-1]
        if not bool(t.get("applied", False)):
            continue
        if name == "HorizontalFlip":
            out["p_hflip"] = 1
        elif name == "VerticalFlip":
            out["p_vflip"] = 1
        elif name == "Affine":
            rot = t.get("rotate") or (t.get("params", {}) if isinstance(t.get("params", {}), dict) else {}).get("rotate")
            if rot is not None and not (isinstance(rot, (list, tuple)) and len(rot) == 2):
                out["p_rot_deg"] = float(rot)
        elif name == "GaussianBlur":
            k = t.get("kernel") or t.get("ksize") or t.get("blur_limit")
            out["p_gblur_ksize"] = int(k) if isinstance(k, (int, float)) else (str(k) if k is not None else "")
        elif name == "MotionBlur":
            k = t.get("blur_limit") or t.get("kernel") or t.get("ksize")
            out["p_mblur_ksize"] = int(k) if isinstance(k, (int, float)) else (str(k) if k is not None else "")
    return out


# ============================================================================
# Job sampling and per-class balancing
# ============================================================================
def collect_style_pool(styles_dir: Optional[str]) -> Optional[List[str]]:
    if not styles_dir:
        return None
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    pool = []
    for ext in exts:
        pool.extend([str(p) for p in Path(styles_dir).rglob(ext)])
    pool.sort()
    return pool or None


def build_balanced_jobs(
    list_of_lists: List[List[str]],
    class_names: List[str],
    target_per_class: Union[int, str] = "max",
    seed: int = 1337,
) -> List[Tuple[int, str, str, int]]:
    """Build a flat job list balanced so every class produces
    `target_per_class` samples. Sources are cycled when undersampling and
    re-drawn at random when oversampling."""
    rng = random.Random(seed)
    per_class_paths = [list(paths) for paths in list_of_lists]
    counts = [len(p) for p in per_class_paths]
    if any(c == 0 for c in counts):
        raise ValueError("Empty class encountered; need at least one image per class.")
    target = max(counts) if (isinstance(target_per_class, str) and target_per_class.lower() == "max") else int(target_per_class)
    next_variant = {p: 0 for paths in per_class_paths for p in paths}
    jobs: List[Tuple[int, str, str, int]] = []
    for ci, paths in enumerate(per_class_paths):
        cname = class_names[ci]
        n_src = len(paths)
        for idx in range(target):
            src = rng.choice(paths) if n_src < target else paths[idx % n_src]
            v_idx = next_variant[src]
            next_variant[src] += 1
            jobs.append((ci, cname, src, v_idx))
    return jobs


def normalize_op_probs(op_probs: Optional[Dict[str, float]]) -> Dict[str, float]:
    if op_probs is None:
        return {"s": 0.5, "p": 0.5, "b": 0.5}
    out = {}
    for k in ["s", "p", "b"]:
        v = float(op_probs.get(k, 0.0))
        out[k] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    return out


def sample_combo_by_ops(
    allowed_ops: str = "spb",
    op_probs: Optional[Dict[str, float]] = None,
    rng: Optional[random.Random] = None,
    enforce_nonempty: bool = True,
) -> str:
    """Independently sample whether each of S, P, B is applied to this job.
    If all three Bernoullis are 0 and `enforce_nonempty` is True, fall back
    to a probability-weighted single-op pick."""
    if rng is None:
        rng = random.Random()
    probs = normalize_op_probs(op_probs)
    picks = {op: (rng.random() < probs[op]) if op in allowed_ops else False for op in ["s", "p", "b"]}
    if enforce_nonempty and not any(picks.values()):
        cand = [op for op in ["s", "p", "b"] if op in allowed_ops]
        w = [probs[op] for op in cand]
        s = sum(w)
        w = [1.0 / len(cand)] * len(cand) if s <= 0 else [wi / s for wi in w]
        choose = cand[int(np.random.choice(len(cand), p=np.array(w, dtype=np.float64)))]
        picks = {op: (op == choose) for op in ["s", "p", "b"]}
    combo = "".join(sorted([op for op, v in picks.items() if v]))
    return combo if combo else "s"


def sample_trunc_gauss(
    rng: random.Random, lo: float, hi: float,
    mu: Optional[float] = None, sigma: Optional[float] = None, max_tries: int = 50,
) -> float:
    """Sample from N(mu, sigma) truncated to [lo, hi]; defaults give a
    midpoint mu and a sigma covering ~99.7% of the interval."""
    lo, hi = float(lo), float(hi)
    if mu is None:
        mu = (lo + hi) / 2.0
    if sigma is None:
        sigma = max(1e-8, (hi - lo) / 6.0)
    for _ in range(max_tries):
        v = rng.gauss(mu, sigma)
        if lo <= v <= hi:
            return float(v)
    return float(min(max(rng.gauss(mu, sigma), lo), hi))


# ============================================================================
# Per-image processing: S -> P -> Resize -> B (Edge Multiply)
# ============================================================================
def process_one_image_with_combo(
    content_path: str,
    class_idx: int,
    class_name: str,
    out_root: Path,
    engine: StyleTransferEngine,
    style_pool: Optional[List[str]],
    rng: random.Random,
    final_size: Tuple[int, int],
    jpeg_ext: str,
    quality_jpg: int,
    png_level: int,
    variant_idx: int,
    combo: str,
    setup_args: Optional[dict] = None,
    edge_ram_cache: Optional[EdgeMaskRAMCache] = None,
    edge_cache: Optional[Dict[str, dict]] = None,
    edge_masks_dir: Optional[str] = None,
    edge_thresh1: int = 30,
    edge_morph_kernel: int = 15,
    edge_fade_ratio: float = 0.20,
    edge_strength_range: Tuple[float, float] = (0.6, 1.0),
    edge_strength_mu: Optional[float] = None,
    edge_strength_sigma: Optional[float] = None,
    edge_gamma_range: Tuple[float, float] = (0.8, 1.2),
    edge_gamma_mu: Optional[float] = None,
    edge_gamma_sigma: Optional[float] = None,
) -> Tuple[str, dict]:
    apply_style = ("s" in combo)
    apply_proc = ("p" in combo)
    apply_border = ("b" in combo)
    tag = "".join(sorted(combo))
    th, tw = int(final_size[0]), int(final_size[1])

    # 0) Edge mask at original size + params
    par_used = None
    if edge_ram_cache is not None:
        mask0, par_used = edge_ram_cache.get_mask_and_params(content_path)
    else:
        mask0 = None
        if edge_masks_dir is not None:
            try:
                key = sha1(content_path.encode("utf-8")).hexdigest()
                npy_path = Path(edge_masks_dir) / f"{key}.npy"
                if npy_path.exists():
                    mask0 = np.load(npy_path)
            except Exception:
                mask0 = None
        if mask0 is None:
            if edge_cache is not None and content_path in edge_cache:
                par = edge_cache[content_path]
                par_used = par
                h0, w0 = int(par["h"]), int(par["w"])
                mask0 = render_ellipse_mask_from_params(
                    h0, w0, par["cx"], par["cy"], par["MA"], par["ma"], par["angle"],
                    par.get("fade_ratio", edge_fade_ratio),
                ).astype(np.float32)
            else:
                img0 = imread_rgb(content_path)
                par_used, mask0 = compute_ellipse_params_once(
                    img0, thresh1=edge_thresh1, morph_kernel_size=edge_morph_kernel, fade_ratio=edge_fade_ratio,
                )

    # 1) S -- style transfer (or load original)
    if apply_style:
        style_path = random.choice(style_pool) if style_pool else ""
        styled = engine.stylize_paths(content_path, style_path)
    else:
        style_path = ""
        styled = imread_rgb(content_path)

    mask_resized = cv2.resize(mask0, (styled.shape[1], styled.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 2) P -- Albumentations replay
    replay = None
    if apply_proc:
        pipe = build_replay_pipeline(setup_args=setup_args)
        transformed = pipe(image=styled)
        aug = transformed["image"]
        replay = transformed["replay"]
        mask_edge = apply_replay_to_mask(replay, mask_resized)
    else:
        aug = styled
        mask_edge = mask_resized

    # 3) Final size by RESIZE (no crop/pad)
    aug = resize_to(aug, (th, tw), is_mask=False)
    mask_edge = resize_to(mask_edge, (th, tw), is_mask=True)

    # 4) B -- edge multiply with Gaussian-sampled strength & gamma
    eg = es = None
    if apply_border:
        m = np.clip(mask_edge, 0, 1).astype(np.float32)
        eg = sample_trunc_gauss(rng, edge_gamma_range[0], edge_gamma_range[1],
                                mu=edge_gamma_mu, sigma=edge_gamma_sigma)
        es = sample_trunc_gauss(rng, edge_strength_range[0], edge_strength_range[1],
                                mu=edge_strength_mu, sigma=edge_strength_sigma)
        if abs(eg - 1.0) > 1e-6:
            m = np.clip(m, 1e-6, 1.0) ** float(eg)
        mix = (1.0 - float(es)) + float(es) * m
        out_rgb = (aug.astype(np.float32) * mix[..., None]).astype(np.uint8)
    else:
        out_rgb = aug

    # 5) Save under {out_root}/{class_idx:02d}_{class_name}/
    stem = Path(content_path).stem
    subdir = out_root / f"{class_idx:02d}_{class_name}"
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"{stem}_c{tag}_v{variant_idx:04d}{jpeg_ext}"
    imwrite_rgb(str(out_path), out_rgb, quality_jpg=quality_jpg, png_level=png_level)

    # 6) Metadata for CSV
    par_meta = {
        "edge_h0": "", "edge_w0": "", "edge_cx0": "", "edge_cy0": "",
        "edge_MA0": "", "edge_ma0": "", "edge_angle0": "",
        "edge_fade_ratio": "", "edge_thresh1": "", "edge_morph_kernel": "", "edge_ellipse_ok": "",
    }
    if par_used is not None:
        par_meta = {
            "edge_h0": par_used.get("h", ""),
            "edge_w0": par_used.get("w", ""),
            "edge_cx0": par_used.get("cx", ""),
            "edge_cy0": par_used.get("cy", ""),
            "edge_MA0": par_used.get("MA", ""),
            "edge_ma0": par_used.get("ma", ""),
            "edge_angle0": par_used.get("angle", ""),
            "edge_fade_ratio": par_used.get("fade_ratio", ""),
            "edge_thresh1": par_used.get("thresh1", ""),
            "edge_morph_kernel": par_used.get("morph_kernel_size", ""),
            "edge_ellipse_ok": par_used.get("ellipse_ok", ""),
        }

    meta = {
        "src_path": content_path,
        "out_path": str(out_path),
        "class_idx": class_idx,
        "class_name": class_name,
        "variant_idx": variant_idx,
        "combo_tag": tag,
        "apply_style": int(apply_style),
        "apply_proc": int(apply_proc),
        "apply_border": int(apply_border),
        "style_path": style_path,
        "final_h": th, "final_w": tw,
        "jpeg_ext": jpeg_ext,
        "jpeg_quality": quality_jpg if jpeg_ext.lower() in [".jpg", ".jpeg"] else "",
        "png_level": png_level if jpeg_ext.lower() == ".png" else "",
        "transforms": json.dumps(summarize_replay_transforms(replay), ensure_ascii=False),
        **extract_quick_fields_from_replay(replay),
        **engine.meta_config(),
        **par_meta,
        "edge_ram_used": int(edge_ram_cache is not None),
        "edge_strength": es if apply_border else "",
        "edge_gamma": eg if apply_border else "",
        "edge_strength_range": str(edge_strength_range),
        "edge_gamma_range": str(edge_gamma_range),
        "edge_strength_mu": "" if edge_strength_mu is None else edge_strength_mu,
        "edge_strength_sigma": "" if edge_strength_sigma is None else edge_strength_sigma,
        "edge_gamma_mu": "" if edge_gamma_mu is None else edge_gamma_mu,
        "edge_gamma_sigma": "" if edge_gamma_sigma is None else edge_gamma_sigma,
    }
    return str(out_path), meta


# ============================================================================
# Top-level runner: one dataset version per call
# ============================================================================
CSV_HEADER = [
    "src_path", "out_path", "class_idx", "class_name", "variant_idx",
    "combo_tag", "apply_style", "apply_proc", "apply_border",
    "style_path", "final_h", "final_w",
    "jpeg_ext", "jpeg_quality", "png_level", "transforms",
    "p_hflip", "p_vflip", "p_rot_deg", "p_gblur_ksize", "p_mblur_ksize",
    "engine", "wct2_model_path", "wct2_option_unpool", "wct2_transfer_at", "wct2_image_size", "wct2_alpha",
    "edge_h0", "edge_w0", "edge_cx0", "edge_cy0", "edge_MA0", "edge_ma0", "edge_angle0",
    "edge_fade_ratio", "edge_thresh1", "edge_morph_kernel", "edge_ellipse_ok",
    "edge_ram_used",
    "edge_strength", "edge_gamma", "edge_strength_range", "edge_gamma_range",
    "edge_strength_mu", "edge_strength_sigma", "edge_gamma_mu", "edge_gamma_sigma",
    "allowed_ops", "op_probs",
]


def run_augmentation_random_ops(
    list_of_lists: List[List[str]],
    class_names: Optional[List[str]],
    out_dir: str,
    # operator sampling
    allowed_ops: str = "spb",
    op_probs: Optional[Dict[str, float]] = None,
    # balancing
    target_per_class: Union[int, str] = "max",
    # style pool (for S)
    styles_dir: Optional[str] = None,
    # WCT2
    engine_mode: str = "wct2",
    wct2_model_path: str = "./model_checkpoints",
    wct2_option_unpool: str = "cat5",
    wct2_transfer_at: Set[str] = frozenset({"decoder"}),
    wct2_image_size: int = 512,
    wct2_alpha: float = 1.0,
    # output
    final_h: int = 256,
    final_w: int = 256,
    seed: int = 1337,
    jpeg_ext: str = ".jpg",
    csv_path: Optional[str] = None,
    csv_flush_every: int = 500,
    quality_jpg: int = 95,
    png_level: int = 3,
    # Albumentations (P)
    setup_args: Optional[dict] = None,
    prob_hflip: float = 0.5,
    prob_vflip: float = 0.0,
    prob_rotate: float = 0.9,
    rotate_limit_deg: int = 180,
    # Edge caches (B)
    edge_cache_json: Optional[str] = None,
    edge_masks_dir: Optional[str] = None,
    use_edge_ram: bool = True,
    edge_ram_preload: bool = False,
    edge_ram_max_items: Optional[int] = None,
    # Edge effect distributions
    edge_strength_range: Tuple[float, float] = (0.6, 1.0),
    edge_strength_mu: Optional[float] = None,
    edge_strength_sigma: Optional[float] = None,
    edge_gamma_range: Tuple[float, float] = (0.8, 1.2),
    edge_gamma_mu: Optional[float] = None,
    edge_gamma_sigma: Optional[float] = None,
) -> pd.DataFrame:
    """Run one balanced augmentation pass over `list_of_lists` and return a
    DataFrame of per-image metadata. Output files are written under
    `out_dir/{class_idx:02d}_{class_name}/` and CSV metadata is streamed to
    `csv_path` (default: `<out_dir>/augmentation_metadata.csv`)."""

    if class_names is None:
        class_names = [f"class{i}" for i in range(len(list_of_lists))]
    filtered_lists, filtered_names, removed = [], [], []
    for name, paths in zip(class_names, list_of_lists):
        if len(paths) > 0:
            filtered_lists.append(paths)
            filtered_names.append(name)
        else:
            removed.append(name)
    if removed:
        print(f"[INFO] Skipping empty classes: {removed}")
    list_of_lists, class_names = filtered_lists, filtered_names
    if len(list_of_lists) == 0:
        raise ValueError("All classes are empty.")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = StyleTransferEngine(
        mode=engine_mode, device=device,
        model_path=wct2_model_path, option_unpool=wct2_option_unpool,
        transfer_at=set(wct2_transfer_at), image_size=wct2_image_size, alpha=wct2_alpha,
    )
    style_pool = collect_style_pool(styles_dir)

    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_jobs = build_balanced_jobs(list_of_lists, class_names, target_per_class=target_per_class, seed=seed)
    total = len(base_jobs)

    next_variant_for_combo: Dict[Tuple[str, str], int] = {}

    edge_cache = None
    if edge_cache_json and Path(edge_cache_json).exists():
        with open(edge_cache_json, "r", encoding="utf-8") as f:
            edge_cache = json.load(f)
    elif edge_cache_json:
        print(f"[WARN] edge_cache_json not found: {edge_cache_json} (will fit on-the-fly for missing)")

    edge_ram_cache = None
    if use_edge_ram:
        edge_ram_cache = EdgeMaskRAMCache(
            edge_cache_params=edge_cache,
            edge_masks_dir=edge_masks_dir,
            preload=edge_ram_preload,
            max_items=edge_ram_max_items,
        )

    csv_out = Path(csv_path) if csv_path else (out_root / "augmentation_metadata.csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = csv_out.exists() and csv_out.stat().st_size > 0
    buffer: List[str] = []

    with open(csv_out, "a", encoding="utf-8") as f:
        if not wrote_header:
            f.write(",".join(CSV_HEADER) + "\n")

        pbar = tqdm(total=total, desc="Augmenting (operator-prob)")
        for (ci, cname, src, _) in base_jobs:
            combo = sample_combo_by_ops(allowed_ops=allowed_ops, op_probs=op_probs, rng=rng, enforce_nonempty=True)
            key = (src, combo)
            v_idx = next_variant_for_combo.get(key, 0)
            next_variant_for_combo[key] = v_idx + 1

            try:
                _, meta = process_one_image_with_combo(
                    content_path=src,
                    class_idx=ci,
                    class_name=cname,
                    out_root=out_root,
                    engine=engine,
                    style_pool=style_pool,
                    rng=rng,
                    final_size=(final_h, final_w),
                    jpeg_ext=jpeg_ext,
                    quality_jpg=quality_jpg,
                    png_level=png_level,
                    variant_idx=v_idx,
                    combo=combo,
                    setup_args=setup_args or DEFAULT_SETUP_ARGS,
                    edge_ram_cache=edge_ram_cache,
                    edge_cache=edge_cache,
                    edge_masks_dir=edge_masks_dir,
                    edge_strength_range=edge_strength_range,
                    edge_strength_mu=edge_strength_mu,
                    edge_strength_sigma=edge_strength_sigma,
                    edge_gamma_range=edge_gamma_range,
                    edge_gamma_mu=edge_gamma_mu,
                    edge_gamma_sigma=edge_gamma_sigma,
                )

                row = [
                    meta["src_path"], meta["out_path"], meta["class_idx"], meta["class_name"], meta["variant_idx"],
                    meta["combo_tag"], meta["apply_style"], meta["apply_proc"], meta["apply_border"],
                    meta["style_path"], meta["final_h"], meta["final_w"],
                    meta["jpeg_ext"], meta["jpeg_quality"], meta["png_level"], meta["transforms"],
                    meta.get("p_hflip", ""), meta.get("p_vflip", ""), meta.get("p_rot_deg", ""),
                    meta.get("p_gblur_ksize", ""), meta.get("p_mblur_ksize", ""),
                    meta["engine"], meta["wct2_model_path"], meta["wct2_option_unpool"], meta["wct2_transfer_at"], meta["wct2_image_size"], meta["wct2_alpha"],
                    meta.get("edge_h0", ""), meta.get("edge_w0", ""), meta.get("edge_cx0", ""), meta.get("edge_cy0", ""),
                    meta.get("edge_MA0", ""), meta.get("edge_ma0", ""), meta.get("edge_angle0", ""),
                    meta.get("edge_fade_ratio", ""), meta.get("edge_thresh1", ""), meta.get("edge_morph_kernel", ""), meta.get("edge_ellipse_ok", ""),
                    meta.get("edge_ram_used", 0),
                    meta.get("edge_strength", ""), meta.get("edge_gamma", ""),
                    meta.get("edge_strength_range", ""), meta.get("edge_gamma_range", ""),
                    meta.get("edge_strength_mu", ""), meta.get("edge_strength_sigma", ""),
                    meta.get("edge_gamma_mu", ""), meta.get("edge_gamma_sigma", ""),
                    allowed_ops, json.dumps(normalize_op_probs(op_probs), ensure_ascii=False),
                ]
                safe = []
                for s in row:
                    s = "" if s is None else str(s)
                    if ("," in s) or ("\n" in s) or ("\"" in s):
                        s = "\"" + s.replace("\"", "\"\"") + "\""
                    safe.append(s)
                buffer.append(",".join(safe))

                if len(buffer) >= csv_flush_every:
                    f.write("\n".join(buffer) + "\n")
                    f.flush()
                    buffer.clear()
            except Exception as e:
                print(f"[ERROR] {src} ({combo}): {e}")
            finally:
                pbar.update(1)
        pbar.close()

        if buffer:
            f.write("\n".join(buffer) + "\n")
            f.flush()

    return pd.read_csv(csv_out)
