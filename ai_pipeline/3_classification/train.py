#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from model import ResNet
import configs
from dataset import FolderClassDataset
from functions import train, evaluate
import sys, json, shlex
from pprint import pformat


# ─────────────────────────────────────────────────────
# Default data root. Override per run with --base-for-paper, or set the
# OTOSCOPE_DATA_ROOT environment variable, or edit the fallback below.
# ─────────────────────────────────────────────────────
BASE_FOR_PAPER = os.environ.get(
    "OTOSCOPE_DATA_ROOT",
    "/path/to/__dataset/__for_paper",
)

DEFAULT_VAL_DIRS = {
    "raw":        os.path.join(BASE_FOR_PAPER, "raw_test"),
    "raw_filter": os.path.join(BASE_FOR_PAPER, "raw_test_filter"),
    "roi":        os.path.join(BASE_FOR_PAPER, "roi_test"),
    "roi_filter": os.path.join(BASE_FOR_PAPER, "roi_test_filter"),
    "otoscope":   os.path.join(BASE_FOR_PAPER, "otoscope"),
}


# ─────────────────────────────────────────────────────
# Argparse
# ─────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Unified trainer for RAW/ROI datasets with identical defaults."
    )
    # Required
    p.add_argument("--dataset", required=True,
                   help="Training dataset (folder path or dataset name, e.g. dataset_pb_raw / dataset_pb_crop)")
    p.add_argument("--trial", type=int, required=True,
                   help="Trial number (integer)")

    # Optional (defaults match the original script)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num-class", type=int, default=4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--save-delay", type=int, default=1)
    p.add_argument("--image-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)

    # Validation-set selection (auto / manual)
    p.add_argument("--val-mode", choices=["auto", "raw", "roi"], default="auto",
                   help="auto: use the ROI sets (roi_test, roi_test_filter) when the dataset name/path contains 'crop', otherwise the RAW sets (raw_test, raw_test_filter)")
    p.add_argument("--val-raw-dir",        default=DEFAULT_VAL_DIRS["raw"])
    p.add_argument("--val-raw-filter-dir", default=DEFAULT_VAL_DIRS["raw_filter"])
    p.add_argument("--val-roi-dir",        default=DEFAULT_VAL_DIRS["roi"])
    p.add_argument("--val-roi-filter-dir", default=DEFAULT_VAL_DIRS["roi_filter"])
    p.add_argument("--val-otoscope-dir",   default=DEFAULT_VAL_DIRS["otoscope"])

    # Validation set used as the checkpoint / early-stopping criterion
    p.add_argument("--ckpt-val-name", default="Test",
                   help="Name of the validation set used for best-model / early stopping (default: 'Test')")

    # Paths / checkpoints
    p.add_argument("--base-for-paper", default=BASE_FOR_PAPER,
                   help="Base root prepended when only a dataset name is given")
    p.add_argument("--ckpt-root", default="./__checkpoints")
    p.add_argument("--log-root", default="./__logs")
    p.add_argument("--resume", default=None, help="Checkpoint path (.pt)")

    # Misc
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-persistent", action="store_true",
                   help="Disable DataLoader persistent_workers")

    return p.parse_args()


# ─────────────────────────────────────────────────────
# Helper: resolve dataset name / path
# ─────────────────────────────────────────────────────
def resolve_train_dir(dataset_arg: str, base_for_paper: str):
    """
    If dataset_arg is a directory, use it as-is.
    Otherwise join base_for_paper/dataset_arg.
    Also returns a display DATASET_NAME.
    """
    if os.path.isdir(dataset_arg):
        train_dir = dataset_arg
        dataset_name = os.path.basename(os.path.normpath(dataset_arg))
    else:
        train_dir = os.path.join(base_for_paper, dataset_arg)
        dataset_name = dataset_arg
    return train_dir, dataset_name


def choose_val_dirs(dataset_name_or_path: str,
                    val_mode: str,
                    raw_dir: str, raw_filter_dir: str,
                    roi_dir: str, roi_filter_dir: str):
    """
    Return (main_test, filter_test).
    When val_mode is auto: use the ROI pair if the name contains 'crop', else the RAW pair.
    When val_mode is raw/roi: force the corresponding pair.
    """
    if val_mode == "raw":
        return raw_dir, raw_filter_dir
    if val_mode == "roi":
        return roi_dir, roi_filter_dir
    # auto
    key = (dataset_name_or_path or "").lower()
    if "crop" in key or "roi" in key:
        return roi_dir, roi_filter_dir
    return raw_dir, raw_filter_dir

def _format_val(v):
    # Convert types that are hard to JSON-serialize (Path, etc.) to strings
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)

def args_to_dict(args):
    # Safely convert nested Namespaces to a dict
    import argparse
    if isinstance(args, argparse.Namespace):
        return {k: args_to_dict(v) for k, v in vars(args).items()}
    if isinstance(args, (list, tuple)):
        return [args_to_dict(x) for x in args]
    if isinstance(args, dict):
        return {k: args_to_dict(v) for k, v in args.items()}
    return _format_val(args)

# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Basic settings
    EPOCHS        = args.epochs
    BATCH_SIZE    = args.batch_size
    LEARNING_RATE = args.lr
    NUM_CLASS     = args.num_class
    K             = min(2, NUM_CLASS)
    PATIENCE      = args.patience
    SAVE_DELAY    = args.save_delay
    IMAGE_SIZE    = tuple(args.image_size)
    WEIGHT_DECAY  = args.weight_decay
    DROPOUT       = args.dropout

    # Training dataset
    TRAIN_DIR, DATASET_NAME = resolve_train_dir(args.dataset, args.base_for_paper)

    # Validation sets (one main & filter pair + always otoscope)
    main_val_dir, filter_val_dir = choose_val_dirs(
        DATASET_NAME,
        args.val_mode,
        args.val_raw_dir, args.val_raw_filter_dir,
        args.val_roi_dir, args.val_roi_filter_dir
    )

    VAL_DIRS = {
        "Test":        main_val_dir,          # raw_test for RAW, roi_test for ROI
        "Test_filter": filter_val_dir,        # raw_test_filter for RAW, roi_test_filter for ROI
        "Otoscope":    args.val_otoscope_dir  # always included
    }

    # Run name / paths
    TRAIN_NAME   = f"{DATASET_NAME}_{args.trial:02d}"
    TRAIN_DETAIL = f"Dataset name: {DATASET_NAME}\nTrial: {args.trial:02d}\nNumClass: {NUM_CLASS}"
    title_line   = "=" * max(40, len(TRAIN_DETAIL.splitlines()[-1]))
    print(title_line, "", TRAIN_DETAIL, "", title_line, sep="\n")

    CKPT_DIR   = os.path.join(args.ckpt_root, TRAIN_NAME)
    LOG_DIR    = os.path.join(args.log_root, TRAIN_NAME)
    JSON_PATH  = os.path.join(LOG_DIR, "metrics.json")
    RESUME_PTH = args.resume

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    # Logging
    start_time = datetime.now()
    with open(os.path.join(LOG_DIR, "train.log"), "a", encoding="utf-8") as f:
        f.write(f"\n===== New Training Run =====\n")
        f.write(f"Start Time: {start_time}\n")
        f.write(f"Train Dir: {TRAIN_DIR}\n")
        f.write(f"Val Dirs: {VAL_DIRS}\n")
        f.write(f"Checkpoint Dir: {CKPT_DIR}\n")
        f.write(f"Resume From: {RESUME_PTH}\n\n")
        f.write(f"Train Detail:\n{TRAIN_DETAIL}\n")
        # --- Args section (one per line) ---
        f.write("[Args]\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"  - {k}: {pformat(_format_val(v), width=120, compact=True)}\n")

    # Initialize metrics.json
    with open(JSON_PATH, "w", encoding="utf-8") as jf:
        json.dump([], jf)

    # ─────────────────────────────────────────────────
    # DataLoader
    # ─────────────────────────────────────────────────
    common_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    # Train
    train_ds = FolderClassDataset(root_dir=TRAIN_DIR, transform=common_transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(not args.no_persistent and args.num_workers > 0)
    )

    # Val
    val_loaders = {}
    for name, path in VAL_DIRS.items():
        ds = FolderClassDataset(root_dir=path, transform=common_transform)
        loader = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=False  # validation is usually left non-persistent
        )
        val_loaders[name] = loader

    print(f"Train samples: {len(train_loader.dataset)}")
    for name, loader in val_loaders.items():
        print(f"{name} samples: {len(loader.dataset)}")

    # ─────────────────────────────────────────────────
    # Model / Loss / Optim
    # ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(configs.resnet18_config, NUM_CLASS, dropout_rate=DROPOUT)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ─────────────────────────────────────────────────
    # Resume
    # ─────────────────────────────────────────────────
    start_epoch      = 0
    best_target_loss = float('inf')
    early_stop_ctr   = 0
    metrics_history  = []

    if RESUME_PTH and os.path.isfile(RESUME_PTH):
        ckpt = torch.load(RESUME_PTH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt.get('optimizer_state_dict', optimizer.state_dict()))
        start_epoch    = ckpt.get('epoch', 0) + 1
        metrics_history= ckpt.get('metrics_history', [])
        prev_targets = [m.get(f"{args.ckpt_val_name.lower()}_loss", None) for m in metrics_history]
        prev_targets = [x for x in prev_targets if x is not None]
        if prev_targets:
            best_target_loss = min(prev_targets)
        print(f"Resuming from epoch {start_epoch}, best {args.ckpt_val_name} loss = {best_target_loss:.4f}")

    # ─────────────────────────────────────────────────
    # Train Loop
    # ─────────────────────────────────────────────────
    for epoch in tqdm(range(start_epoch, EPOCHS), desc="Epochs"):
        # Train
        tr_loss, tr_acc1, tr_acck = train(model, train_loader, optimizer, criterion, device, k=K)

        # Validate
        current = {}
        for name, loader in val_loaders.items():
            v_loss, v_acc1, v_acck = evaluate(model, loader, criterion, device, k=K)
            current[name] = (v_loss, v_acc1, v_acck)

        # Record JSON
        entry = {
            "epoch":        epoch + 1,
            "train_loss":   tr_loss,
            "train_acc1":   tr_acc1,
            "train_acck":   tr_acck,
        }
        for name, (v_loss, v_acc1, v_acck) in current.items():
            key_base = name.lower()
            entry[f"{key_base}_loss"] = v_loss
            entry[f"{key_base}_acc1"] = v_acc1
            entry[f"{key_base}_acck"] = v_acck

        metrics_history.append(entry)
        with open(JSON_PATH, "w", encoding="utf-8") as jf:
            json.dump(metrics_history, jf, indent=2)

        # Checkpoint / early-stopping criterion set
        if args.ckpt_val_name not in current:
            raise KeyError(f"--ckpt-val-name '{args.ckpt_val_name}' not found in VAL sets: {list(current.keys())}")
        target_loss = current[args.ckpt_val_name][0]

        if target_loss < best_target_loss:
            best_target_loss = target_loss
            early_stop_ctr = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics_history':      metrics_history
            }, os.path.join(CKPT_DIR, "_ResNet_model_best.pt"))
        else:
            early_stop_ctr += 1

        # Periodic save
        if epoch % SAVE_DELAY == 0:
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics_history':      metrics_history
            }, os.path.join(CKPT_DIR, f"ResNet_{epoch:03d}.pt"))

        # Early Stopping
        if early_stop_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Console output
        line = (f"Epoch {epoch+1:03d} | "
                f"Train L {tr_loss:.3f}, Acc1 {tr_acc1*100:5.2f}%, Acc{K} {tr_acck*100:5.2f}%  ||  ")
        for name, (vl, va1, vak) in current.items():
            line += (f"{name[:12]:>12} L {vl:.3f}, A1 {va1*100:5.2f}%, A{K} {vak*100:5.2f}%  ")
        print(line)

    # Finish
    end_time = datetime.now()
    with open(os.path.join(LOG_DIR, "train.log"), "a", encoding="utf-8") as f:
        f.write(f"\nTraining finished at {end_time}, duration {end_time - start_time}\n")
    print(f"{TRAIN_NAME} === Training done. Best {args.ckpt_val_name} loss: {best_target_loss:.4f}")


if __name__ == "__main__":
    main()
