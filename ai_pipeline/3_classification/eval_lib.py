# eval_lib.py
#
# Lightweight evaluation helpers for the notebook: load a checkpoint, run it
# over a validation folder, and report top-1 / top-2 accuracy plus a confusion
# matrix. This covers the "basic metrics" path. The full per-subset evaluation
# and the Figure-5 confusion-matrix panels are produced separately.

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import ResNet
import configs
from dataset import FolderClassDataset

CLASS_NAMES = ["NORMAL", "PERFORATION", "RETRACTION", "TYMPANOSCLEROSIS"]


def build_transform(image_size=(256, 256)):
    """Evaluation transform. Matches the training transform in train.py:
    Resize (bilinear) -> ToTensor -> Normalize(0.5, 0.5)."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


def _strip_module_prefix(state_dict):
    """Drop a leading 'module.' (added when a model is wrapped in DataParallel)."""
    if any(k.startswith("module.") for k in state_dict):
        return {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in state_dict.items()}
    return state_dict


def load_resnet(ckpt_path: str, num_class: int = 4, dropout: float = 0.3,
                device: Optional[torch.device] = None) -> torch.nn.Module:
    """Load a ResNet-18 from either a full training checkpoint (a dict with
    'model_state_dict') or a slimmed weights-only state_dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(ckpt_path, map_location=device)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    state = _strip_module_prefix(state)

    model = ResNet(configs.resnet18_config, num_class, dropout_rate=dropout)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def make_loader(val_dir: str, image_size: Tuple[int, int] = (256, 256),
                batch_size: int = 64, num_workers: int = 4) -> DataLoader:
    ds = FolderClassDataset(root_dir=val_dir, transform=build_transform(image_size))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def predict(model, loader, device: Optional[torch.device] = None):
    """Return (y_true, y_pred, probs) as numpy arrays."""
    if device is None:
        device = next(model.parameters()).device
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        p = torch.softmax(logits, dim=1)
        ys.append(y.numpy())
        preds.append(p.argmax(dim=1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return (np.concatenate(ys), np.concatenate(preds), np.concatenate(probs))


def topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 2) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    hits = (topk == y_true[:, None]).any(axis=1)
    return float(hits.mean())


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                  class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Print top-1 / top-2 accuracy and a per-class report; return a summary dict."""
    from sklearn.metrics import classification_report, confusion_matrix

    class_names = class_names or CLASS_NAMES
    labels = list(range(len(class_names)))
    acc1 = float((y_true == y_pred).mean())
    acc2 = topk_accuracy(y_true, probs, k=2)
    print(f"Top-1 accuracy: {acc1*100:.2f}%")
    print(f"Top-2 accuracy: {acc2*100:.2f}%")
    print()
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=class_names, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"acc1": acc1, "acc2": acc2, "confusion_matrix": cm}


def plot_confusion(cm: np.ndarray, class_names: Optional[List[str]] = None,
                   normalize: bool = True, title: str = "Confusion matrix"):
    """Render a single confusion matrix (row-normalised by default)."""
    import matplotlib.pyplot as plt

    class_names = class_names or CLASS_NAMES
    mat = cm.astype(float)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if normalize else mat.max())
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fmt = "{:.2f}" if normalize else "{:.0f}"
    thresh = (mat.max() + mat.min()) / 2.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
