# dataset.py
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _class_key(dirname: str) -> Tuple[int, str]:
    """
    '00_NORMAL' → (0, 'NORMAL'), '03_TYMPANOSCLEROSIS' → (3, 'TYMPANOSCLEROSIS')
    Names without a numeric prefix sort last as (999, UPPERDIRNAME).
    """
    name = dirname.strip().strip("/").strip("\\")
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return int(parts[0]), parts[1].upper()
    return 999, name.upper()

class FolderClassDataset(Dataset):
    """
    Reads a layout with one subfolder per class under the root folder.

    Example:
    root_dir/
      00_NORMAL/
        img1.png, img2.png, ...
      01_PERFORATION/
      02_RETRACTION/
      03_TYMPANOSCLEROSIS/

    transform:
      - PIL-based transforms such as torchvision.transforms.Compose
      - albumentations is also accepted (auto-detected), but torchvision is
        recommended for this project
    """
    def __init__(
        self,
        root_dir: str,
        transform=None,
        return_path: bool = False
    ):
        self.root_dir    = Path(root_dir)
        self.transform   = transform
        self.return_path = return_path

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root dir not found: {self.root_dir}")

        # 1) Collect and sort class folders (by the 00_, 01_ … prefix)
        class_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]
        if not class_dirs:
            raise RuntimeError(f"No class subdirectories under: {self.root_dir}")

        class_dirs.sort(key=lambda p: _class_key(p.name))
        # Assign labels in sorted order (0..C-1)
        self.classes: List[str] = [ _class_key(p.name)[1] for p in class_dirs ]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 2) Collect image samples
        samples: List[Tuple[Path, int]] = []
        for cdir in class_dirs:
            _, cname = _class_key(cdir.name)
            if cname not in self.class_to_idx:
                # Should not happen in practice; kept as a safeguard
                continue
            label = self.class_to_idx[cname]

            for root, _, files in os.walk(cdir):
                for fn in files:
                    if Path(fn).suffix.lower() in IMG_EXTS:
                        samples.append((Path(root) / fn, label))

        if not samples:
            raise RuntimeError(f"No images found in class subdirectories under: {self.root_dir}")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        # Load with PIL and convert to RGB
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int):
        fpath, label = self.samples[idx]
        img = self._load_image(fpath)

        if self.transform:
            # Auto-detect torchvision vs albumentations
            if isinstance(self.transform, (transforms.Compose, torch.nn.Module)):
                img = self.transform(img)  # torchvision (PIL→Tensor)
            else:
                # albumentations-style: assume a dict is returned
                img = self.transform(image=np.array(img))["image"]

        label = torch.tensor(label, dtype=torch.long)
        if self.return_path:
            return img, label, str(fpath)
        return img, label

    def __repr__(self):
        return (f"FolderClassDataset(n={len(self)}, root='{self.root_dir}', "
                f"classes={self.classes})")
