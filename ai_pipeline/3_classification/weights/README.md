# Best-epoch weights

Eight ResNet-18 checkpoints, one per (augmentation version × content variant)
run, at the best epoch reported in the paper. These let you evaluate a trained
model without retraining.

| File | Configuration | Best epoch | Oto accuracy |
|------|---------------|-----------:|-------------:|
| `dataset_p_f2_raw_best.pt`    | Endoscope            |  9 | 0.6556 |
| `dataset_p_f2_crop_best.pt`   | Endoscope + C        | 17 | 0.6556 |
| `dataset_pb_f2_raw_best.pt`   | Endoscope + EB       | 11 | 0.8000 |
| `dataset_pb_f2_crop_best.pt`  | Endoscope + C, EB    | 11 | 0.8556 |
| `dataset_sp_f2_raw_best.pt`   | Endoscope + S        | 23 | 0.8444 |
| `dataset_sp_f2_crop_best.pt`  | Endoscope + C, S     | 37 | 0.9111 |
| `dataset_spb_f2_raw_best.pt`  | Endoscope + S, EB    | 23 | 0.8778 |
| `dataset_spb_f2_crop_best.pt` | Endoscope + C, S, EB | 37 | **0.9556** |

The last row (`dataset_spb_f2_crop_best.pt`, S + C + EB) is the paper's
highest-accuracy model at 95.5% — the one deployed as `model.ptl` in the phone
and Glass apps. The "best epoch" is the epoch chosen by evaluating every epoch
on the test set (the paper's reported epoch), not the training-time
validation-loss checkpoint — the two differ, and these files carry the
reported epoch.

## Format

Each file is a slimmed checkpoint (~45 MB) — the model weights only, without the
optimizer state that the training-time checkpoints carry. It is a dict:

```python
{
  "model_state_dict": ...,     # ResNet-18, 4-class head
  "epoch_one_based": int,      # the best epoch above
  "dataset": str,              # e.g. "dataset_sp_f2_crop"
  "run": str,                  # source run folder incl. trial
  "arch": "resnet18",
  "num_class": 4,
  "source": str,               # original checkpoint path
}
```

## Loading

Use the helper in `../eval_lib.py`, which accepts both these slimmed files and
full training checkpoints:

```python
from eval_lib import load_resnet, make_loader, predict, basic_metrics, CLASS_NAMES

model  = load_resnet("weights/dataset_sp_f2_crop_best.pt", num_class=4, dropout=0.3)
loader = make_loader("/path/to/roi_test", num_workers=4)
y_true, y_pred, probs = predict(model, loader)
basic_metrics(y_true, y_pred, probs, CLASS_NAMES)
```

Class order is fixed by the folder prefixes: `0=NORMAL`, `1=PERFORATION`,
`2=RETRACTION`, `3=TYMPANOSCLEROSIS`.
