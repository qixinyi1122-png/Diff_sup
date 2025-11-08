import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils.config_sup import CONFIGCLASS
from utils.utils import to_cuda


def get_val_cfg(cfg: CONFIGCLASS, split="val", copy=True):
    if copy:
        from copy import deepcopy
        val_cfg = deepcopy(cfg)
    else:
        val_cfg = cfg

    # === 关键修改：尊重原始 dataset_root，不做任何重写 ===
    # 例如你在外部传入：/root/.../data/diff_no_no/train/lsun_adm
    # test.py 会在此基础上再拼接 /real 或 /adm
    val_cfg.dataset_root_dire = os.path.join(val_cfg.dataset_root_dire, split, val_cfg.exp_name)
    val_cfg.dataset_root_diff = os.path.join(val_cfg.dataset_root_diff, split, val_cfg.exp_name)

    val_cfg.datasets = cfg.datasets_test
    val_cfg.isTrain = False

    # 确定性评估（不带随机增广）
    val_cfg.aug_resize = True
    val_cfg.aug_crop = False
    val_cfg.aug_flip = False
    val_cfg.aug_norm = True
    val_cfg.blur_prob = 0.0
    val_cfg.jpg_prob = 0.0
    val_cfg.rz_interp = ["bilinear"]
    val_cfg.serial_batches = True
    val_cfg.jpg_method = ["pil"]

    # Currently assumes jpg_prob, blur_prob 0 or 1
    if len(val_cfg.blur_sig) == 2:
        b_sig = val_cfg.blur_sig
        val_cfg.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_cfg.jpg_qual) != 1:
        j_qual = val_cfg.jpg_qual
        val_cfg.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_cfg

def validate(model: nn.Module, cfg: CONFIGCLASS):
    """Compute per-sample predictions for one checkpoint and return
    relative paths, true labels, and predicted probabilities.

    Returns a dict with keys:
      - 'relpath': List[str]  (e.g., 'real/img_0001.png')
      - 'y_true': np.ndarray  (shape [N], dtype=float/int)
      - 'y_pred': np.ndarray  (shape [N], dtype=float)
    No aggregate metrics are computed here.
    """
    import os
    from torch.utils.data import DataLoader
    from utils.datasets_sup import create_dataloader

    # 1) Build the original dataloader
    base_loader = create_dataloader(cfg)
    base_ds = base_loader.dataset

    # 2) Local wrapper to expose relative paths (real/xxx.png, gen/xxx.png, ...)
    class WithRelpathDataset(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base
            self.root = getattr(base, "root", cfg.dataset_root)
            self._samples = getattr(base, "samples", None) or getattr(base, "imgs", None)

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            item = self.base[idx]  # (img, label) or (img, label, path)
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                img, label, rel = item[0], item[1], str(item[2])
            else:
                img, label = item
                if self._samples is None:
                    # cannot extract path → return binary
                    return img, label
                abs_path = self._samples[idx][0]
                rel = os.path.relpath(abs_path, self.root).replace(os.sep, "/")
            return img, label, rel

    # 3) Rebuild a local loader with identical batch params, deterministic order
    loader = DataLoader(
        WithRelpathDataset(base_ds),
        batch_size=base_loader.batch_size,
        shuffle=False,
        num_workers=base_loader.num_workers,
        pin_memory=getattr(base_loader, "pin_memory", False),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    relpaths_all: list[str] = []
    y_true_list: list[float] = []
    y_pred_list: list[float] = []

    with torch.no_grad():
        for data in loader:
            if isinstance(data, (list, tuple)) and len(data) == 3:
                imgs, labels, relpaths = data
            else:
                imgs, labels = data
                relpaths = None

            imgs = to_cuda(imgs, device)

            # Forward WITHOUT passing meta/relpath
            try:
                logits = model(imgs)
            except TypeError:
                logits = model(imgs, None)
            probs = logits.sigmoid().flatten()

            y_true_list.extend(labels.flatten().tolist())
            y_pred_list.extend(probs.tolist())

            if relpaths is not None:
                relpaths_all.extend([str(r) for r in relpaths])
            else:
                # If paths unavailable, append placeholders to keep lengths aligned
                relpaths_all.extend(["" for _ in range(len(probs))])

    return {
        "relpath": relpaths_all,
        "y_true": np.array(y_true_list),
        "y_pred": np.array(y_pred_list),
    }

def metrics(y_true, y_pred, thr):
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    print(f"[DEBUG] All samples (true label, pred > {thr}, raw pred):")
    # for t, p in zip(y_true, y_pred):
    #     print(f"  true={int(t)}  pred={(p > 0.5):5}  raw={p:.4f}")
    print(y_true)
    print(y_pred > thr)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thr)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thr)
    acc = accuracy_score(y_true, y_pred > thr)
    ap = average_precision_score(y_true, y_pred)
    # ROC-AUC: threshold-free ranking metric on the ROC curve
    auc = roc_auc_score(y_true, y_pred)

    results = {
        "ACC": acc,
        "AP": ap,          # PR-AUC (Average Precision)
        "AUC": auc,        # ROC-AUC
        "R_ACC": r_acc,
        "F_ACC": f_acc,
    }
    return results
