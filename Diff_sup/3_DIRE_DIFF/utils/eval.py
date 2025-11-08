import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils.config import CONFIGCLASS
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
    val_cfg.dataset_root = os.path.join(val_cfg.dataset_root, split, val_cfg.exp_name)

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
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    from utils.datasets import create_dataloader

    data_loader = create_dataloader(cfg)
    print(f"DEBUG: validating datasets = {cfg.datasets}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        y_true, y_pred = [], []
        for data in data_loader:
            img, label, meta = data if len(data) == 3 else (*data, None)
            in_tens = to_cuda(img, device)
            meta = to_cuda(meta, device)
            predict = model(in_tens, meta).sigmoid()
            y_pred.extend(predict.flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print("[DEBUG] All samples (true label, pred > 0.5, raw pred):")
    # for t, p in zip(y_true, y_pred):
    #     print(f"  true={int(t)}  pred={(p > 0.5):5}  raw={p:.4f}")
    print(y_true)
    print(y_pred > 0.5)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
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
