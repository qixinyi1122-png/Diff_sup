from utils.config_sup import cfg  # isort: split

import csv
import os
import tempfile
import shutil
import copy
from typing import Dict, List, Tuple

import torch
import numpy as np

from utils.eval_sup import get_val_cfg, validate, metrics
from utils.utils import get_network


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return x


def _extract_per_file_truth_and_scores(result_dict: Dict) -> Dict[str, Tuple[int, float]]:
    """
    Map validate()'s new output to {relpath: (y_true, prob)}.
    Expected keys: 'relpath', 'y_true', 'y_pred'.
    """
    if not isinstance(result_dict, dict):
        return {}
    if not all(k in result_dict for k in ("relpath", "y_true", "y_pred")):
        return {}
    rels = result_dict.get("relpath", [])
    yt = result_dict.get("y_true", [])
    yp = result_dict.get("y_pred", [])
    out: Dict[str, Tuple[int, float]] = {}
    n = min(len(rels), len(yt), len(yp))
    for i in range(n):
        k = str(rels[i])
        if not k:
            continue
        try:
            out[k] = (int(yt[i]), float(yp[i]))
        except Exception:
            pass
    return out


# --- Load base cfg for testing ---
cfg = get_val_cfg(cfg, split="test", copy=False)

# === Required: primary model/dataset ===
assert cfg.ckpt_path_dire, "Please specify the path to the dire model checkpoint (cfg.ckpt_path_dire)"
model_name = os.path.basename(cfg.ckpt_path_dire).replace(".pth", "")
dataset_root_dire = cfg.dataset_root_dire
print(f"DIRE.ckpt_path = {cfg.ckpt_path_dire}")
print(f"DIRE.dataset_root = {dataset_root_dire}")

# === Optional: secondary model/dataset for SUP fusion ===
# Provide in your YAML or CLI override:
#   cfg.ckpt_path_b     : path to second checkpoint
#   cfg.dataset_root_b  : root of second dataset (folder structure mirrors the first)
assert cfg.ckpt_path_diff, "Please specify the path to the diff model checkpoint (cfg.ckpt_path_diff)"
ckpt_path_diff = cfg.ckpt_path_diff
dataset_root_diff = cfg.dataset_root_diff
use_and = bool(ckpt_path_diff and dataset_root_diff)
if use_and:
    print(f"DIFF.ckpt_path = {ckpt_path_diff}")
    print(f"DIFF.dataset_root = {dataset_root_diff}")

rows = []
print(f"'{cfg.exp_name}:{model_name}' model testing on...")

# Pairwise binary eval: assume cfg.datasets_test only contains generator sets
gens = list(cfg.datasets_test)

real_dir_dire = os.path.join(dataset_root_dire, "real")
if not os.path.exists(real_dir_dire):
    raise FileNotFoundError(f"Real dataset directory not found at {real_dir_dire}")

save_path = os.path.join(cfg.exp_root, f"test/{cfg.exp_name}_{model_name}{'_SUP' if use_and else ''}.csv")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Preload models
model_dire = get_network(cfg.arch)
state_dict_dire = torch.load(cfg.ckpt_path_dire, map_location="cpu")
model_dire.load_state_dict(state_dict_dire["model"])
model_dire.cuda()
model_dire.eval()

if use_and:
    model_diff = get_network(cfg.arch)
    state_dict_diff = torch.load(ckpt_path_diff, map_location="cpu")
    model_diff.load_state_dict(state_dict_diff["model"])
    model_diff.cuda()
    model_diff.eval()

for i, gen_name in enumerate(gens):
    # --- Construct temp roots for A (and B if enabled) with symlinks ---
    tmp_root_dire = tempfile.mkdtemp(prefix="binary_eval_DIRE_")
    os.symlink(os.path.join(dataset_root_dire, "real"), os.path.join(tmp_root_dire, "real"))
    os.symlink(os.path.join(dataset_root_dire, gen_name), os.path.join(tmp_root_dire, gen_name))

    cfg_dire = copy.deepcopy(cfg)
    cfg_dire.dataset_root = tmp_root_dire
    cfg_dire.datasets = ["real", gen_name]

    # Run validate for A
    test_results_dire = validate(model_dire, cfg_dire)
    map_dire = _extract_per_file_truth_and_scores(test_results_dire)

    print(f"[DIRE] {gen_name}:")
    for k, v in test_results_dire.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {float(v):.5f}")
        else:
            print(f"{k}: {_safe_float(v)}")
    print("-" * 50)

    # Optional: B path
    if use_and:
        tmp_root_diff = tempfile.mkdtemp(prefix="binary_eval_B_")
        os.symlink(os.path.join(dataset_root_diff, "real"), os.path.join(tmp_root_diff, "real"))
        os.symlink(os.path.join(dataset_root_diff, gen_name), os.path.join(tmp_root_diff, gen_name))

        cfg_diff = copy.deepcopy(cfg)
        cfg_diff.dataset_root = tmp_root_diff
        cfg_diff.datasets = ["real", gen_name]

        test_results_diff = validate(model_diff, cfg_diff)
        map_diff = _extract_per_file_truth_and_scores(test_results_diff)

        print(f"[DIFF] {gen_name}:")
        for k, v in test_results_diff.items():
            if isinstance(v, (int, float)):
                print(f"{k}: {float(v):.5f}")
            else:
                print(f"{k}: {_safe_float(v)}")
        print("-" * 50)

        # === SUP fusion using validate() outputs (relpath,y_true,y_pred) ===
        if map_dire and map_diff:
            common_keys = sorted(set(map_dire.keys()) & set(map_diff.keys()))

            y_true_list, pA_list, pB_list = [], [], []
            thr = getattr(cfg, "decision_threshold", 0.5)

            for k in common_keys:
                tA, pA = map_dire[k]
                tB, pB = map_diff[k]
                if tA != tB:
                    continue
                y_true_list.append(tA)
                pA_list.append(pA)
                pB_list.append(pB)

            y_true_arr = np.array(y_true_list)
            pA_arr = np.array(pA_list)
            pB_arr = np.array(pB_list)
            # p_and = np.maximum(pA_arr, pB_arr)  # SUP-equivalent fusion
            # 根据 cfg.SUP 决定融合方式：SUP=True→取最小(AND)；SUP=False→取最大(OR)
            if cfg.SUP:
                print("两个都真才真")
                method = np.minimum
                
            else:
                print("一个真就真")
                method = np.maximum

            p_and = method(pA_arr, pB_arr)
                

            and_dict = metrics(y_true_arr, p_and, thr)

            # Print SUP metrics
            print(f"[SUP] {gen_name}:")
            for k, v in and_dict.items():
                print(f"{k}: {float(v):.5f}")
            print("*" * 50)

            # === Save a row for CSV ===
            if i == 0:
                rows.append(["TestSet", "mode", *list(and_dict.keys())])
            rows.append([gen_name, "SUP", *[and_dict[k] for k in and_dict.keys()]])

            # Also record per-model overall metric if available (optional)
            # Try to include a common metric like 'acc' from A/B for reference
            if "acc" in test_results_dire and isinstance(test_results_dire["acc"], (int, float)):
                rows.append([gen_name, "DIRE_only_acc", float(test_results_dire["acc"])])
            if "acc" in test_results_diff and isinstance(test_results_diff["acc"], (int, float)):
                rows.append([gen_name, "DIFF_only_acc", float(test_results_diff["acc"])])

            # Save per-file SUP predictions (optional detailed log)
            per_file_save = os.path.join(cfg.exp_root, f"test/{cfg.exp_name}_{model_name}_SUP_{gen_name}_perfile.csv")
            os.makedirs(os.path.dirname(per_file_save), exist_ok=True)
            with open(per_file_save, "w", newline="") as fpf:
                wr = csv.writer(fpf)
                wr.writerow(["key", "y_true", "pA", "pB", "predA", "predB", "predSUP"])
                for k in common_keys:
                    (tA, pA) = map_dire[k]
                    (_,  pB) = map_diff[k]
                    pra = 1 if pA > thr else 0
                    prb = 1 if pB > thr else 0
                    prand = 1 if method(pA,pB) > thr else 0
                    wr.writerow([k, tA, pA, pB, pra, prb, prand])
        else:
            # If we cannot get per-file scores, we still emit the A/B metrics separately
            print(f"[WARN] validate() did not return per-file scores for BOTH models; SUP fusion by filename is skipped for '{gen_name}'.")
            if i == 0:
                rows.append(["TestSet"] + list(test_results_dire.keys()))
            rows.append([f"A::{gen_name}"] + [test_results_dire[k] for k in test_results_dire.keys()])
            rows.append([f"B::{gen_name}"] + [test_results_diff[k] for k in test_results_diff.keys()])

        # Clean B temp root
        shutil.rmtree(tmp_root_diff)

    else:
        # Single-model path: keep original behavior
        print(f"{gen_name}:")
        for k, v in test_results_dire.items():
            if isinstance(v, (int, float)):
                print(f"{k}: {float(v):.5f}")
            else:
                print(f"{k}: {_safe_float(v)}")
        print("*" * 50)

        if i == 0:
            rows.append(["TestSet"] + list(test_results_dire.keys()))
        rows.append([gen_name] + list(test_results_dire.values()))

    # Persist CSV after each gen to avoid data loss
    with open(save_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # Clean A temp root
    shutil.rmtree(tmp_root_dire)

print(f"\n✅ All test results saved to {save_path}")
