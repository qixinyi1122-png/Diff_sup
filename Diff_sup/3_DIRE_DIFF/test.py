from utils.config import cfg  # isort: split

import csv
import os

import torch
import tempfile, shutil
import copy

from utils.eval import get_val_cfg, validate
from utils.utils import get_network

cfg = get_val_cfg(cfg, split="test", copy=False)

assert cfg.ckpt_path, "Please specify the path to the model checkpoint"
print(f"ckpt_path is {cfg.ckpt_path}")
model_name = os.path.basename(cfg.ckpt_path).replace(".pth", "")
dataset_root = cfg.dataset_root  # keep it
print(f"dataset_root is {dataset_root}")
rows = []
print(f"'{cfg.exp_name}:{model_name}' model testing on...")

# Pairwise binary eval: assume cfg.datasets_test only contains generator sets
gens = list(cfg.datasets_test)

real_dir = os.path.join(dataset_root, "real")
if not os.path.exists(real_dir):
    raise FileNotFoundError(f"Real dataset directory not found at {real_dir}")

save_path = os.path.join(cfg.exp_root, f"test/{cfg.exp_name}_{model_name}.csv")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

for i, gen_name in enumerate(gens):
    cfg_i = copy.deepcopy(cfg)

    # === 构造临时目录，软链接 real 和 gen ===
    tmp_root = tempfile.mkdtemp(prefix="binary_eval_")
    os.symlink(os.path.join(dataset_root, "real"),
               os.path.join(tmp_root, "real"))
    os.symlink(os.path.join(dataset_root, gen_name),
               os.path.join(tmp_root, gen_name))

    cfg_i.dataset_root = tmp_root
    cfg_i.datasets = ["real", gen_name]

    model = get_network(cfg_i.arch)
    state_dict = torch.load(cfg_i.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.cuda()
    model.eval()

    test_results = validate(model, cfg_i)
    print(f"{gen_name}:")
    for k, v in test_results.items():
        print(f"{k}: {v:.5f}")
    print("*" * 50)


    # === 保存结果 ===
    if i == 0:
        rows.append(["TestSet"] + list(test_results.keys()))
    rows.append([gen_name] + list(test_results.values()))

    # 每次更新 CSV，防止中断丢数据
    with open(save_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # === 清理临时目录 ===
    shutil.rmtree(tmp_root)


print(f"\n✅ All test results saved to {save_path}")

