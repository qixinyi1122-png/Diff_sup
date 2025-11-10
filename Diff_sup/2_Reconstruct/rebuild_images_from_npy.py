#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch as th
import cv2
from typing import Dict, Tuple

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def to_u8_vis(t: th.Tensor, min_val: float = None, max_val: float = None) -> th.Tensor:
    """
    和 recon_metrics.py 中一模一样的可视化逻辑：
    - 输入: (N, C, H, W) 的 torch.Tensor
    - 若给定 min_val/max_val: 按固定区间线性映射到 [0,255]
    - 否则: 对每个样本自适应缩放
    返回: (N, H, W, C) 的 uint8 张量
    """
    t = t.clone().detach()

    # 固定放缩：使用统一区间
    if (min_val is not None) and (max_val is not None):
        denom = max_val - min_val
        if abs(denom) < 1e-8:
            denom = 1e-8
        x01 = (t - min_val) / denom
        x255 = (x01.clamp(0, 1) * 255.0).round().to(th.uint8)
        return x255.permute(0, 2, 3, 1).contiguous()

    # 自适应放缩：逐样本归一化
    n, c, h, w = t.shape
    t_min = t.amin(dim=(1, 2, 3), keepdim=True)
    t_max = t.amax(dim=(1, 2, 3), keepdim=True)
    has_neg = (t_min < 0).any()

    if has_neg:
        amax = th.maximum(-t_min, t_max)  # 每个样本的对称上界
        amax = th.clamp(amax, min=1e-8)
        x01 = (t / (2 * amax)) + 0.5
    else:
        denom = th.clamp(t_max, min=1e-8)
        x01 = t / denom

    x255 = (x01.clamp(0, 1) * 255.0).round().to(th.uint8)
    return x255.permute(0, 2, 3, 1).contiguous()


def build_basename2ext_map(images_dir: str) -> Dict[str, str]:
    """
    从原始图像根目录构建一个:
        basename -> extension
    的字典。
    例如: 'ILSVRC2012_val_00000001' -> '.JPEG'
    """
    mapping: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(images_dir):
        for fn in filenames:
            lower = fn.lower()
            if lower.endswith(IMAGE_EXTS):
                base, ext = os.path.splitext(fn)
                mapping[base] = ext  # 例如 ".JPEG"
    return mapping


def prepare_vis_ranges(
    vis_mode: str,
    abs_d1: bool,
    abs_d2: bool,
    abs_final_diff: bool,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    和 recon_metrics.py 中 main 函数里的区间逻辑一致：
    - D1/D2:
        abs=True  -> [0,2]
        abs=False -> [-2,2]
    - DiffRecon:
        若 D1/D2 都 abs=True:
            base = 2.0
        否则:
            base = 4.0
        然后根据 abs_final_diff 决定是否变成 [0,base] 或 [-base,base]
    """
    if vis_mode != "fixed":
        return (None, None), (None, None), (None, None)

    vis_min_d1, vis_max_d1 = (0.0, 2.0) if abs_d1 else (-2.0, 2.0)
    vis_min_d2, vis_max_d2 = (0.0, 2.0) if abs_d2 else (-2.0, 2.0)

    base = 2.0 if (abs_d1 and abs_d2) else 4.0
    if abs_final_diff:
        vis_min_diff, vis_max_diff = 0.0, base
    else:
        vis_min_diff, vis_max_diff = -base, base

    return (vis_min_d1, vis_max_d1), (vis_min_d2, vis_max_d2), (vis_min_diff, vis_max_diff)


def save_from_npy_root(
    npy_root: str,
    out_root: str,
    vis_min: float,
    vis_max: float,
    suffix: str,
    basename2ext: Dict[str, str],
    default_ext: str,
):
    """
    通用函数：从某个 npy 根目录（例如 dire 保存的 D1.npy）
    递归找到所有 *{suffix}.npy 文件，重建图像并保存到 out_root，
    保留相对目录结构。

    - suffix: "_D1.npy" / "_D2.npy" / "_DiffRecon.npy"
    """
    for dirpath, _, filenames in os.walk(npy_root):
        for fn in filenames:
            if not fn.endswith(suffix):
                continue

            npy_path = os.path.join(dirpath, fn)
            arr = np.load(npy_path)  # 形状应为 (C,H,W) 或 (1,C,H,W)，float32
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]  # (1,C,H,W)
            elif arr.ndim == 4:
                # 已经带 batch 维，按第一张处理
                pass
            else:
                print(f"[WARN] 非预期形状 {arr.shape} 跳过: {npy_path}")
                continue

            t = th.from_numpy(arr).float()
            img_u8 = to_u8_vis(t, vis_min, vis_max)[0]  # (H,W,C) uint8

            # 相对目录结构：保证 "sdv1/<class>/..." 这层不丢
            rel_dir = os.path.relpath(dirpath, npy_root)
            out_dir = os.path.join(out_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            # 从文件名中恢复原始 basename
            base = fn[: -len(suffix)]  # 去掉 "_D1.npy" / "_D2.npy" / "_DiffRecon.npy"

            # 根据原始 image_dir 的映射找到扩展名，否则用默认
            ext = basename2ext.get(base, default_ext)
            out_name = base + ext
            out_path = os.path.join(out_dir, out_name)

            # BGR 保存
            img_bgr = cv2.cvtColor(img_u8.numpy(), cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, img_bgr)


def str2bool(v):
    """
    安全解析命令行布尔参数：
    支持: True/False, true/false, 1/0, y/n 等
    """
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(
        description="从 all_Reconstruct 中的 .npy 重新生成误差图像（不跑模型）"
    )
    parser.add_argument("--images_dir", type=str, required=True,
                        help="原始图像根目录（用来恢复扩展名，如 .JPEG/.jpg/.png）")
    parser.add_argument("--dire_npy_root", type=str, required=True,
                        help="D1 对应的 npy 根目录（如 all_Reconstruct/.../dire/...）")
    parser.add_argument("--diff_npy_root", type=str, required=True,
                        help="DiffRecon 对应的 npy 根目录（如 all_Reconstruct/.../diffrecon/...）")

    parser.add_argument("--out_d1", type=str, required=True,
                        help="D1 图像输出根目录（新建 Reconstruct 里的目录）")
    parser.add_argument("--out_diff", type=str, required=True,
                        help="DiffRecon 图像输出根目录")

    parser.add_argument("--vis_mode", type=str, default="fixed",
                        choices=["adaptive", "fixed"],
                        help="和原脚本一致：'adaptive' 或 'fixed'")
    parser.add_argument("--abs_d1", type=str2bool, default=True)
    parser.add_argument("--abs_final_diff", type=str2bool, default=False)

    parser.add_argument("--default_ext", type=str, default=".png",
                        help="如果在原始 images_dir 找不到 basename，就用这个扩展名")

    args = parser.parse_args()

    print("[INFO] 构建 basename->ext 映射（扫描原始图像目录）...")
    basename2ext = build_basename2ext_map(args.images_dir)
    print(f"[INFO] 共记录 {len(basename2ext)} 个文件名到扩展名映射")

    (vis_min_d1, vis_max_d1), _, (vis_min_diff, vis_max_diff) = \
        prepare_vis_ranges(args.vis_mode, args.abs_d1, False, args.abs_final_diff)

    print(f"[INFO] vis ranges: D1=({vis_min_d1},{vis_max_d1}), Diff=({vis_min_diff},{vis_max_diff})")

    print("[STEP] 重建 D1 图像...")
    save_from_npy_root(
        npy_root=args.dire_npy_root,
        out_root=args.out_d1,
        vis_min=vis_min_d1,
        vis_max=vis_max_d1,
        suffix="_D1.npy",
        basename2ext=basename2ext,
        default_ext=args.default_ext,
    )

    print("[STEP] 重建 DiffRecon 图像...")
    save_from_npy_root(
        npy_root=args.diff_npy_root,
        out_root=args.out_diff,
        vis_min=vis_min_diff,
        vis_max=vis_max_diff,
        suffix="_DiffRecon.npy",
        basename2ext=basename2ext,
        default_ext=args.default_ext,
    )

    print("[DONE] 全部从 npy 重建完毕。")


if __name__ == "__main__":
    main()
