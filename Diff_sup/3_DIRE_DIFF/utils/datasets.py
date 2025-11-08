import os
from io import BytesIO
from random import choice, random

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import ImageFolder

from utils.config import CONFIGCLASS

ImageFile.LOAD_TRUNCATED_IMAGES = True


def dataset_folder(root: str, cfg: CONFIGCLASS):
    if cfg.mode == "binary":
        return binary_dataset(root, cfg)
    if cfg.mode == "filename":
        return FileNameDataset(root, cfg)
    raise ValueError("cfg.mode needs to be binary or filename.")


def binary_dataset(root: str, cfg: CONFIGCLASS):
    # 调试打印，确认传进来的 root
    print("DEBUG: ImageFolder root =", root, " datasets =", getattr(cfg, "datasets", None))

    identity_transform = transforms.Lambda(lambda img: img)

    if cfg.isTrain or cfg.aug_resize:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, cfg))
    else:
        rz_func = identity_transform

    if cfg.isTrain:
        crop_func = transforms.RandomCrop(cfg.cropSize)
    else:
        # Eval/Test must be deterministic and fixed-size
        crop_func = transforms.CenterCrop(cfg.cropSize)

    if cfg.isTrain and cfg.aug_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = identity_transform

    # ImageFolder：会扫描 root 下的子文件夹；每个子文件夹名视为一个类别；每张图片会自动被标记为 (path, class_index)；同时应用下面的 transform 序列：
    dataset = datasets.ImageFolder(
        root,
        transforms.Compose(
            [
                rz_func,
                transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if cfg.aug_norm
                else identity_transform,
            ]
        )
    )

    # ---- Force consistent labels: real -> 0, others -> 1 ----重新定义标签规则（核心）

    # 先找到 “real” 类的原编号 real_idx
    real_idx = None
    for cls, idx in dataset.class_to_idx.items():
        if cls.lower() == "real":
            real_idx = idx
            break

    # Determine the non-real class name if uniquely defined
    # 确定 “非real” 类的名字，如果只存在一个非 real 类，那就是本身；如果有多个，统一为“other”
    non_real_names = [c for c in dataset.class_to_idx.keys() if c.lower() != "real"]
    other_name = non_real_names[0] if len(non_real_names) == 1 else "other"

    #  创建新的标签表
    new_targets = []
    if real_idx is None:
        # If 'real' class is absent, map all to 1 (other)
        # 如果没找到 real（极少数情况）→ 全部设为 1；
        for _, _label in dataset.samples:
            new_targets.append(1)
        dataset.class_to_idx = {"real": 0, other_name: 1}
        dataset.classes = ["real", other_name]
    else:
        # 只要原标签等于 real_idx → 改为 0；其他一律改为 1；更新类名和映射：{'real':0, 'adm':1}。
        for _, _label in dataset.samples:
            new_targets.append(0 if _label == real_idx else 1)
        dataset.class_to_idx = {"real": 0, other_name: 1}
        dataset.classes = ["real", other_name]

    # Apply remapped targets to samples/targets
    # 重写数据集（path，label）
    dataset.samples = [(p, t) for (p, _), t in zip(dataset.samples, new_targets)]
    dataset.targets = list(new_targets)

    # ---- Debug: show mapping between file names and new labels ----
    print("\n[DEBUG] Dataset Summary:")
    print("Classes:", dataset.classes)
    print("Class mapping:", dataset.class_to_idx)
    print("Example samples (first 5 and last 5):")

    total = len(dataset.samples)
    show_n = 5
    head_samples = dataset.samples[:show_n]
    tail_samples = dataset.samples[-show_n:] if total > show_n else []

    for i, (path, label) in enumerate(head_samples):
        print(f"  [HEAD {i}] {os.path.basename(path)} -> {label}")

    if tail_samples:
        print("  ...")
        for i, (path, label) in enumerate(tail_samples, start=total - len(tail_samples)):
            print(f"  [TAIL {i}] {os.path.basename(path)} -> {label}")

    print(f"Total samples: {total}\n")

    return dataset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def blur_jpg_augment(img: Image.Image, cfg: CONFIGCLASS):
    img: np.ndarray = np.array(img)
    if cfg.isTrain:
        if random() < cfg.blur_prob:
            sig = sample_continuous(cfg.blur_sig)
            gaussian_blur(img, sig)

        if random() < cfg.jpg_prob:
            method = sample_discrete(cfg.jpg_method)
            qual = sample_discrete(cfg.jpg_qual)
            img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s: list):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s: list):
    return s[0] if len(s) == 1 else choice(s)


def gaussian_blur(img: np.ndarray, sigma: float):
    # Work in float to avoid uint8 wrap/rounding artifacts
    tmp = img.astype(np.float32, copy=True)
    gaussian_filter(tmp[:, :, 0], output=tmp[:, :, 0], sigma=sigma)
    gaussian_filter(tmp[:, :, 1], output=tmp[:, :, 1], sigma=sigma)
    gaussian_filter(tmp[:, :, 2], output=tmp[:, :, 2], sigma=sigma)
    np.clip(tmp, 0, 255, out=tmp)
    img[:, :, :] = tmp.astype(np.uint8)


def cv2_jpg(img: np.ndarray, compress_val: int) -> np.ndarray:
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    if not result:
        raise RuntimeError("cv2.imencode failed during JPEG augmentation.")
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img: np.ndarray, compress_val: int):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img: np.ndarray, compress_val: int, key: str) -> np.ndarray:
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img: Image.Image, cfg: CONFIGCLASS) -> Image.Image:
    # Deterministic interpolation for eval; sampling only during training
    interp = sample_discrete(cfg.rz_interp) if cfg.isTrain else cfg.rz_interp[0]
    return TF.resize(img, cfg.loadSize, interpolation=rz_dict.get(interp, Image.BILINEAR))


# def get_dataset(cfg: CONFIGCLASS):
#     dset_lst = []
#     for dataset in cfg.datasets:
#         root = os.path.join(cfg.dataset_root, dataset)
#         dset = dataset_folder(root, cfg)
#         dset_lst.append(dset)
#     return torch.utils.data.ConcatDataset(dset_lst)

def get_dataset(cfg: CONFIGCLASS):
    # 二分类：ImageFolder 直接指向父目录（.../train 或 .../val）
    if cfg.mode == "binary":
        root = cfg.dataset_root  # 例如 .../diffrecon/train 或 .../diffrecon/val
        print("get_dataset", root)
        return dataset_folder(root, cfg)

    # 其它模式（如 filename）保持原逻辑
    dset_lst = []
    for dataset in cfg.datasets:
        root = os.path.join(cfg.dataset_root, dataset)
        dset = dataset_folder(root, cfg)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    # Support both ConcatDataset and single ImageFolder
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        datasets_list = dataset.datasets
    else:
        datasets_list = [dataset]

    targets = []
    for d in datasets_list:
        if hasattr(d, "targets"):
            targets.extend(d.targets)
        elif hasattr(d, "samples"):
            # torchvision ImageFolder has .samples as list of (path, class_index)
            targets.extend([cls for _, cls in d.samples])
        else:
            raise AttributeError("Dataset does not expose 'targets' or 'samples' needed for class-balanced sampling.")

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))


def create_dataloader(cfg: CONFIGCLASS):
    shuffle = not cfg.serial_batches if (cfg.isTrain and not cfg.class_bal) else False
    # 打乱顺序---如果是训练模式 (cfg.isTrain=True) 且不使用类别均衡采样（not cfg.class_bal）
    dataset = get_dataset(cfg)

    sampler = get_bal_sampler(dataset) if cfg.class_bal else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
    )
