import argparse
import os
import sys
from abc import ABC
from typing import Type


class DefaultConfigs(ABC):
    ####### base setting ######
    gpus = [0]
    seed = 3407
    arch = "resnet50"
    datasets = ["lsun_adm"]
    datasets_test = ["lsun_adm"]
    mode = "binary"
    class_bal = False
    batch_size = 128
    loadSize = 256
    cropSize = 224
    epoch = "latest"
    num_workers = 20
    serial_batches = False
    isTrain = False

    # data augmentation
    rz_interp = ["bilinear"]
    blur_prob = 0.0
    blur_sig = [0.5]
    jpg_prob = 0.0
    jpg_method = ["cv2"]
    jpg_qual = [75]
    gray_prob = 0.0
    aug_resize = True
    aug_crop = True
    aug_flip = True
    aug_norm = True

    ####### train setting ######
    warmup = False
    warmup_epoch = 3
    earlystop = False
    earlystop_epoch = 20
    optim = "adam"
    new_optim = False
    loss_freq = 400
    save_latest_freq = 2000
    save_epoch_freq = 20
    continue_train = False
    epoch_count = 1
    last_epoch = -1
    nepoch = 400
    beta1 = 0.9
    lr = 0.0001
    init_type = "normal"
    init_gain = 0.02
    pretrained = True

    # paths information
    # root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # root_dir = "/root/autodl-tmp"
    dataset_root = "/root/autodl-tmp/data/dire"
    # os.path.join(root_dir, "data", "dire")
    exp_root = "/root/autodl-tmp/dire/exp"

    # =========================
    # test-only：仅测试/融合脚本(test.py)会读取的键；
    # 训练(train.py)不会使用这些键，保留在全局配置便于 --opts 动态覆盖。
    # =========================
    SUP = True
    
    # A 侧（DIRE）与 B 侧（DIFF）的数据根；默认给出一个常见路径，也可通过 --opts 覆盖
    dataset_root_dire = "/root/autodl-tmp/data/dire"        # test-only
    dataset_root_diff = "/root/autodl-tmp/data/diffrecon"   # test-only

    # 两个模型各自的 checkpoint 路径；默认留空，通常运行时用 --opts 传入
    ckpt_path_dire = ""  # test-only，例如 /path/to/dire/exp/.../ckpt/model_epoch_latest.pth
    ckpt_path_diff = ""  # test-only，例如 /path/to/diff/exp/.../ckpt/model_epoch_latest.pth

    # AND/SUP 融合判定阈值；metrics 与逐文件 CSV 的 predA/predB/predSUP 使用
    decision_threshold = 0.5  # test-only
    # os.path.join(root_dir, "dire", "exp")
    _exp_name = ""
    exp_dir = ""
    ckpt_dir = ""
    logs_path = ""
    ckpt_path = ""

    @property
    def exp_name(self):
        return self._exp_name

    @exp_name.setter
    def exp_name(self, value: str):
        self._exp_name = value
        self.exp_dir: str = os.path.join(self.exp_root, self.exp_name)
        self.ckpt_dir: str = os.path.join(self.exp_dir, "ckpt")
        self.logs_path: str = os.path.join(self.exp_dir, "logs.txt")

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def to_dict(self):
        dic = {}
        for fieldkey in dir(self):
            fieldvalue = getattr(self, fieldkey)
            if not fieldkey.startswith("__") and not callable(fieldvalue) and not fieldkey.startswith("_"):
                dic[fieldkey] = fieldvalue
        return dic


def args_list2dict(arg_list: list):
    assert len(arg_list) % 2 == 0, f"Override list has odd length: {arg_list}; it must be a list of pairs"
    return dict(zip(arg_list[::2], arg_list[1::2]))


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    elif v.lower() in ("true", "yes", "on", "y", "t", "1"):
        return True
    elif v.lower() in ("false", "no", "off", "n", "f", "0"):
        return False
    else:
        return bool(v)


def str2list(v: str, element_type=None) -> list:
    if not isinstance(v, (list, tuple, set)):
        v = v.lstrip("[").rstrip("]")
        v = v.split(",")
        v = list(map(str.strip, v))
        if element_type is not None:
            v = list(map(element_type, v))
    return v


CONFIGCLASS = Type[DefaultConfigs]

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default=[0], type=int, nargs="+")
parser.add_argument("--exp_name", default="", type=str)
parser.add_argument("--ckpt", default="model_epoch_latest.pth", type=str)
parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
args = parser.parse_args()

if os.path.exists(os.path.join(DefaultConfigs.exp_root, args.exp_name, "config.py")):
    sys.path.insert(0, os.path.join(DefaultConfigs.exp_root, args.exp_name))
    from config import cfg

    cfg: CONFIGCLASS
else:
    cfg = DefaultConfigs()

if args.opts:
    opts = args_list2dict(args.opts)
    for k, v in opts.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unrecognized option: {k}")
        original_type = type(getattr(cfg, k))
        if original_type == bool:
            setattr(cfg, k, str2bool(v))
        elif original_type in (list, tuple, set):
            setattr(cfg, k, str2list(v, type(getattr(cfg, k)[0])))
        else:
            setattr(cfg, k, original_type(v))

cfg.gpus: list = args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu) for gpu in cfg.gpus])
cfg.exp_name = args.exp_name
cfg.ckpt_path: str = os.path.join(cfg.ckpt_dir, args.ckpt)

if isinstance(cfg.datasets, str):
    cfg.datasets = cfg.datasets.split(",")
