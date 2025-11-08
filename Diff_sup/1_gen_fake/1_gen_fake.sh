#!/usr/bin/env bash
set -e


# torch 2.8.0 + py3.12 + CUDA 12.8

#####################################
PROMPT_COUNT=12000               
SRC_SAVE_ROOT=/root/autodl-tmp/laion         # 生成图片的时候放的位置
OUT_DATASET_ROOT=/root/autodl-tmp/new_datasets/images    #划分生成的图片为train，val，test所在文件夹

#####################################

echo "[1/7] 安装基础工具（需要 sudo）"
sudo apt-get update
sudo apt-get install -y git wget curl parallel pigz imagemagick

echo "[2/7] 安装 Python 包（diffusers 等）"

pip install --upgrade pip
pip install diffusers transformers accelerate safetensors datasets pillow tqdm einops huggingface_hub

echo "[3/7] 克隆仓库"
cd /root
git clone https://github.com/georgecazenavette/easy-diffusion-generation.git

echo "[4/7] 生成 prompts (${PROMPT_COUNT} 条)"
cd /root
cat > build_prompts.py << PY
import datasets
ds = datasets.load_dataset("wanng/midjourney-v5-202304-clean", split="train")
ds = ds.filter(lambda ex: ex.get("upscaled", False)).shuffle(seed=42)
prompts = [ex['clean_prompts'] for ex in ds.select(range(${PROMPT_COUNT}))]
with open("fake_inversion.txt", "w") as f:
    f.write("\\n".join(prompts))
print("wrote fake_inversion.txt with", len(prompts), "lines")
PY

python build_prompts.py

# 放到仓库 prompts/ 里，并且名字叫 fake_inversion.txt
mv /root/fake_inversion.txt /root/easy-diffusion-generation/prompts/fake_inversion.txt

echo "[5/7] 安装仓库自己的依赖（如果有）"
cd /root/easy-diffusion-generation
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

echo "[6/7] 批量生成 8 个模型的图片"
MODELS="kandinsky2 kandinsky3 pixart-alpha-512 playground-25 sdxl stable-cascade vega wurstchen2"

for m in $MODELS; do
  echo "== generate with model: $m =="
  python gen.py \
    --model=${m} \
    --prompts=fake_inversion \
    --save_root ${SRC_SAVE_ROOT}/${m} \
    --det_seed \
    --skip_existing
done

echo "[7/7] 生成并运行划分脚本"
cat > /root/split_dataset.sh << 'SPLIT'
#!/usr/bin/env bash
set -e
SRC_ROOT=$1
DST_ROOT=$2

MODELS="kandinsky2 kandinsky3 pixart-alpha-512 playground-25 sdxl stable-cascade vega wurstchen2"

for m in $MODELS; do
  src_dir=${SRC_ROOT}/${m}
  train_dir=${DST_ROOT}/train/laion/${m}
  val_dir=${DST_ROOT}/val/laion/${m}
  test_dir=${DST_ROOT}/test/laion/${m}

  mkdir -p "$train_dir" "$val_dir" "$test_dir"

  echo "== split $m =="

  count=0
  # 不依赖文件名顺序，只要是文件就按出现顺序分
  find "$src_dir" -maxdepth 1 -type f | while read -r file; do
    if [ $count -lt 10000 ]; then
      cp "$file" "$train_dir/"
    elif [ $count -lt 11000 ]; then
      cp "$file" "$val_dir/"
    elif [ $count -lt 12000 ]; then
      cp "$file" "$test_dir/"
    else
      break
    fi
    count=$((count+1))
  done
done
SPLIT

chmod +x /root/split_dataset.sh
/root/split_dataset.sh ${SRC_SAVE_ROOT} ${OUT_DATASET_ROOT}

echo "全部完成 ✅"
echo "生成图片在: ${SRC_SAVE_ROOT}"
echo "划分后数据在: ${OUT_DATASET_ROOT}"