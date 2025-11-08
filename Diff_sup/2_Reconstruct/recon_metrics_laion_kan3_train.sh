#!/usr/bin/env bash
set -euo pipefail

# 清除所有 .ipynb_checkpoints 文件夹
find /root/ -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
find /root/autodl-tmp/ -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

echo "[INFO] 已清除所有 .ipynb_checkpoints 文件夹"

# =========================
# 配置区（按需修改）
# =========================
SPLITS=("train" "val" "test")
DATASETS=("real" "kandinsky3")

MODEL_PATH="/root/autodl-tmp/models/256x256_diffusion_uncond.pt"

BATCH=135
HAS_SUBFOLDER="False"

TIMESTEPS="ddim20"
USE_DDIM=True
VIS_MODE="fixed"

export CUDA_VISIBLE_DEVICES=0

for SPLIT in "${SPLITS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    IMG_DIR="/root/autodl-tmp/new_datasets/images/$SPLIT/laion/$DATASET"

    OUT_R="/root/autodl-tmp/all_Reconstruct/new_datasets/$VIS_MODE/recons/$SPLIT/kan3_adm/$DATASET"
    OUT_R2="/root/autodl-tmp/all_Reconstruct/new_datasets/$VIS_MODE/recons2/$SPLIT/kan3_adm/$DATASET"
    OUT_D1="/root/autodl-tmp/all_Reconstruct/new_datasets/$VIS_MODE/dire/$SPLIT/kan3_adm/$DATASET"
    OUT_D2="/root/autodl-tmp/all_Reconstruct/new_datasets/$VIS_MODE/dire2/$SPLIT/kan3_adm/$DATASET"
    OUT_DIFF="/root/autodl-tmp/all_Reconstruct/new_datasets/$VIS_MODE/diffrecon/$SPLIT/kan3_adm/$DATASET"

    mkdir -p "$OUT_R" "$OUT_R2" "$OUT_D1" "$OUT_D2" "$OUT_DIFF"

    echo "==[INFO]== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

    LIST_FILE="/tmp/recon_${SPLIT}_${DATASET}.txt"

    echo "==[STEP]== 扫描图像..."
    find "$IMG_DIR" -type f \
      \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) \
      -print | sort > "$LIST_FILE"

    TOTAL=$(wc -l < "$LIST_FILE" || echo 0)
    if [[ "$TOTAL" -eq 0 ]]; then
      echo "!! 没找到任何图片：$IMG_DIR"
      rm -f "$LIST_FILE"
      exit 1
    fi

    echo "==[INFO]== 总图片数: $TOTAL（不切块，直接处理全部图像）"

    echo "==[RUN]== 开始处理全部图像（一次性）"
    time python recon_metrics.py \
      --images_dir "$IMG_DIR" \
      --recons_dir "$OUT_R" \
      --recons2_dir "$OUT_R2" \
      --dire_dir   "$OUT_D1" \
      --dire2_dir  "$OUT_D2" \
      --diff_dir   "$OUT_DIFF" \
      --model_path "$MODEL_PATH" \
      --has_subfolder "$HAS_SUBFOLDER" \
      --image_size 256 \
      --num_channels 256 \
      --num_res_blocks 2 \
      --channel_mult "1,1,2,2,4,4" \
      --attention_resolutions "32,16,8" \
      --num_heads -1 \
      --num_head_channels 64 \
      --resblock_updown True \
      --use_scale_shift_norm True \
      --learn_sigma True \
      --class_cond False \
      --clip_denoised True \
      --use_ddim "$USE_DDIM" \
      --timestep_respacing "$TIMESTEPS" \
      --batch_size "$BATCH" \
      --num_samples "$TOTAL" \
      --save_npy True \
      --abs_d1 True \
      --abs_d2 True \
      --abs_final_diff False \
      --use_fp16 True \
      --use_checkpoint False \
      --vis_mode "$VIS_MODE"

    echo "==[ALL DONE]== 全部处理完成（未使用切块）。输出目录："
    echo "  recons:   $OUT_R"
    echo "  recons2:  $OUT_R2"
    echo "  dire(D1): $OUT_D1"
    echo "  dire2(D2):$OUT_D2"
    echo "  diff:     $OUT_DIFF"

    # ====== 新增：把图片拎到 Reconstruct 下 ======
    NEW_OUT_D1="/root/autodl-tmp/Reconstruct/new_datasets/$VIS_MODE/dire/$SPLIT/kan3_adm/$DATASET"
    NEW_OUT_DIFF="/root/autodl-tmp/Reconstruct/new_datasets/$VIS_MODE/dire/$SPLIT/kan3_adm/$DATASET"
    mkdir -p "$NEW_OUT_D1" "$NEW_OUT_DIFF"

    # 只搬常见图片格式
    find "$OUT_D1"   -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) -exec mv {} "$NEW_OUT_D1" \;
    find "$OUT_DIFF" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) -exec mv {} "$NEW_OUT_DIFF" \;

    # 清理当前任务的临时清单文件
    rm -f "$LIST_FILE"
  done
done