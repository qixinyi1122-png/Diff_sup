#!/usr/bin/env bash
set -euo pipefail

find /root/autodl-tmp -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

#############################
# 配置区（按需修改）
#############################
export CUDA_VISIBLE_DEVICES=0        # 选择 GPU



#laion_test
EXP_NAME="laion_adm"
DATASETS_TEST='[kandinsky3, sdxl, vega, kandinsky2, pixart-alpha-512, playground-25, stable-cascade, wurstchen2]'

DATASET_ROOT=new_datasets



DATAS=("dire" "diffrecon")
BATCH_SIZES=(128 256 512)
Train_Models=("lsun_adm" "img_adm" "kan3_adm" "sdxl_adm")

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for DATA in "${DATAS[@]}"; do
        for Train_Model in "${Train_Models[@]}"; do
            echo ">>> 开始测试：Train_Model=${Train_Model}, BATCH_SIZE=${BATCH_SIZE}, DATA=${DATA}"
            CKPT="/root/autodl-tmp/result/${Train_Model}/${BATCH_SIZE}/fixed/${DATA}/exp/${Train_Model}/ckpt/model_epoch_best.pth"

            python test.py \
              --exp_name "$EXP_NAME" \
              --ckpt "$CKPT" \
              exp_root "/root/autodl-tmp/result/${Train_Model}/${BATCH_SIZE}/fixed/${DATA}/exp/" \
              batch_size 512 \
              datasets_test "${DATASETS_TEST}" \
              dataset_root "/root/autodl-tmp/Reconstruct/${DATASET_ROOT}/fixed/${DATA}"

        done
    done
done

# # ====== 所有循环结束后再关机 ======
# echo "[ALL DONE] 所有任务完成，准备关机..."
# /usr/bin/shutdown
    