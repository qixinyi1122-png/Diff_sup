#!/usr/bin/env bash
set -euo pipefail

# 清理无用的 Jupyter 检查点目录（可选）
find /root/autodl-tmp -type d -name ".ipynb_checkpoints" -exec rm -rf {} + || true


export CUDA_VISIBLE_DEVICES=0        # 选择 GPU

# ------------ 选择评测数据集与 split 组 ------------
# # LSUN-Bedroom 示例
EXP_NAME="lsun_adm"
# DATASETS_TEST='[adm, pndm, ddpm, dalle2, iddpm, if, ldm, midjourney, sdv1_new, sdv2, vqdiffusion, stylegan_official, projectedgan, diff-projectedgan, diff-stylegan]'
DATASETS_TEST='[adm, pndm]'
DATASET_ROOT=lsun_bedroom



SUP="False"  # True=AND(SUP)；False=OR
if [ "${SUP}" = "True" ]; then
  METHOD="INF"  #两个都是假图才是假图
else
  METHOD="SUP"  #只要有一个是假的我就认为是假的
fi


# ------------ 两个模型各自的数据子目录（A= DΙRE，B= DIFF）------------
DATA_A="dire"         # A 侧（DIRE）数据子目录
DATA_B="diffrecon"    # B 侧（DIFF）数据子目录


# Train_Models=("lsun_adm" "img_adm" "kan3_adm" "sdxl_adm")
Train_Models=("lsun_adm")

BATCH_SIZES=(128 256 512)
Thresholds=(0.2923932188 0.5 0.7071067812)

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for Train_Model in "${Train_Models[@]}"; do
        for Threshold in "${Thresholds[@]}"; do
            echo ">>> 开始测试：Train_Model=${Train_Model}, BATCH_SIZE=${BATCH_SIZE}, Threshold=${Threshold}"


            CKPT_DIRE="/root/autodl-tmp/result/${Train_Model}/${BATCH_SIZE}/fixed/${DATA_A}/exp/${Train_Model}/ckpt/model_epoch_best.pth"
            CKPT_DIFF="/root/autodl-tmp/result/${Train_Model}/${BATCH_SIZE}/fixed/${DATA_B}/exp/${Train_Model}/ckpt/model_epoch_best.pth"
            
            
            # ------------ 数据根（两侧各自一套）------------
            DATASET_ROOT_DIRE="/root/autodl-tmp/Reconstruct/${DATASET_ROOT}/fixed/${DATA_A}"
            DATASET_ROOT_DIFF="/root/autodl-tmp/Reconstruct/${DATASET_ROOT}/fixed/${DATA_B}"
            
            
            
            # 运行目录（用于 test 输出与中间文件）
            EXP_DIR_DIRE="/root/autodl-tmp/result/${Train_Model}/${BATCH_SIZE}/fixed/${METHOD}_thr_${Threshold}/"
            
            
            python test_sup.py \
              --exp_name "${EXP_NAME}" \
              SUP "${SUP}" \
              ckpt_path_dire "${CKPT_DIRE}" \
              ckpt_path_diff "${CKPT_DIFF}" \
              dataset_root_dire "${DATASET_ROOT_DIRE}" \
              dataset_root_diff "${DATASET_ROOT_DIFF}" \
              datasets_test "${DATASETS_TEST}" \
              exp_root "${EXP_DIR_DIRE}" \
              batch_size 512 \
              decision_threshold ${Threshold}

        done
    done
done

