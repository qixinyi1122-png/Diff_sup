#!/usr/bin/env bash
set -euo pipefail

##### 清理无用目录 #####
find /root/autodl-tmp -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
echo "<<< 清理无用目录 /root/autodl-tmp + .ipynb_checkpoints"

#############################
# 配置区（按需修改）
#############################
export CUDA_VISIBLE_DEVICES=0        # 选择 GPU
export EXP_NAME="kan3_adm"           # 实验名字（用于区分不同实验）

# 训练数据集（如果 train.py 里期望是列表/JSON 字符串，就保持这种写法）
DATASETS="kan3_adm" 

# 两种映射模式（你之前写成了不合法的 Bash 语法）
VIS_MODE="fixed"

# 你的数据子配置（用于拼接路径）
# DATAS=("dire")
DATAS=("diffrecon" "dire")

# 是否在所有循环完成后关机（true/false）
DO_SHUTDOWN=false

BATCH_SIZES=(128 256 512)

#############################
# 启动训练
#############################
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for DATA in "${DATAS[@]}"; do
      echo ">>> 开始训练：BATCH_SIZE=${BATCH_SIZE}, DATA=${DATA}"
    
      # 构造路径（按你的约定）
      DATASET_ROOT="/root/autodl-tmp/Reconstruct/new_datasets/${VIS_MODE}/${DATA}"
      EXP_ROOT="/root/autodl-tmp/result/kan3_adm/${BATCH_SIZE}/${VIS_MODE}/${DATA}/exp"

      
      # 创建输出目录
      mkdir -p "${EXP_ROOT}"
    
      # 如果 train.py 使用 argparse（常见写法），参数需要带 "--"
      # 若你的 train.py 使用其他解析方式，请按需改回去
      python train.py \
        --exp_name "${EXP_NAME}" \
        dataset_root "${DATASET_ROOT}" \
        exp_root "${EXP_ROOT}" \
        batch_size "${BATCH_SIZE}" \
        datasets "${DATASETS}"
    
      echo "<<< 完成：BATCH_SIZE=${BATCH_SIZE}, 输出目录：${EXP_ROOT}"
    done
done

#############################
# 所有循环结束后的动作
#############################
# echo "[ALL DONE] 所有任务完成。"

# if [[ "${DO_SHUTDOWN}" == "true" ]]; then
#   echo "系统将在 60 秒后关机..."
#   sleep 60
#   /usr/bin/shutdown
# fi