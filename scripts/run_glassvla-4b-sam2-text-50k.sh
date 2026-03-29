#!/bin/bash
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate groundedsam2_mine

# bash scripts/run_glassvla-4b-sam2-text-50k.sh 2>&1 | tee run_glassvla-4b-sam2-text-50k.log

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_LOADER_LAYERS_DISABLE=VK_LAYER_MESA_device_select:VK_LAYER_INTEL_nullhw
export QT_QPA_PLATFORM=xcb
export __GLX_VENDOR_LIBRARY_NAME=nvidia

export ENABLE_IMAGE_SIMPLIFICATION=1 # 是否启用图像简化
# IMAGE_SIMPLIFICATION_BLUR_SIGMA 已由模型内部 adaptive_blur 控制，这里不再使用
export IMAGE_SIMPLIFICATION_DETECTION_INTERVAL=20 # GroundingDino检测频率（1=每帧检测，禁用SAM2追踪）

export IMAGE_SIMPLIFICATION_BOX_THRESHOLD=0.35   # groundingDino使用的box threshold
export IMAGE_SIMPLIFICATION_TEXT_THRESHOLD=0.20  # groundingDino使用的text threshold


model_name=spatialvla
tasks=(
    # bridge.sh
    drawer_variant_agg.sh
    drawer_visual_matching.sh
    move_near_variant_agg.sh
    move_near_visual_matching.sh
    pick_coke_can_variant_agg.sh
    pick_coke_can_visual_matching.sh

    # put_in_drawer_variant_agg.sh
    # put_in_drawer_visual_matching.sh
)

ckpts=(
    /home/futuremm/workplace/ckpts/mlp/checkpoint-50000
)

action_ensemble_temp=-0.8

for ckpt_path in ${ckpts[@]}; do
    logging_dir=results_glassvla-4b-sam2-text-50k/$(basename $ckpt_path)${action_ensemble_temp}
    mkdir -p $logging_dir
    
    for i in ${!tasks[@]}; do
        task=${tasks[$i]}
        echo "running $task with simplification..."
        device=1
        bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir $device
    done

    python tools/calc_metrics_evaluation_videos.py --log-dir-root $logging_dir >>$logging_dir/total.metrics
    echo "Results: $logging_dir"
done
