#!/bin/env bash
source activate HACplus_env
scenes_mipnerf=("garden" "bicycle" "stump" "bonsai" "counter" "kitchen" "room" "treehill" "flowers") # 9 scenes
scenes_db=("drjohnson" "playroom") # 2 scenes
scenes_tandt=("train" "truck") # 2 scenes

scene_list=("stump"  "counter" "kitchen" "room" "treehill" "flowers" "drjohnson"  "garden" "bicycle" "bonsai" "playroom"  "train" "truck")
# scene_list=("stump")
for scene in "${scene_list[@]}"; do
    echo Processing scene: $scene

    if [[ " ${scenes_mipnerf[*]} " == *" $scene "* ]]; then
        dataset="mipnerf360"
    elif [[ " ${scenes_db[*]} " == *" $scene "*   ]]; then
        dataset="db"
    elif [[ " ${scenes_tandt[*]} " == *" $scene "*   ]]; then
        dataset="tandt"
    else
        echo The scene [$scene] not exists, please recheck!
        exit 1
    fi

# scene=garden
# dataset=mipnerf360
    lmbda=0.004
    mask_lr_final=0.0015
    CUDA_VISIBLE_DEVICES=0  CUDA_LAUNCH_BLOCKING=1 python test_fps.py -s  ../../../data/GS/${dataset}/${scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/ex3rendering_speed/${dataset}/${scene}/${lmbda} --lmbda ${lmbda} --mask_lr_final ${mask_lr_final} --save_iterations 1000 30000
done