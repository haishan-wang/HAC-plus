import os
from pathlib import Path
# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for scene in ['bicycle']:
#         mask_lr_final = 0.0005 * lmbda / 0.001
#         mask_lr_final = min(mask_lr_final, 0.0015)
#         one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s  ../../../data/GS/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/mipnerf360/{scene}/{lmbda} --lmbda {lmbda} --mask_lr_final {mask_lr_final}'
#         os.system(one_cmd)

def submit_job(lmbda, run_script= False, job_name = 'HACplus'):
    mask_lr_final = 0.0005 * lmbda / 0.001
    mask_lr_final = min(mask_lr_final, 0.0015)
    scripts = [
'''#!/bin/env bash
#SBATCH --gres=gpu:1''',
f'#SBATCH --job-name={job_name}',
f'#SBATCH --output=scripts/ex1reproduce/sbatch_logs/{job_name}/%A_%a.log',
'''#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --array=0-12
#SBATCH --account ellis_users

##SBATCH --partition=gpu-a100-80g
# * basic settings
source activate HACplus_env
hostname
nvidia-smi
echo "Running in partition: $SLURM_JOB_PARTITION"
echo "Job nodes: $SLURM_JOB_NODELIST"

path_base=../../../data/GS

# * get the scene and dataset
#             0         1         2       3         4         5         6          7         8         9           10        11      12
scene_list=("stump"  "counter" "kitchen" "room" "treehill" "flowers" "drjohnson"  "garden" "bicycle" "bonsai" "playroom"  "train" "truck")
scenes_mipnerf=("garden" "bicycle" "stump" "bonsai" "counter" "kitchen" "room" "treehill" "flowers") # 9 scenes
scenes_db=("drjohnson" "playroom") # 2 scenes
scenes_tandt=("train" "truck") # 2 scenes
SCENE=${scene_list[SLURM_ARRAY_TASK_ID]}
if [[ " ${scenes_mipnerf[*]} " == *" $SCENE "* ]]; then
    DATASET="mipnerf360"
elif [[ " ${scenes_db[*]} " == *" $SCENE "*   ]]; then
    DATASET="db"
elif [[ " ${scenes_tandt[*]} " == *" $SCENE "*   ]]; then
    DATASET="tandt"
else
    echo The scene [$SCENE] not exists, please recheck!
    exit 1
fi
echo scene: $SCENE, dataset: $DATASET
dataset=$DATASET
scene=$SCENE
path_source="$path_base"/"$dataset"/"$scene"
path_output="$output_base"/"$dataset"/"$scene"

''',
        f'lmbda={lmbda}',
        f'mask_lr_final={mask_lr_final}',
        'CUDA_VISIBLE_DEVICES=0 python train.py -s  ../../../data/GS/${dataset}/${scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/ex1reproduce/${dataset}/${scene}/${lmbda} --lmbda ${lmbda} --mask_lr_final ${mask_lr_final}',
        
    ]
    script = '\n'.join(scripts)
    save_path = Path(f'scripts/ex1reproduce/tmp/submit.sh')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w') as f:
        f.write(script)
    print(f'Saved sbatch script to {save_path}')
    if run_script:
        os.system(f'sbatch {save_path}')
        
        
if __name__ == '__main__':
    lmbda = 0.004
    
    mask_lr_final = 0.0005 * lmbda / 0.001
    mask_lr_final = min(mask_lr_final, 0.0015)
    submit_job(lmbda = lmbda, run_script = True)