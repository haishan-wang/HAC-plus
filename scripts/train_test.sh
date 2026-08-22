
# * basic settings
source activate HACplus_env
hostname
nvidia-smi
echo "Running in partition: $SLURM_JOB_PARTITION"
echo "Job nodes: $SLURM_JOB_NODELIST"

path_base=../../../data/GS
output_base=output/test

scene=ship
dataset=nerf_synthetic
path_source="$path_base"/"$dataset"/"$scene"
path_output="$output_base"/"$dataset"/"$scene"


iterations=30000
test_iterations=$(seq 0 100 ${iterations})

lmbda=0.001
mask_lr_final=0.00008
CUDA_VISIBLE_DEVICES=0 python train.py -s  ../../../data/GS/${dataset}/${scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations ${iterations} -m ${path_output}/${scene}/${lmbda} --lmbda ${lmbda} --mask_lr_final ${mask_lr_final} --use_wandb  --test_iterations ${test_iterations}  --image_record --wandb_exp_name ${scene}_track