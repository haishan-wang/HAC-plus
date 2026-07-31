#!/bin/env bash
if [[ $SERVER_NAME == 'pcsc' ]]; then
    if [[ $1 == "gpu" ]]; then
        srun --account=project_2007966 --gres=gpu:v100:1 --partition gpu --time=$2:00:00 --mem=4G --pty bash 
        # srun --account=project_2004878 --gres=gpu:v100:1 --partition gpu --time=2:00:00 --mem=4G --pty bash 
        
    fi
    if [[ $1 == "cpu" ]]; then
        sinteractive --account project_2007966 --mem 4G  --time $2:00:00 
    fi
else
    if [[ $1 == "gpudebug" ]]; then
        srun --gpus=1 -p gpu-debug --time=00:30:00 --mem=16G --pty bash 
    fi
    if [[ $1 == "gpu" ]]; then
        # srun --gpus=1 --time=$2:00:00 --mem=4G --pty bash 
        srun  --time=$2:00:00 --mem=32G --gres=gpu:1 --partition=gpu-h200-141g-ellis --account ellis_users --pty bash  
        # srun  --time=$2:00:00 --mem=32G --gres=gpu:1 --partition=gpu-a100-80g --pty bash  
        # srun  --time=$2:00:00 --mem=32G --gres=gpu:1 --partition=gpu-v100-32g --pty bash 
    fi
    if [[ $1 == "cpu" ]]; then
        srun --time=$2:00:00 --mem=2G --pty bash 
    fi
fi

# slurm p can check the partition info
# salloc can request resource without getin 

# srun -p interactive --time=2:00:00 --mem=4G --gres=gpu:1 --pty bash 
# srun -p gpu-debug --time=2:00:00 --mem=4G --gpus=1 --pty bash 

# srun --time=1:00:00 --mem=2G --pty bash 
# python main.py --data_name PubMed%        




# salloc --gpus=1 --time=2:00:00 --mem=4G bash 

# salloc  --time=3:00:00 --mem=32G --gres=gpu:1 --partition=gpu-h200-141g-ellis --account ellis_users  bash  

# srun  --time=1:00:00 --mem=64G --gres=gpu:1 --partition=gpu-v100-16g --pty bash  