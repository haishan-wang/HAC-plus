"""ex5: full HAC++ benchmark -- PSNR/SSIM/LPIPS, encoded size, training time and FPS on
mipnerf360, db, tandt, nerf_synthetic and bungeenerf.

Training is upstream HAC++ (YihangChen-ee/HAC-plus) unchanged; per-dataset hyper-parameters are
exactly those of the upstream run_shell_*.py scripts, including lmbda (one per dataset).
One SLURM array task = one (dataset, scene, lmbda). Each finished run writes
<model_path>/metrics.json; collect.py gathers them and compares with results/.

Usage (from the repo root on triton):
    python scripts/ex5full_benchmark/submit.py                         # write the sbatch script only
    python scripts/ex5full_benchmark/submit.py --run                   # ... and submit it
    python scripts/ex5full_benchmark/submit.py --datasets db tandt --run
    python scripts/ex5full_benchmark/submit.py --lmbdas 0.0005 --run      # override lmbda for all datasets
"""
import argparse
from pathlib import Path

EXP = 'ex5full_benchmark'
DATA_BASE = '../../../data/GS'
OUTPUT_BASE = f'outputs/{EXP}'

# dataset -> scenes and the flags used by the corresponding upstream run_shell_*.py
DATASETS = {
    'mipnerf360': dict(  # run_shell_mip360.py
        scenes=['bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai', 'flowers', 'treehill'],
        lmbdas=[0.004], lod=0, voxel_size=0.001, update_init_factor=16,
        mask_lr_final=lambda l: min(0.0005 * l / 0.001, 0.0015)),
    'db': dict(  # run_shell_db.py
        scenes=['playroom', 'drjohnson'],
        lmbdas=[0.004], lod=0, voxel_size=0.005, update_init_factor=16,
        mask_lr_final=lambda l: 0.00008 * l / 0.001),
    'tandt': dict(  # run_shell_tnt.py
        scenes=['truck', 'train'],
        lmbdas=[0.004], lod=0, voxel_size=0.01, update_init_factor=16,
        mask_lr_final=lambda l: 0.0001 * l / 0.001),
    'nerf_synthetic': dict(  # run_shell_blender.py
        scenes=['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship'],
        lmbdas=[0.001], lod=0, voxel_size=0.001, update_init_factor=4,
        mask_lr_final=lambda l: 0.00008 * l / 0.001),
    'bungeenerf': dict(  # run_shell_bungee.py
        scenes=['amsterdam', 'bilbao', 'hollywood', 'pompidou', 'quebec', 'rome'],
        lmbdas=[0.004], lod=30, voxel_size=0, update_init_factor=128,
        mask_lr_final=lambda l: 0.0001 * l / 0.001),
}


def build_jobs(datasets, lmbdas, iterations, fps_frames, output_base=OUTPUT_BASE, scenes=None):
    jobs = []
    for ds in datasets:
        cfg = DATASETS[ds]
        for lmbda in (lmbdas or cfg['lmbdas']):  # default: the lmbda of the upstream run_shell_*.py
            for scene in cfg['scenes']:
                if scenes and scene not in scenes:
                    continue
                model_path = f'{output_base}/{ds}/{scene}/{lmbda}'
                cmd = (f'python train.py -s {DATA_BASE}/{ds}/{scene} --eval --lod {cfg["lod"]} '
                       f'--voxel_size {cfg["voxel_size"]} --update_init_factor {cfg["update_init_factor"]} '
                       f'--iterations {iterations} -m {model_path} --lmbda {lmbda} '
                       f'--mask_lr_final {cfg["mask_lr_final"](lmbda):.8g} --fps_frames {fps_frames}')
                jobs.append((model_path, cmd))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS))
    # 0.004 / 0.0005 should correspond to the HAC++-lowrate / -highrate rows in results/
    p.add_argument('--scenes', nargs='+', default=None, help='only these scenes (default: all of --datasets)')
    p.add_argument('--tag', default='', help='separate run, e.g. "smoke": outputs/{EXP}_<tag>, tmp/*_<tag>.*')
    p.add_argument('--lmbdas', nargs='+', type=float, default=None,
                   help='override lmbda for all datasets (default: per-dataset value from run_shell_*.py)')
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--fps_frames', type=int, default=1000)
    p.add_argument('--partition', default='gpu-h200-141g-ellis')
    p.add_argument('--time', default='8:00:00')
    p.add_argument('--mem', default='128G')  # 64G got garden/flowers OOM-killed in ex3
    p.add_argument('--job_name', default=None, help='default HACplus_{EXP}[_<tag>]')
    p.add_argument('--run', action='store_true', help='submit with sbatch')
    args = p.parse_args()

    suffix = f'_{args.tag}' if args.tag else ''
    args.job_name = args.job_name or f'HACplus_{EXP}{suffix}'
    jobs = build_jobs(args.datasets, args.lmbdas, args.iterations, args.fps_frames,
                      output_base=OUTPUT_BASE + suffix, scenes=args.scenes)
    assert jobs, 'no jobs selected'
    tmp = Path(f'scripts/{EXP}/tmp')
    tmp.mkdir(parents=True, exist_ok=True)
    jobs_file = tmp / f'jobs{suffix}.txt'
    jobs_file.write_text('\n'.join(f'{mp}\t{cmd}' for mp, cmd in jobs) + '\n')

    script = f'''#!/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --job-name={args.job_name}
#SBATCH --output=scripts/{EXP}/sbatch_logs/{args.job_name}/%A_%a.log
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}
#SBATCH --partition={args.partition}
#SBATCH --account ellis_users
#SBATCH --array=0-{len(jobs) - 1}

source activate HACplus_env
hostname
nvidia-smi
echo "Running in partition: $SLURM_JOB_PARTITION, node: $SLURM_JOB_NODELIST"

# G-PCC codec for the anchor positions, symlinked into HACplus_env by build_tmc3.sh
command -v tmc3 || {{ echo "tmc3 not found in HACplus_env, run scripts/{EXP}/build_tmc3.sh first"; exit 1; }}

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {jobs_file})
MODEL_PATH=$(echo "$LINE" | cut -f1)
CMD=$(echo "$LINE" | cut -f2)
if [ -f "$MODEL_PATH/metrics.json" ]; then
    echo "$MODEL_PATH/metrics.json exists, skipping"; exit 0
fi
echo "$CMD"
CUDA_VISIBLE_DEVICES=0 $CMD
'''
    save_path = tmp / f'submit{suffix}.sh'
    save_path.write_text(script)
    print(f'{len(jobs)} jobs -> {jobs_file}\nsbatch script -> {save_path}')
    if args.run:
        import subprocess
        Path(f'scripts/{EXP}/sbatch_logs/{args.job_name}').mkdir(parents=True, exist_ok=True)
        subprocess.run(['sbatch', str(save_path)], check=True)


if __name__ == '__main__':
    main()
