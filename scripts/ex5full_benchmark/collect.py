"""Gather outputs/ex5full_benchmark/*/*/*/metrics.json, compare with the paper numbers in results/.

    python scripts/ex5full_benchmark/collect.py [outputs/ex5full_benchmark_smoke]
writes outputs/ex5full_benchmark/summary.csv and prints per-scene + per-dataset tables.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('outputs/ex5full_benchmark')
RESULTS = Path('results')
RESULTS_DIR = {'mipnerf360': 'MipNeRF360', 'db': 'DeepBlending', 'tandt': 'TanksAndTemples',
               'nerf_synthetic': 'SyntheticNeRF'}  # no bungeenerf numbers in results/
# which lmbda produced the paper's lowrate/highrate rows is not stated, so show both
PAPER_ROWS = {'low': 'HAC++-lowrate', 'high': 'HAC++-highrate'}
PAPER_KEYS = [f'paper_{t}_{k}' for t in PAPER_ROWS for k in ('PSNR', 'SSIM', 'LPIPS', 'size_MB', 'num_gaussians')]
FIELDS = ['dataset', 'scene', 'lmbda', 'PSNR', 'SSIM', 'LPIPS', 'size_MB', 'num_active_gaussians',
          'training_time_s', 'fps', 'gpu'] + PAPER_KEYS


def paper_numbers(ds, scene):
    f = RESULTS / RESULTS_DIR.get(ds, '_') / f'{scene}.csv'
    if not f.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(f)):
        for t, name in PAPER_ROWS.items():
            if r['Submethod'] == name:
                out.update({f'paper_{t}_PSNR': float(r['PSNR']), f'paper_{t}_SSIM': float(r['SSIM']),
                            f'paper_{t}_LPIPS': float(r['LPIPS']),
                            f'paper_{t}_size_MB': float(r['Size [Bytes]']) / 2**20,  # same MB (2**20) as size.json
                            f'paper_{t}_num_gaussians': int(r['#Gaussians'])})
    return out


def fmt(v, n=3):
    return '-' if v is None else (f'{v:.{n}f}' if isinstance(v, float) else str(v))


def main():
    if not ROOT.is_dir():
        sys.exit(f'{ROOT} does not exist yet (no runs started)')
    rows, missing = [], []
    for d in sorted(ROOT.glob('*/*/*')):
        if not d.is_dir():
            continue
        ds, scene, lmbda = d.parts[-3], d.parts[-2], float(d.parts[-1])
        mfile = d / 'metrics.json'
        if not mfile.exists():
            missing.append(str(d))
            continue
        m = json.load(open(mfile))
        row = {'dataset': ds, 'scene': scene, 'lmbda': lmbda, **{k: m.get(k) for k in FIELDS if k in m}}
        row.update(paper_numbers(ds, scene))
        rows.append(row)

    with open(ROOT / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    cols = ['dataset', 'scene', 'lmbda', 'PSNR', 'paper_low_PSNR', 'paper_high_PSNR', 'SSIM', 'LPIPS',
            'size_MB', 'paper_low_size_MB', 'paper_high_size_MB', 'training_time_s', 'fps']
    print(' | '.join(cols))
    for r in rows:
        print(' | '.join(fmt(r.get(c)) for c in cols))

    print('\nPer-dataset mean (scenes that finished):')
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r['lmbda'])].append(r)
    for (ds, lmbda), rs in sorted(groups.items()):
        mean = lambda k: (sum(r[k] for r in rs if r.get(k) is not None) / max(1, sum(r.get(k) is not None for r in rs)))
        print(f'{ds:15s} lmbda={lmbda:<7} n={len(rs):2d}  PSNR {mean("PSNR"):.2f} '
              f'(paper low/high {mean("paper_low_PSNR"):.2f}/{mean("paper_high_PSNR"):.2f})  '
              f'SSIM {mean("SSIM"):.3f}  LPIPS {mean("LPIPS"):.3f}  size {mean("size_MB"):.2f}MB '
              f'(paper low/high {mean("paper_low_size_MB"):.2f}/{mean("paper_high_size_MB"):.2f})  '
              f'train {mean("training_time_s")/60:.1f}min  fps {mean("fps"):.0f}')

    if missing:
        print(f'\n{len(missing)} runs without metrics.json (unfinished or crashed):')
        print('\n'.join('  ' + m for m in missing))
    print(f'\nwrote {ROOT / "summary.csv"}')


if __name__ == '__main__':
    main()
