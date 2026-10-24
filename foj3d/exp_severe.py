"""Crossover hunt: does any noise level make large R beat small R?

Left half of volume gets P_severe photons/voxel, right half P=30 (mild).
Sweep P_severe in {1.0, 1.5, 2.0}; global R in {6, 8, 10, 12}.
If severe-half prefers R>6 at some level -> adaptive-R has headroom.
"""
import json, time, sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import zoom

sys.path.insert(0, '/content/3D-Field-of-Junctions')
import field_of_junctions3d as foj_module
foj_module.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import chunked_foj, metrics

BASE = dict(stride=2, eta=0.01, delta=0.05, lr_angles=0.005, lr_x0y0z0=0.05,
            lambda_boundary_final=0.1, lambda_color_final=0.001, nvals=5,
            num_initialization_iters=1, num_refinement_iters=1,
            greedy_step_every_iters=50, parallel_mode=True)

def mkopts(R):
    o = dict(BASE); o['R'] = R
    return SimpleNamespace(**o)

v = np.load('examples/image_of_engine/cone_ntrain_25_angle_360/2_engine_cone/vol_gt.npy')
clean = np.clip(zoom(v.astype(np.float32), 0.5, order=1), 0, 1)
D, H, W = clean.shape
half = np.zeros(clean.shape, np.float32); half[:, :, W // 2:] = 1.0   # right = mild
left = 1.0 - half
rng = np.random.default_rng(0)

results = {}
for P_sev in (1.0, 1.5, 2.0):
    P = left * P_sev + half * 30.0
    noisy = np.clip(rng.poisson(clean.astype(np.float64) * P) / P, 0, 1).astype(np.float32)
    row = {'in_left': metrics.psnr(clean[left > 0], noisy[left > 0]),
           'in_right': metrics.psnr(clean[half > 0], noisy[half > 0])}
    print(f'\n=== P_severe={P_sev}  input L/R = {row["in_left"]:.2f} / {row["in_right"]:.2f} dB',
          flush=True)
    for R in (6, 8, 10, 12):
        t0 = time.time()
        out, _ = chunked_foj.chunked_foj(noisy, mkopts(R), chunk_patches=13,
                                         halo=R, verbose=False)
        row[f'R{R}_left'] = metrics.psnr(clean[left > 0], out[left > 0])
        row[f'R{R}_right'] = metrics.psnr(clean[half > 0], out[half > 0])
        row[f'R{R}_all'] = metrics.psnr(clean, out)
        row[f'R{R}_s'] = time.time() - t0
        print(f'  R={R}: overall {row[f"R{R}_all"]:.2f} | '
              f'severe {row[f"R{R}_left"]:.2f} | mild {row[f"R{R}_right"]:.2f} '
              f'({row[f"R{R}_s"]:.0f}s)', flush=True)
        results[f'P{P_sev}'] = row

json.dump(results, open('exp_severe_half.json', 'w'), indent=2)
print('\nDONE')
