"""STRICT paper-protocol reproduction: engine 256^3, det 256^2, 20 non-uniform
views, projection-domain Poisson P50, CGLS -> FoJ R in {4,6,8,10,12}.

Reproduces the shape of paper Table 3 verbatim. Checkpointed per-25-chunks;
skips finished work on restart (safe across session preemption).
Non-uniform angles: seeded stratified-jitter, recorded in results JSON.
"""
import json, os, time, sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, '/content/3D-Field-of-Junctions')
import field_of_junctions3d as foj_module
foj_module.dev = torch.device('cuda')
import chunked_foj, metrics
import projector as pj

BASE = dict(stride=2, eta=0.01, delta=0.05, lr_angles=0.005, lr_x0y0z0=0.05,
            lambda_boundary_final=0.1, lambda_color_final=0.001, nvals=5,
            num_initialization_iters=1, num_refinement_iters=1,
            greedy_step_every_iters=50, parallel_mode=True)

def mkopts(R):
    o = dict(BASE); o['R'] = R
    return SimpleNamespace(**o)

DS, DET, N_VIEW, N_CGLS, P_N = 256, 256, 20, 15, 50.0
RESULTS = 'exp_strict256.json'

def save(res):
    json.dump(res, open(RESULTS, 'w'), indent=2)

results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}

# --- 20 non-uniform views: stratified jitter, seed 0, recorded in JSON ---
rng = np.random.default_rng(0)
base = np.linspace(0, 2 * np.pi, N_VIEW, endpoint=False)
jit = rng.uniform(-0.45, 0.45, N_VIEW) * (2 * np.pi / N_VIEW)
angles = np.sort((base + jit) % (2 * np.pi))
results['angles_rad'] = [round(float(a), 4) for a in angles]

clean = np.load('examples/image_of_engine/cone_ntrain_25_angle_360/2_engine_cone/vol_gt.npy').astype(np.float32)

# --- projections + noisy CGLS recon (skipped if already on disk) ---
if os.path.exists('recon256_strict_P50.npy'):
    recn = np.load('recon256_strict_P50.npy')
    print('recon already on disk, skipping pipeline', flush=True)
else:
    vol = torch.from_numpy(clean)[None, None].cuda()
    t0 = time.time()
    proj_clean = pj.forward_project(vol, angles, det_hw=DET)
    print(f'projections {tuple(proj_clean.shape)} {time.time()-t0:.0f}s '
          f'max {proj_clean.max():.2f}', flush=True)
    phat, _ = pj.add_poisson_noise(proj_clean, P_N, seed=0)
    del proj_clean
    t0 = time.time()
    rec = pj.cgls(phat.cuda(), angles, (DS, DS, DS), det_hw=DET,
                  n_iters=N_CGLS, verbose=True)
    recn = rec[0, 0].cpu().numpy().astype(np.float32)
    np.save('recon256_strict_P50.npy', recn)
    results['cgls_time_s'] = time.time() - t0
    del rec, phat, vol
    torch.cuda.empty_cache()

results['input_psnr'] = metrics.psnr(clean, recn)
results['input_ssim'] = metrics.ssim3d(clean, recn)
print(f'recon PSNR {results["input_psnr"]:.2f} dB', flush=True)
save(results)

for R in (12, 6, 10, 8, 4):          # decisive order first
    out_path = f'foj256_strictP50_R{R}.npy'
    if os.path.exists(out_path):
        out = np.load(out_path)
    else:
        t0 = time.time()
        out, _ = chunked_foj.chunked_foj(
            recn, mkopts(R), chunk_patches=13, halo=R, verbose=True,
            checkpoint_path=f'ckpt_strict_R{R}.npz', ckpt_every=25)
        np.save(out_path, out)
        os.remove(f'ckpt_strict_R{R}.npz')
        results[f'R{R}_s'] = time.time() - t0
    results[f'R{R}'] = dict(psnr=metrics.psnr(clean, out),
                            ssim3d=metrics.ssim3d(clean, out),
                            bf=metrics.boundary_fscore(clean, out)[0])
    print(f'R={R}: PSNR {results[f"R{R}"]["psnr"]:.2f} '
          f'SSIM {results[f"R{R}"]["ssim3d"]:.3f}', flush=True)
    save(results)

print('\nDONE')
