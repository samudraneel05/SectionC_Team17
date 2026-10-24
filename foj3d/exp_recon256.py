"""Definitive paper reproduction at native 256^3.

vol_gt 256^3 -> cone-beam (25 views, det 512^2, meta_data geometry)
-> Poisson P_n=50 at detector -> CGLS 15 iters -> FoJ R in {6,10,12} chunked.

Scale-relative hypothesis test: paper reports R12>R10>R6 at P50/256^3;
at 128^3 we observe the opposite (R6 best). If 256^3 reproduces R12>R10>R6,
the optimal R is ~N/21 voxels, i.e., scale-relative, not noise-adaptive.
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

DS, DET, N_VIEW, N_CGLS, P_N = 256, 512, 25, 15, 50.0
RESULTS = 'exp_recon256.json'

def save(res):
    json.dump(res, open(RESULTS, 'w'), indent=2)

clean = np.load('examples/image_of_engine/cone_ntrain_25_angle_360/2_engine_cone/vol_gt.npy').astype(np.float32)
vol = torch.from_numpy(clean)[None, None].cuda()
angles = np.linspace(0, 2 * np.pi, N_VIEW, endpoint=False)

results = {}
if os.path.exists(RESULTS):
    results = json.load(open(RESULTS))
t0 = time.time()
proj_clean = pj.forward_project(vol, angles, det_hw=DET)
print(f'projections {tuple(proj_clean.shape)} in {time.time()-t0:.0f}s '
      f'max {proj_clean.max():.2f}', flush=True)
torch.save(proj_clean.cpu(), 'proj256_clean.pt')

phat, counts = pj.add_poisson_noise(proj_clean, P_N, seed=0)
del proj_clean, counts
torch.cuda.empty_cache()

t0 = time.time()
rec = pj.cgls(phat.cuda(), angles, (DS, DS, DS), det_hw=DET, n_iters=N_CGLS,
              verbose=True)
recn = rec[0, 0].cpu().numpy().astype(np.float32)
np.save('recon256_P50.npy', recn)
results['cgls_time_s'] = time.time() - t0
results['input_psnr'] = metrics.psnr(clean, recn)
results['input_ssim'] = metrics.ssim3d(clean, recn)
print(f'\nrecon PSNR {results["input_psnr"]:.2f} dB '
      f'({results["cgls_time_s"]:.0f}s)\n', flush=True)
save(results)
del rec, phat
torch.cuda.empty_cache()

for R in (12, 6, 10):          # R12 first — hypothesis-critical ordering
    out_path = f'foj256_P50_R{R}.npy'
    if os.path.exists(out_path):
        out = np.load(out_path)
        results[f'R{R}'] = dict(psnr=metrics.psnr(clean, out),
                                ssim3d=metrics.ssim3d(clean, out),
                                bf=metrics.boundary_fscore(clean, out)[0])
        save(results)
        print(f'R={R} already done, re-scored', flush=True)
        continue
    t0 = time.time()
    out, _ = chunked_foj.chunked_foj(
        recn, mkopts(R), chunk_patches=13, halo=R, verbose=True,
        checkpoint_path=f'ckpt256_R{R}.npz', ckpt_every=25)
    np.save(out_path, out)
    if os.path.exists(f'ckpt256_R{R}.npz'):
        os.remove(f'ckpt256_R{R}.npz')
    results[f'R{R}'] = dict(psnr=metrics.psnr(clean, out),
                            ssim3d=metrics.ssim3d(clean, out),
                            bf=metrics.boundary_fscore(clean, out)[0],
                            s=time.time() - t0,
                            peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    print(f'\nR={R}: PSNR {results[f"R{R}"]["psnr"]:.2f} '
          f'SSIM {results[f"R{R}"]["ssim3d"]:.3f} '
          f'({results[f"R{R}"]["s"]:.0f}s, {results[f"R{R}"]["peak_vram_gb"]:.1f}GB)',
          flush=True)
    save(results)

print('\nDONE')
