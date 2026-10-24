"""Spatially-varying PROJECTION-domain noise -> spatially-varying recon noise.

Detector split: left half P=50 photons/px, right half P=1000. Rays from the
low-count side traverse the volume half they face more obliquely -> the recon
noise becomes spatially structured. Then global R-sweep + per-region analysis
to see whether different volume regions prefer different patch sizes.

Same 128^3 engine setup as exp_recon128.
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
import projector as pj

BASE = dict(stride=2, eta=0.01, delta=0.05, lr_angles=0.005, lr_x0y0z0=0.05,
            lambda_boundary_final=0.1, lambda_color_final=0.001, nvals=5,
            num_initialization_iters=1, num_refinement_iters=1,
            greedy_step_every_iters=50, parallel_mode=True)

def mkopts(R):
    o = dict(BASE); o['R'] = R
    return SimpleNamespace(**o)

DS, DET, N_VIEW, N_CGLS = 128, 256, 25, 15

v = np.load('examples/image_of_engine/cone_ntrain_25_angle_360/2_engine_cone/vol_gt.npy')
clean = np.clip(zoom(v.astype(np.float32), DS / 256.0, order=1), 0, 1)
vol = torch.from_numpy(clean)[None, None].cuda()
angles = np.linspace(0, 2 * np.pi, N_VIEW, endpoint=False)

proj_clean = pj.forward_project(vol, angles, det_hw=DET)

# detector photon field: left half severe, right half mild
Pn = torch.full((N_VIEW, DET, DET), 1000.0)
Pn[:, :, : DET // 2] = 50.0
phat, counts = pj.add_poisson_noise(proj_clean, Pn, seed=0)

t0 = time.time()
rec = pj.cgls(phat.cuda(), angles, (DS, DS, DS), det_hw=DET, n_iters=N_CGLS,
              verbose=False)
recn = rec[0, 0].cpu().numpy()
np.save('recon128_spatialPn.npy', recn)
print(f'recon done ({time.time()-t0:.0f}s)  PSNR {metrics.psnr(clean, recn):.2f}',
      flush=True)

# where is the recon error concentrated?
err = np.abs(recn - clean)
D = DS
zones = {
    'x<64':  (slice(None), slice(None), slice(0, D // 2)),
    'x>=64': (slice(None), slice(None), slice(D // 2, D)),
}
results = {'input_psnr': metrics.psnr(clean, recn),
           'err_by_zone': {k: float(err[s].mean()) for k, s in zones.items()}}
print('err by zone:', results['err_by_zone'], flush=True)

outs = {}
for R in (6, 10, 12):
    t0 = time.time()
    out, _ = chunked_foj.chunked_foj(recn, mkopts(R), chunk_patches=13,
                                     halo=R, verbose=False)
    outs[R] = out
    row = {'psnr': metrics.psnr(clean, out),
           'ssim3d': metrics.ssim3d(clean, out),
           'bf': metrics.boundary_fscore(clean, out)[0],
           's': time.time() - t0}
    for k, s in zones.items():
        row[f'psnr_{k}'] = metrics.psnr(clean[s], out[s])
    results[f'R{R}'] = row
    print(f'R={R}:', {k: round(v, 2) for k, v in row.items()}, flush=True)

# oracle spatial fusion: per-voxel best of the three runs -> headroom bound
stack = np.stack([outs[R] for R in (6, 10, 12)])
mse = (stack - clean[None]) ** 2
oracle = stack[np.argmin(mse, axis=0), *np.meshgrid(*[np.arange(D)] * 3,
                                                 indexing='ij').__getitem__(slice(0, 3))]
# simpler oracle: per-region choice via index trick
idx = np.argmin(mse, axis=0)
oracle = np.take_along_axis(stack, idx[None], axis=0)[0]
results['oracle_fusion'] = dict(psnr=metrics.psnr(clean, oracle),
                                ssim3d=metrics.ssim3d(clean, oracle))
print('oracle per-voxel fusion PSNR:', round(results['oracle_fusion']['psnr'], 2))
print('R-choice histogram:', dict(zip(*np.unique(idx, return_counts=True))))

json.dump(results, open('exp_recon_spatial.json', 'w'), indent=2)
np.savez_compressed('exp_recon_spatial.npz', clean=clean, recon=recn,
                    R6=outs[6], R10=outs[10], R12=outs[12], oracle=oracle,
                    Pn_field=Pn.numpy())
print('DONE')
