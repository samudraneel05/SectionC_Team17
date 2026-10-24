"""Paper-faithful reproduction at 128^3:
vol_gt -> cone-beam projections (25 views, det 256^2, geometry from
meta_data.json scaled) -> projection-domain Poisson noise (P_n photons/pixel)
-> CGLS recon -> FoJ R-sweep.

Question: does the R-trend under *reconstruction* noise differ from iid
voxel noise (where R6 dominated at all severities)?
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

DS = 128
DET = 256          # 2:1 det:vol ratio, matching paper 512:256
N_VIEW = 25
N_CGLS = 15
P_LEVELS = [50.0, 500.0]     # severe + mild projection-domain photon counts

v = np.load('examples/image_of_engine/cone_ntrain_25_angle_360/2_engine_cone/vol_gt.npy')
clean = np.clip(zoom(v.astype(np.float32), DS / 256.0, order=1), 0, 1)
vol = torch.from_numpy(clean)[None, None].cuda()
angles = np.linspace(0, 2 * np.pi, N_VIEW, endpoint=False)

print('== clean projections ==', flush=True)
t0 = time.time()
proj_clean = pj.forward_project(vol, angles, det_hw=DET)
print(f'   {time.time()-t0:.0f}s  proj {tuple(proj_clean.shape)} max {proj_clean.max():.3f}',
      flush=True)

results = {}
# noiseless recon sanity
t0 = time.time()
rec0 = pj.cgls(proj_clean, angles, (DS, DS, DS), det_hw=DET, n_iters=N_CGLS,
               verbose=False)
rec0n = rec0[0, 0].cpu().numpy()
print(f'noiseless CGLS: PSNR {metrics.psnr(clean, rec0n):.2f} dB '
      f'({time.time()-t0:.0f}s)', flush=True)
results['cgls_clean_psnr'] = metrics.psnr(clean, rec0n)
np.save('recon128_clean.npy', rec0n)

for P_n in P_LEVELS:
    phat, counts = pj.add_poisson_noise(proj_clean, P_n, seed=0)
    t0 = time.time()
    rec = pj.cgls(phat.cuda(), angles, (DS, DS, DS), det_hw=DET,
                  n_iters=N_CGLS, verbose=False)
    recn = rec[0, 0].cpu().numpy()
    in_psnr = metrics.psnr(clean, recn)
    print(f'\n=== P_n={P_n}: recon PSNR {in_psnr:.2f} dB '
          f'({time.time()-t0:.0f}s) ===', flush=True)
    np.save(f'recon128_P{int(P_n)}.npy', recn)
    row = {'input_psnr': in_psnr,
           'input_ssim': metrics.ssim3d(clean, recn)}
    for R in (6, 10, 12):
        t0 = time.time()
        out, _ = chunked_foj.chunked_foj(recn, mkopts(R), chunk_patches=13,
                                         halo=R, verbose=False)
        row[f'R{R}'] = dict(psnr=metrics.psnr(clean, out),
                            ssim3d=metrics.ssim3d(clean, out),
                            bf=metrics.boundary_fscore(clean, out)[0],
                            s=time.time() - t0)
        torch.cuda.empty_cache()
        print(f'  R={R}: PSNR {row[f"R{R}"]["psnr"]:.2f} '
              f'SSIM {row[f"R{R}"]["ssim3d"]:.3f} ({row[f"R{R}"]["s"]:.0f}s)',
              flush=True)
    results[f'Pn{int(P_n)}'] = row
    json.dump(results, open('exp_recon128.json', 'w'), indent=2)

print('\nDONE')
