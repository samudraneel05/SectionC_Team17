"""Scale-resolution 2x2: is optimal R set by voxels-per-structure or noise?

crop native-res 128^3 cube from the 256^3 P50 CGLS recon -> FoJ R in {6,10,12}.
Prediction under scale-relative hypothesis: R12 wins on native-res crop
(features are 2x larger in voxels), opposite of downsampled 128^3 where R6 won.
GT crop = central 128^3 of vol_gt.
"""
import json, time, sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, '/content/3D-Field-of-Junctions')
import field_of_junctions3d as foj_module
foj_module.dev = torch.device('cuda')
import chunked_foj, metrics

BASE = dict(stride=2, eta=0.01, delta=0.05, lr_angles=0.005, lr_x0y0z0=0.05,
            lambda_boundary_final=0.1, lambda_color_final=0.001, nvals=5,
            num_initialization_iters=1, num_refinement_iters=1,
            greedy_step_every_iters=50, parallel_mode=True)

def mkopts(R):
    o = dict(BASE); o['R'] = R
    return SimpleNamespace(**o)

clean256 = np.load('examples/image_of_engine/cone_ntrain_25_angle_360/2_engine_cone/vol_gt.npy').astype(np.float32)
rec256 = np.load('recon256_P50.npy')
c = 64                      # central 128^3 crop
clean = clean256[c:c + 128, c:c + 128, c:c + 128]
rec = rec256[c:c + 128, c:c + 128, c:c + 128]

res = {'input_psnr': metrics.psnr(clean, rec),
       'input_ssim': metrics.ssim3d(clean, rec)}
print(f'native-res crop input: {res["input_psnr"]:.2f} dB', flush=True)

for R in (6, 10, 12):
    t0 = time.time()
    out, _ = chunked_foj.chunked_foj(rec, mkopts(R), chunk_patches=13,
                                     halo=R, verbose=False,
                                     checkpoint_path=f'ckpt_crop_R{R}.npz')
    res[f'R{R}'] = dict(psnr=metrics.psnr(clean, out),
                        ssim3d=metrics.ssim3d(clean, out),
                        bf=metrics.boundary_fscore(clean, out)[0],
                        s=time.time() - t0)
    np.save(f'crop256_P50_R{R}.npy', out)
    print(f'R={R}: {res[f"R{R}"]["psnr"]:.2f} dB  '
          f'SSIM {res[f"R{R}"]["ssim3d"]:.3f}  ({res[f"R{R}"]["s"]:.0f}s)',
          flush=True)
    json.dump(res, open('exp_crop_scale.json', 'w'), indent=2)
print('DONE')
