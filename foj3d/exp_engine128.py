"""First real experiment matrix: engine 128^3, quadrants photon field.

Runs: noisy | global R in {6,10,12} | fusion | adaptive_r
Saves metrics JSON + npz of all volumes. Designed for Colab (cwd=repo root).
"""
import json, time, sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import zoom

sys.path.insert(0, '/content/3D-Field-of-Junctions')
import field_of_junctions3d as foj_module
foj_module.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import chunked_foj, noise_fields, noise_estimator, metrics

BASE = dict(stride=2, eta=0.01, delta=0.05, lr_angles=0.005, lr_x0y0z0=0.05,
            lambda_boundary_final=0.1, lambda_color_final=0.001, nvals=5,
            num_initialization_iters=1, num_refinement_iters=1,
            greedy_step_every_iters=50, parallel_mode=True)

def mkopts(R):
    o = dict(BASE); o['R'] = R
    return SimpleNamespace(**o)

def r_for_q(q):
    return 6 if q < 0.33 else (10 if q < 0.66 else 12)

def main():
    v = np.load('examples/image_of_engine/cone_ntrain_25_angle_360/2_engine_cone/vol_gt.npy')
    clean = zoom(v.astype(np.float32), 0.5, order=1)          # 128^3
    clean = np.clip(clean, 0, 1)
    np.save('engine128_clean.npy', clean)

    P = noise_fields.photon_field(clean.shape, 'quadrants')
    noisy = noise_fields.add_noise(clean, P, seed=0)
    sigma_true = noise_fields.true_sigma(clean, P)

    t0 = time.time()
    sigma_hat = noise_estimator.estimate_sigma_voxel(noisy)
    q_map = noise_estimator.normalize01(sigma_hat)
    # estimator sanity: correlation with true sigma
    m = sigma_true > 1e-3
    corr = float(np.corrcoef(sigma_hat[m], sigma_true[m])[0, 1])
    print(f'[est] sigma_hat/true corr={corr:.3f} ({time.time()-t0:.0f}s)', flush=True)

    results = {}
    outs = {'noisy': noisy}
    results['noisy'] = dict(psnr=metrics.psnr(clean, noisy),
                            ssim3d=metrics.ssim3d(clean, noisy),
                            bf=metrics.boundary_fscore(clean, noisy)[0],
                            zones=metrics.per_zone_metrics(clean, noisy, P),
                            runtime_s=0.0)
    print('[noisy]', json.dumps(results['noisy']), flush=True)

    for R in (6, 10, 12):
        t0 = time.time()
        out, _ = chunked_foj.chunked_foj(noisy, mkopts(R), chunk_patches=16,
                                         halo=R, verbose=False)
        outs[f'R{R}'] = out
        results[f'R{R}'] = dict(
            psnr=metrics.psnr(clean, out), ssim3d=metrics.ssim3d(clean, out),
            bf=metrics.boundary_fscore(clean, out)[0],
            zones=metrics.per_zone_metrics(clean, out, P),
            runtime_s=time.time() - t0,
            peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9)
        torch.cuda.reset_peak_memory_stats()
        print(f'[R{R}]', json.dumps(results[f'R{R}']), flush=True)

    # fusion of the three global runs with sigma-derived weights
    t0 = time.time()
    outs3 = np.stack([outs['R6'], outs['R10'], outs['R12']])
    score = np.stack([1.0 / (q_map + 0.1), np.full_like(q_map, 1.0), q_map + 0.1])
    w = np.exp(3.0 * score); w /= w.sum(0, keepdims=True)
    fused = np.clip((w * outs3).sum(0), 0, 1)
    outs['fusion'] = fused
    results['fusion'] = dict(
        psnr=metrics.psnr(clean, fused), ssim3d=metrics.ssim3d(clean, fused),
        bf=metrics.boundary_fscore(clean, fused)[0],
        zones=metrics.per_zone_metrics(clean, fused, P),
        runtime_s=sum(results[f'R{R}']['runtime_s'] for R in (6, 10, 12)))
    print('[fusion]', json.dumps(results['fusion']), flush=True)

    # adaptive_r: per-chunk R from median sigma quantile of the chunk
    t0 = time.time()
    sg = noise_estimator.sigma_to_patch_grid(q_map, 10, BASE['stride'])
    def opts_fn(pz, py, px):
        sl = sg[pz:pz + 16, py:py + 16, px:px + 16]
        q = float(np.median(sl)) if sl.size else 0.5
        return mkopts(r_for_q(q))
    out, bnd = chunked_foj.chunked_foj(noisy, mkopts(10), chunk_patches=16,
                                       opts_fn=opts_fn, halo=12, verbose=True)
    outs['adaptive_r'] = out
    results['adaptive_r'] = dict(
        psnr=metrics.psnr(clean, out), ssim3d=metrics.ssim3d(clean, out),
        bf=metrics.boundary_fscore(clean, out)[0],
        zones=metrics.per_zone_metrics(clean, out, P),
        runtime_s=time.time() - t0,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9)
    print('[adaptive_r]', json.dumps(results['adaptive_r']), flush=True)

    results['sigma_corr'] = corr
    np.savez_compressed('exp_engine128_quadrants.npz',
                        clean=clean, noisy=noisy, P=P,
                        sigma_true=sigma_true, sigma_hat=sigma_hat,
                        **{k: v for k, v in outs.items() if k != 'noisy'})
    json.dump(results, open('exp_engine128_quadrants.json', 'w'), indent=2)
    print('DONE -> exp_engine128_quadrants.{npz,json}')

if __name__ == '__main__':
    main()
