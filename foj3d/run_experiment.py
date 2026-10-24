"""Experiment driver: clean volume + photon field -> noisy -> FoJ -> metrics.

Usage:
  python run_experiment.py --vol engine.npy --pattern quadrants --mode adaptive_r
Modes:
  global     : one fixed R for the whole volume
  fusion     : run several global R, fuse outputs with sigma-derived weights
  adaptive_r : per-chunk R chosen from local sigma (zone-wise grids)
"""
import argparse, json, time
from types import SimpleNamespace

import numpy as np
import torch

import field_of_junctions3d as foj_module
foj_module.dev = torch.device(
    'mps' if torch.backends.mps.is_available() else
    ('cuda' if torch.cuda.is_available() else 'cpu'))

import chunked_foj, noise_fields, noise_estimator, metrics

BASE = dict(stride=2, eta=0.01, delta=0.05,
            lr_angles=0.005, lr_x0y0z0=0.05,
            lambda_boundary_final=0.1, lambda_color_final=0.001,
            nvals=5, num_initialization_iters=1, num_refinement_iters=1,
            greedy_step_every_iters=50, parallel_mode=True)

# sigma-quantile -> patch size (paper ablation: R grows with noise)
def r_for_q(q):
    return 6 if q < 0.33 else (10 if q < 0.66 else 12)


def make_opts(R):
    o = dict(BASE); o['R'] = R
    return SimpleNamespace(**o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vol', required=True, help='.npy clean volume [D,H,W]')
    ap.add_argument('--pattern', default='quadrants')
    ap.add_argument('--mode', default='global',
                    choices=['global', 'fusion', 'adaptive_r'])
    ap.add_argument('--R', type=int, default=10, help='global-mode patch size')
    ap.add_argument('--chunk', type=int, default=13)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='result.npz')
    args = ap.parse_args()

    clean = np.load(args.vol).astype(np.float32)
    if clean.ndim == 4:
        clean = clean[..., 0]
    clean = np.clip(clean, 0, 1)

    P = noise_fields.photon_field(clean.shape, args.pattern)
    noisy = noise_fields.add_noise(clean, P, seed=args.seed)
    sigma_true = noise_fields.true_sigma(clean, P)

    t0 = time.time()
    sigma_hat = noise_estimator.estimate_sigma_voxel(noisy)
    q_map = noise_estimator.normalize01(sigma_hat)          # per-voxel reliability
    print(f'noise map: {time.time()-t0:.1f}s; '
          f'sigma_hat med={np.median(sigma_hat):.4f}')

    t0 = time.time()
    if args.mode == 'global':
        out, bnd = chunked_foj.chunked_foj(noisy, make_opts(args.R),
                                           chunk_patches=args.chunk)
        out = out[..., 0]
    elif args.mode == 'fusion':
        outs = []
        for R in (6, 10, 12):
            o, _ = chunked_foj.chunked_foj(noisy, make_opts(R),
                                           chunk_patches=args.chunk)
            outs.append(o[..., 0])
        outs = np.stack(outs)
        # softmax weights: low-sigma favors R=6, high-sigma favors R=12
        score = np.stack([1.0 / (q_map + 0.1),          # R=6
                          np.full_like(q_map, 1.0),     # R=10
                          q_map + 0.1])                 # R=12
        w = np.exp(3.0 * score)
        w /= w.sum(0, keepdims=True)
        out = (w * outs).sum(0)
        bnd = np.zeros_like(out)
    elif args.mode == 'adaptive_r':
        # patch-grid sigma for chunk-wise R selection
        sg = noise_estimator.sigma_to_patch_grid(q_map, 10, BASE['stride'])
        sg_d = sg.shape[0]  # quantile label per patch-index

        def opts_fn(pz, py, px):
            # majority zone of this chunk's patch grid
            sl = sg[pz:pz + args.chunk, py:py + args.chunk, px:px + args.chunk]
            q = float(np.median(sl)) if sl.size else 0.5
            return make_opts(r_for_q(q))

        out, bnd = chunked_foj.chunked_foj(noisy, make_opts(args.R),
                                           chunk_patches=args.chunk,
                                           opts_fn=opts_fn)
        out = out[..., 0]
    dt = time.time() - t0

    res = {
        'mode': args.mode, 'pattern': args.pattern, 'R': args.R,
        'psnr_in': metrics.psnr(clean, noisy),
        'psnr': metrics.psnr(clean, out),
        'ssim3d': metrics.ssim3d(clean, out),
        'bf': metrics.boundary_fscore(clean, out)[0],
        'runtime_s': dt,
        'zones': metrics.per_zone_metrics(clean, out, P),
    }
    print(json.dumps(res, indent=2))
    np.savez_compressed(args.out, clean=clean, noisy=noisy, out=out,
                        P=P, sigma_true=sigma_true, sigma_hat=sigma_hat,
                        boundary=bnd, metrics=res)
    print('saved ->', args.out)


if __name__ == '__main__':
    main()
