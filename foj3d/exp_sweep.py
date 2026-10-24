"""Crop-interior R sweep over CGLS iteration count (effective noise level).

Generalizes exp_cgls_sweep.py over object, noise seed and input clamping:
  --clamp 1 : FoJ input = clip(CGLS, 0, 1)   (our pipeline so far)
  --clamp 0 : FoJ input = raw CGLS iterate    (hypothesized paper pipeline)
FoJ output is scored both unclipped and clipped to [0,1].

Crop: 128^3 centred on the GT centre of mass (voxels > 0.05), shifted to stay
inside the volume. Exact chunking (halo=0, no per-block clip): by patch
separability at init=1/refine=1 (verified in exp_assembly), voxels at distance
>= R-1 from the crop border are identical to a full-volume run, so PSNR is
computed on the common interior [12:116]^3 (margin 12 >= max R - 1).

usage: python exp_sweep.py --obj engine --seed 0 --clamp 0 --iters 5,10,15
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, '/content/3D-Field-of-Junctions')
import field_of_junctions3d as foj_module
foj_module.dev = torch.device('cuda')
import chunked_foj, metrics
from strict_protocol import mkopts, load_gt, recon_snapshots

ap = argparse.ArgumentParser()
ap.add_argument('--obj', default='engine')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--clamp', type=int, default=1)
ap.add_argument('--iters', default='5,10,15,25,40,60,100')
ap.add_argument('--rs', default='4,6,8,10,12')
ap.add_argument('--P', type=float, default=50.0)
ap.add_argument('--backend', default='tigre')
ap.add_argument('--angles', default='jitter')
args = ap.parse_args()
ITERS = [int(x) for x in args.iters.split(',')]
RS = [int(x) for x in args.rs.split(',')]

OUT = (f'sweep_{args.backend}_{args.angles}_{args.obj}_P{int(args.P)}'
       f'_s{args.seed}_clamp{args.clamp}.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

gt = load_gt(args.obj)
assert gt.shape == (256, 256, 256), gt.shape
res['gt_min_max'] = [float(gt.min()), float(gt.max())]

com = np.array(np.nonzero(gt > 0.05)).mean(1).round().astype(int)
c0 = np.clip(com - 64, 0, 256 - 128)
crop = tuple(slice(int(a), int(a) + 128) for a in c0)
inner = (slice(12, 116),) * 3
res['crop_origin'] = [int(a) for a in c0]
gtc = gt[crop]
gti = gtc[inner]

snaps = recon_snapshots(ITERS, P_n=args.P, seed=args.seed, obj=args.obj,
                        backend=args.backend, angles=args.angles)

for k in ITERS:
    ent = res.setdefault(f'cg{k}', {})
    raw = snaps[k]
    inp = np.clip(raw, 0, 1) if args.clamp else raw
    ent['input_psnr_full_clamped'] = metrics.psnr(gt, np.clip(raw, 0, 1))
    ent['input_psnr_full_unclamped'] = metrics.psnr(gt, raw)
    inpc = inp[crop]
    ent['input_psnr_inner'] = metrics.psnr(gti, inpc[inner])
    print(f'[{args.backend} {args.angles} {args.obj} P{int(args.P)} s{args.seed} '
          f'clamp{args.clamp} cg{k}] input full '
          f'clamped {ent["input_psnr_full_clamped"]:.2f} unclamped '
          f'{ent["input_psnr_full_unclamped"]:.2f} | inner (as fed) '
          f'{ent["input_psnr_inner"]:.2f}', flush=True)
    for R in RS:
        if f'R{R}' in ent:
            continue
        t0 = time.time()
        out, _ = chunked_foj.chunked_foj(inpc, mkopts(R), chunk_patches=13,
                                         halo=0, verbose=False,
                                         clip_blocks=False, clip_output=False)
        oi = out[inner]
        ent[f'R{R}'] = dict(psnr_inner=metrics.psnr(gti, oi),
                            psnr_inner_clipped=metrics.psnr(gti, np.clip(oi, 0, 1)),
                            s=time.time() - t0)
        e = ent[f'R{R}']
        print(f'  R={R}: inner PSNR {e["psnr_inner"]:.3f} '
              f'(clipped {e["psnr_inner_clipped"]:.3f}) ({e["s"]:.0f}s)', flush=True)
        json.dump(res, open(OUT, 'w'), indent=2)
    for tag in ('psnr_inner', 'psnr_inner_clipped'):
        ent[f'best_R_{tag}'] = max(RS, key=lambda r: ent[f'R{r}'][tag])
    print(f'  -> best R: unclipped {ent["best_R_psnr_inner"]}, '
          f'clipped {ent["best_R_psnr_inner_clipped"]}', flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)
print('DONE')
