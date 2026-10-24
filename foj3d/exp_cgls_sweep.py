"""Experiment B — does the optimal patch size move with effective noise level?

The paper does not state its CGLS iteration count. Later CGLS iterates are
less implicitly regularized (noisier), so the iteration count sets the
effective input noise. Sweep k in ITERS; for each, FoJ R in {4..12}.

FoJ is evaluated on the central 128^3 crop [64:192] with exact (halo-free,
unclipped) chunking. Because patches are fit independently at init=1/refine=1,
every voxel at distance >= R-1 from the crop border receives exactly the same
patch set as in a full-volume run, so PSNR is reported on the common interior
[12:116]^3 (margin 12 >= max R - 1). Full-volume input PSNR is logged so the
regime can be matched to paper Table 3 (their R4 output = 11.33 dB).
"""
import json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, '/content/3D-Field-of-Junctions')
import field_of_junctions3d as foj_module
foj_module.dev = torch.device('cuda')
import chunked_foj, metrics
from strict_protocol import mkopts, load_gt, recon_snapshots

ITERS = [5, 10, 15, 25, 40, 60, 100]
RS = (4, 6, 8, 10, 12)
OUT = 'exp_cgls_sweep.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

gt = load_gt()
snaps = recon_snapshots(ITERS)
crop = (slice(64, 192),) * 3
inner = (slice(12, 116),) * 3
gtc = gt[crop]

for k in ITERS:
    kk = f'cg{k}'
    ent = res.setdefault(kk, {})
    raw = snaps[k]
    rec = np.clip(raw, 0, 1)
    ent['input_psnr_full_clamped'] = metrics.psnr(gt, rec)
    ent['input_psnr_full_unclamped'] = metrics.psnr(gt, raw)
    recc = rec[crop]
    ent['input_psnr_inner'] = metrics.psnr(gtc[inner], recc[inner])
    print(f'[{kk}] input full {ent["input_psnr_full_clamped"]:.2f} '
          f'(unclamped {ent["input_psnr_full_unclamped"]:.2f}) '
          f'inner {ent["input_psnr_inner"]:.2f}', flush=True)
    for R in RS:
        if f'R{R}' in ent:
            continue
        t0 = time.time()
        out, _ = chunked_foj.chunked_foj(recc, mkopts(R), chunk_patches=13,
                                         halo=0, verbose=False, clip_blocks=False)
        ent[f'R{R}'] = dict(psnr_inner=metrics.psnr(gtc[inner], out[inner]),
                            s=time.time() - t0)
        print(f'  R={R}: inner PSNR {ent[f"R{R}"]["psnr_inner"]:.3f} '
              f'({ent[f"R{R}"]["s"]:.0f}s)', flush=True)
        json.dump(res, open(OUT, 'w'), indent=2)
    best = max(RS, key=lambda r: ent[f'R{r}']['psnr_inner'])
    ent['best_R'] = best
    print(f'  -> best R at cg{k}: {best}', flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)
print('DONE')
