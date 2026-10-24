"""Experiment A — is chunked assembly equivalent to the upstream global FoJ?

Claim under test: at init=1/refine=1 (lambda ramp factor = 0) every patch is
fit independently and the output is the uniform overlap average (paper Eq. 4),
so a halo-free, unclipped chunk partition must reproduce the global model up
to float error (+ a possible Adam-eps effect from loss.mean() over N patches).

Input: central 80^3 crop of the strict 256^3 P50 CGLS-15 recon (clamped [0,1],
identical to exp_strict256). Variants per R:
  global      upstream FieldOfJunctions3D on the whole crop
  exact       chunked, halo=0, no per-block clip
  halo        chunked, halo=R, per-block clip (the driver used for 17.82/17.46)
  global_init / exact_init   same, refine=0 (isolates the Adam step)
"""
import json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, '/content/3D-Field-of-Junctions')
import field_of_junctions3d as foj_module
foj_module.dev = torch.device('cuda')
import chunked_foj, metrics
from strict_protocol import mkopts, load_gt, recon_snapshots

OUT = 'exp_assembly.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

gt = load_gt()
rec = np.clip(recon_snapshots([15])[15], 0, 1)
sl = (slice(88, 168),) * 3
gtc, recc = gt[sl], rec[sl]
res['input_psnr_crop'] = metrics.psnr(gtc, recc)
print(f'crop input PSNR {res["input_psnr_crop"]:.3f}', flush=True)


def run_global(vol, opts):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m = foj_module.FieldOfJunctions3D(vol[..., None], opts)
    for it in range(m.num_iters):
        m.step(it)
    out = np.clip(m.global_volume[0, 0].detach().cpu().numpy(), 0, 1)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del m
    torch.cuda.empty_cache()
    return out, time.time() - t0, peak


def run_chunked(vol, opts, halo, clip_blocks):
    t0 = time.time()
    out, _ = chunked_foj.chunked_foj(vol, opts, chunk_patches=13, halo=halo,
                                     verbose=False, clip_blocks=clip_blocks)
    return out, time.time() - t0


def cmp(a, b):
    d = np.abs(a - b)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                frac_gt_1e3=float((d > 1e-3).mean()))


for R in (4, 6, 8, 10, 12):
    key = f'R{R}'
    if key in res:
        continue
    o = mkopts(R)
    g, tg, peak = run_global(recc, o)
    e, te = run_chunked(recc, o, halo=0, clip_blocks=False)
    h, th = run_chunked(recc, o, halo=R, clip_blocks=True)
    oi = mkopts(R, num_refinement_iters=0)
    gi, _, _ = run_global(recc, oi)
    ei, _ = run_chunked(recc, oi, halo=0, clip_blocks=False)
    res[key] = dict(
        psnr=dict(global_=metrics.psnr(gtc, g), exact=metrics.psnr(gtc, e),
                  halo=metrics.psnr(gtc, h), global_init=metrics.psnr(gtc, gi),
                  exact_init=metrics.psnr(gtc, ei)),
        diff=dict(exact_vs_global=cmp(e, g), halo_vs_global=cmp(h, g),
                  exact_init_vs_global_init=cmp(ei, gi),
                  global_vs_global_init=cmp(g, gi)),
        time_s=dict(global_=tg, exact=te, halo=th), global_peak_gib=peak)
    p = res[key]['psnr']
    print(f'R={R}: global {p["global_"]:.3f} exact {p["exact"]:.3f} '
          f'halo {p["halo"]:.3f} | init-only global {p["global_init"]:.3f} '
          f'exact {p["exact_init"]:.3f} | max|exact-global| '
          f'{res[key]["diff"]["exact_vs_global"]["max_abs"]:.2e} '
          f'max|halo-global| {res[key]["diff"]["halo_vs_global"]["max_abs"]:.2e} '
          f'peak {peak:.1f}GiB', flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)
print('DONE')
