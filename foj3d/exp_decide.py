"""Deciding-question experiment (seed 1 only, exploratory):

  Does guide-constrained reconstruction beat FoJ post-hoc (18.85 dB) even
  with a PERFECT (ground-truth) edge guide?

    - guidedGT from CGLS start and warm-started from the FoJ output
    - guidedFoJ warm-started from FoJ (can the data term polish FoJ at all?)
    - guidedFoJ from CGLS at stronger lam (past the lf=10 grid edge)
    - fbsGT reference (post-hoc bilateral solve with GT guide; cheap)

If the GT guide cannot beat FoJ alone, the formulation is dead regardless of
guide quality. Writes decide_results.json after EVERY entry so partial
results survive a session loss.
"""
import json
import os
import sys
import time

sys.argv = ['exp_decide.py']          # neutralize exp_bilateral's argparse
sys.path.insert(0, '/content/3D-Field-of-Junctions')

import numpy as np

import exp_bilateral as E
import bilateral3d as B
import metrics
from strict_protocol import load_gt, DS

OUT = 'decide_results.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(res, open(OUT, 'w'), indent=2)


def run(key, fn):
    if key in res:
        print(f'  {key}: (cached) {res[key]}', flush=True)
        return
    t0 = time.time()
    out = fn()
    res[key] = dict(s=time.time() - t0, **E.psnr_entry(gt, out))
    print(f'  {key}: {E.fmt(res[key])}', flush=True)
    save()


if __name__ == '__main__':
    gt = load_gt('engine')
    geo = E._geo()
    if 'dot_test' not in res:
        res['dot_test'] = E.check_adjoint(geo, E.ANG, (DS, DS, DS))
        save()
    proj = E.proj_cache(1, gt)
    x0 = E.input_cache(1)
    foj = E.foj_cache(1, x0)
    res['input'] = E.psnr_entry(gt, x0)
    res['foj_r12'] = E.psnr_entry(gt, foj)
    save()
    print(f"[decide] input {res['input']['psnr3d']:.2f}  "
          f"foj {res['foj_r12']['psnr3d']:.2f}", flush=True)

    SS, SR, IT = 8, 0.05, 50
    g_gt = gt.astype(np.float32)
    g_foj = np.ascontiguousarray(foj, np.float32)

    # cheap post-hoc reference: FBS on x0 with the perfect guide
    for lf in (10, 30, 100):
        run(f'fbsGT_lf{lf}', lambda lf=lf: E.fbs(x0, g_gt, lf, SS, SR))

    for gname, guide in (('gt', g_gt), ('foj', g_foj)):
        grid = B.BilateralGrid3D(guide, SS, SR, device=E.DEV)
        lap = E.make_lapply(grid)
        for iname, xi in (('cgls', x0), ('foj', foj)):
            scale = E.lam_guided_scale(xi, proj, geo, lap)
            lfs = ([3, 10, 30, 100, 300] if (gname, iname) == ('gt', 'cgls')
                   else [1, 3, 10, 30, 100] if (gname, iname) == ('gt', 'foj')
                   else [3, 10, 30, 100] if (gname, iname) == ('foj', 'foj')
                   else [30, 100])   # foj guide from x0: extend past boundary
            for lf in lfs:
                key = f'guided_{gname}_{iname}_lf{lf}'
                run(key, lambda xi=xi, sc=scale, lf=lf:
                    E.guided_recon(xi, proj, geo, lap, lam=lf * sc, iters=IT))
        del grid
    print('DECIDE DONE', flush=True)
