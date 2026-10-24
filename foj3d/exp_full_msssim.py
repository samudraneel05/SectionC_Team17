"""Full-volume 256^3 R sweep with the paper's metrics.

  3D PSNR  : full-volume MSE, peak 1 (= R2-Gaussian metric_vol), on the FoJ
             output clipped to [0,1] (also logged unclipped).
  MS-SSIM  : on 360 projections, one per degree, det 256^2, same cone geometry
             (paper Sec. 5.1). pytorch_msssim.ms_ssim, win 11, 5 scales.
             Two normalizations logged (paper does not specify):
               perview : each projection / its own max (R2-Gaussian metric_proj)
               shared  : both / max over all GT projections
Exact chunking (halo=0, no per-block clip), checkpointed per R.

usage: python exp_full_msssim.py --obj engine --seed 0 --cg 15 --clamp 0
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from pytorch_msssim import ms_ssim

sys.path.insert(0, '/content/3D-Field-of-Junctions')
import field_of_junctions3d as foj_module
foj_module.dev = torch.device('cuda')
import chunked_foj, metrics
import projector as pj
from strict_protocol import mkopts, load_gt, recon_snapshots, DET

ap = argparse.ArgumentParser()
ap.add_argument('--obj', default='engine')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--cg', type=int, default=15)
ap.add_argument('--clamp', type=int, default=1)
ap.add_argument('--rs', default='12,6,10,8,4')
ap.add_argument('--P', type=float, default=50.0)
ap.add_argument('--backend', default='tigre')
ap.add_argument('--angles', default='jitter')
args = ap.parse_args()
RS = [int(x) for x in args.rs.split(',')]

tag = (f'{args.backend}_{args.angles}_{args.obj}_P{int(args.P)}_s{args.seed}'
       f'_cg{args.cg}_clamp{args.clamp}')
OUT = f'full_{tag}.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

gt = load_gt(args.obj)
raw = recon_snapshots([args.cg], P_n=args.P, seed=args.seed, obj=args.obj,
                      backend=args.backend, angles=args.angles)[args.cg]
inp = np.clip(raw, 0, 1) if args.clamp else raw

ANG360 = np.deg2rad(np.arange(360, dtype=np.float64))


@torch.no_grad()
def project360(vol):
    if args.backend == 'tigre':
        import tigre
        from ct_pipeline import tigre_geometry
        p = tigre.Ax(np.transpose(np.asarray(vol, np.float32), (2, 1, 0)).copy(),
                     tigre_geometry(DET, vol.shape[0]), ANG360.astype(np.float32))
        return torch.from_numpy(np.ascontiguousarray(p[:, ::-1, :])).cuda().float()
    v = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[None, None].cuda()
    return pj.forward_project(v, ANG360, det_hw=DET).float()


P_gt = project360(gt)
gmax = P_gt.max()


@torch.no_grad()
def proj_msssim(vol):
    P = project360(vol)
    a = P_gt / P_gt.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    b = P / P.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    per = ms_ssim(b[:, None], a[:, None], data_range=1.0, size_average=True).item()
    sh = ms_ssim((P / gmax).clamp(0, None)[:, None], (P_gt / gmax)[:, None],
                 data_range=1.0, size_average=True).item()
    return per, sh


def score(vol):
    vc = np.clip(vol, 0, 1)
    per, sh = proj_msssim(vc)
    return dict(psnr3d=metrics.psnr(gt, vc), psnr3d_unclipped=metrics.psnr(gt, vol),
                msssim_perview=per, msssim_shared=sh)


if 'input' not in res:
    res['input'] = score(inp)
    json.dump(res, open(OUT, 'w'), indent=2)
i = res['input']
print(f'[{tag}] input: PSNR {i["psnr3d"]:.2f} (unclipped {i["psnr3d_unclipped"]:.2f}) '
      f'MS-SSIM perview {i["msssim_perview"]:.4f} shared {i["msssim_shared"]:.4f}',
      flush=True)

for R in RS:
    if f'R{R}' in res:
        continue
    t0 = time.time()
    ck = f'ckpt_{tag}_R{R}.npz'
    out, _ = chunked_foj.chunked_foj(inp, mkopts(R), chunk_patches=13, halo=0,
                                     verbose=False, clip_blocks=False,
                                     clip_output=False, checkpoint_path=ck,
                                     ckpt_every=50)
    s = time.time() - t0
    res[f'R{R}'] = dict(score(out), s=s)
    if os.path.exists(ck):
        os.remove(ck)
    e = res[f'R{R}']
    print(f'  R={R}: PSNR {e["psnr3d"]:.3f} (unclipped {e["psnr3d_unclipped"]:.3f}) '
          f'MS-SSIM perview {e["msssim_perview"]:.4f} shared {e["msssim_shared"]:.4f} '
          f'({s:.0f}s)', flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)
print('DONE')
