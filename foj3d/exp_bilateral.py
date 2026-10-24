"""Junction-guided bilateral reconstruction: FoJ guide -> graph-Laplacian
regularizer inside cone-beam CT reconstruction.

  Guided recon: x* = argmin 1/2||Ax - p||^2 + lam/2 x^T L_g x
                solved by CG on (A^T A + lam L_g) x = A^T p, warm start CGLS-60.
  Post-hoc FBS: bilateral_solve(target=x0, c=1, guide=G).

Protocol: engine 256^3, det 256^2, 20 jittered views, P50 Poisson noise,
TIGRE CGLS-60 UNCLAMPED input (same as full_tigre_jitter_engine_P50_s0_cg60).

  tuning seed 1 -> bilateral_results_seed1_tuning.json
  test    seed 0 -> bilateral_results_seed0.json  (uses tuned settings)

Resumable: every finished entry is written to the JSON immediately; re-running
skips existing keys. Recon/projection/FoJ caches live on the VM as .npy.

usage:
  python exp_bilateral.py            # seed-1 tuning, then seed-0 test
  python exp_bilateral.py --seed 1   # one side only
  python exp_bilateral.py --smoke    # 64^3 sanity pass (smoke_results.json)
"""
import argparse
import inspect
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, '/content/3D-Field-of-Junctions')
import ct_pipeline
import field_of_junctions3d as foj_module
import chunked_foj
import metrics
import bilateral3d as B
from strict_protocol import mkopts, load_gt, get_angles, DS, DET

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=-1, help='0=test, 1=tuning, -1=both')
ap.add_argument('--smoke', action='store_true')
args = ap.parse_args()

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    foj_module.dev = torch.device('cuda')

ANG = get_angles('jitter')
ANG_F32 = ANG.astype(np.float32)
ANG360 = np.deg2rad(np.arange(360, dtype=np.float64)).astype(np.float32)
P_NOISE, CG_ITERS, CG_ITERS_TUNE = 50.0, 50, 50
SS_GRID = [4, 8]
SR_GRID = [0.05, 0.1]
LAM_FRAC_GUIDED = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]  # lam = frac*||Ax0-p||^2/x0'Lx0
LAM_FBS = [0.03, 0.1, 0.3, 1, 3, 10, 30]           # dimensionless grid weight
TV_GRID_ABS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


# ---------------- TIGRE wrappers (volume layout [D,H,W]; proj stored flipped) --

def _geo(nvox=DS, det=DET):
    return ct_pipeline.tigre_geometry(det, nvox)


def ax_np(v, geo, ang=ANG_F32):
    import tigre
    p = tigre.Ax(
        np.transpose(np.ascontiguousarray(v, np.float32), (2, 1, 0)), geo, ang)
    return np.ascontiguousarray(p[:, ::-1, :])


_ATB_KW = {}


def atb_np(p, geo, ang=ANG_F32):
    """Backprojection with the *matched* (adjoint) kernel — required so that
    A^T A is the exact transpose of A and CG sees a symmetric operator."""
    import tigre
    global _ATB_KW
    if not _ATB_KW:
        try:
            params = inspect.signature(tigre.Atb).parameters
        except (ValueError, TypeError):
            params = {}
        if 'backprojection_type' in params or not params:
            _ATB_KW = dict(backprojection_type='matched')
        else:
            _ATB_KW = {}          # kwarg name unknown -> logged deviation
        print(f'[atb] tigre.Atb signature={params}; kwargs={_ATB_KW}',
              flush=True)
    v = tigre.Atb(np.ascontiguousarray(p[:, ::-1, :], np.float32), geo, ang,
                  **_ATB_KW)
    return np.ascontiguousarray(np.transpose(v, (2, 1, 0)))


def dot_test(geo, ang, vol_shape):
    """<Ax,y> vs <x,A^T y> on random fields; returns relative discrepancy."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, vol_shape).astype(np.float32)
    A = ax_np(x, geo, ang)
    y = rng.uniform(0, 1, A.shape).astype(np.float32)
    lhs = float((A * y).sum())
    rhs = float((x * atb_np(y, geo, ang)).sum())
    rel = abs(lhs - rhs) / max(abs(lhs), 1e-12)
    print(f'[dot_test] <Ax,y>={lhs:.6f}  <x,Aty>={rhs:.6f}  rel={rel:.3e}',
          flush=True)
    return rel


def check_adjoint(geo, ang, vol_shape):
    rel = dot_test(geo, ang, vol_shape)
    if rel > 1e-2:
        raise RuntimeError(
            f'TIGRE Ax/Atb adjoint check failed (rel={rel:.3e}); CG would see '
            'a non-symmetric operator. Aborting.')
    return rel


def proj_cache(seed, vol, ang=ANG, det=DET):
    path = f'bil_proj_s{seed}.npy'
    if os.path.exists(path):
        return np.load(path)
    p = ct_pipeline.simulate(vol, ang, P_NOISE, seed, det=det, backend='tigre')
    np.save(path, p)
    return p


def input_cache(seed, vol_shape=DS):
    path = f'bil_input_s{seed}.npy'
    if os.path.exists(path):
        return np.load(path)
    from strict_protocol import recon_snapshots
    v = recon_snapshots([60], P_n=P_NOISE, seed=seed, obj='engine',
                        backend='tigre', angles='jitter')[60]
    np.save(path, v)
    return v


def foj_cache(seed, inp):
    path = f'bil_foj_r12_s{seed}.npy'
    if os.path.exists(path):
        return np.load(path)
    t0 = time.time()
    out, _ = chunked_foj.chunked_foj(
        inp, mkopts(12), chunk_patches=13, halo=0, verbose=True,
        clip_blocks=False, clip_output=False,
        checkpoint_path=f'ckpt_bil_foj_s{seed}.npz', ckpt_every=50)
    print(f'foj R12 seed{seed}: {time.time() - t0:.0f}s', flush=True)
    np.save(path, out.astype(np.float32))
    return out


# ---------------- TV baseline -------------------------------------------------

def tv_recon(proj, geo, ang, param, niter=100):
    """TIGRE TV baseline: fista with explicit TV weight `tvlambda`
    (hyper = Lipschitz step, left at default). Falls back to asd_pocs with
    its explicit regularization strength `alpha` if fista is missing.
    Returns (vol, desc)."""
    import tigre.algorithms as algs
    p = np.ascontiguousarray(proj[:, ::-1, :], np.float32)
    a = np.asarray(ang, np.float32)

    fn = getattr(algs, 'fista', None)
    if fn is not None:
        try:
            sig = inspect.signature(fn)
            print(f'[tv] fista signature: {sig}', flush=True)
        except (ValueError, TypeError):
            print('[tv] fista signature unavailable (Cython?)', flush=True)
        res = fn(p, geo, a, niter, tvlambda=param)
        vol = res[0] if isinstance(res, tuple) else res
        vol = np.ascontiguousarray(np.transpose(vol, (2, 1, 0))).astype(np.float32)
        return vol, f'fista(tvlambda={param})'

    fn = getattr(algs, 'asd_pocs', None)
    if fn is None:
        raise RuntimeError('no TIGRE TV algorithm usable (no fista/asd_pocs)')
    try:
        sig = inspect.signature(fn)
        print(f'[tv] asd_pocs signature: {sig}', flush=True)
    except (ValueError, TypeError):
        print('[tv] asd_pocs signature unavailable', flush=True)
    res = fn(p, geo, a, niter, alpha=param)
    vol = res[0] if isinstance(res, tuple) else res
    vol = np.ascontiguousarray(np.transpose(vol, (2, 1, 0))).astype(np.float32)
    return vol, f'asd_pocs(alpha={param})'


# ---------------- guided / fbs solvers ---------------------------------------

def make_lapply(grid):
    def lapply(v):
        t = torch.from_numpy(np.ascontiguousarray(v, np.float32)).to(DEV)
        return grid.apply_L(t).cpu().numpy()
    return lapply


def guided_recon(x0, proj, geo, lapply, lam, iters=CG_ITERS, ang=ANG_F32):
    """CG on (A^T A + lam L) x = A^T p, warm start x0."""
    b = atb_np(proj, geo, ang)

    def mop(v):
        return atb_np(ax_np(v, geo, ang), geo, ang) + lam * lapply(v)

    x = np.ascontiguousarray(x0, np.float32).copy()
    r = b - mop(x)
    p = r.copy()
    rs = float((r * r).sum())
    rs0 = rs
    nb = float(np.sqrt((b * b).sum()))
    best_x, best_rs = x.copy(), rs
    for i in range(iters):
        Ap = mop(p)
        den = float((p * Ap).sum())
        if not np.isfinite(den) or den <= 0:
            print(f'    guided it{i + 1}: non-PSD/NaN step, stop', flush=True)
            break
        alpha = rs / max(den, 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float((r * r).sum())
        if not np.isfinite(rs_new) or rs_new > 1e6 * rs0:
            print(f'    guided it{i + 1}: diverged, keep best', flush=True)
            x = best_x
            break
        if rs_new < best_rs:
            best_rs = rs_new
            best_x = x.copy()
        p = r + (rs_new / max(rs, 1e-30)) * p
        rs = rs_new
        if i % 10 == 9:
            print(f'    guided it{i + 1}: ||r||/||b||={np.sqrt(rs) / nb:.4f}',
                  flush=True)
    return x


def lam_guided_scale(x0, proj, geo, lapply, ang=ANG_F32):
    """lam such that lam*x0'Lx0 = frac * ||A x0 - p||^2 at the warm start."""
    lx = lapply(x0)
    l0 = float((x0 * lx).sum())
    r = ax_np(x0, geo, ang) - proj
    d0 = float((r * r).sum())
    return d0 / max(l0, 1e-12)


def fbs(target, guide, lam, ss, sr, iters=50):
    t = torch.from_numpy(np.ascontiguousarray(target, np.float32)).to(DEV)
    c = torch.ones_like(t)
    out = B.bilateral_solve(t, c, torch.from_numpy(
        np.ascontiguousarray(guide, np.float32)).to(DEV), lam, ss, sr,
        iters=iters, device=DEV)
    return out.cpu().numpy()


# ---------------- metrics -----------------------------------------------------

@torch.no_grad()
def project360(vol, geo_det=DET):
    import tigre
    p = tigre.Ax(np.transpose(np.ascontiguousarray(vol, np.float32), (2, 1, 0)),
                 _geo(vol.shape[0], geo_det), ANG360)
    return torch.from_numpy(np.ascontiguousarray(p[:, ::-1, :])).to(DEV).float()


def make_scorer(gt):
    from pytorch_msssim import ms_ssim
    P_gt = project360(gt)
    gmax = P_gt.max()

    @torch.no_grad()
    def score(vol, full=True):
        vc = np.clip(vol, 0, 1)
        d = dict(psnr3d=fin(metrics.psnr(gt, vc)),
                 psnr3d_unclipped=fin(metrics.psnr(gt, vol)))
        if full and np.isfinite(vc).all():
            P = project360(vc)
            a = P_gt / P_gt.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            b = P / P.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            d['msssim_perview'] = ms_ssim(b[:, None], a[:, None],
                                        data_range=1.0).item()
            d['msssim_shared'] = ms_ssim((P / gmax).clamp(0, None)[:, None],
                                         (P_gt / gmax)[:, None],
                                         data_range=1.0).item()
        return d
    return score


def save(res, path):
    json.dump(res, open(path, 'w'), indent=2)


def fin(x):
    """float or None (JSON-safe) for non-finite values."""
    return float(x) if np.isfinite(x) else None


def psnr_entry(gt, out):
    return dict(psnr3d=fin(metrics.psnr(gt, np.clip(out, 0, 1))),
                psnr3d_unclipped=fin(metrics.psnr(gt, out)))


def fmt(e):
    p = e.get('psnr3d')
    pu = e.get('psnr3d_unclipped')
    s = f'PSNR {p:.2f}/{pu:.2f}' if p is not None and pu is not None else 'PSNR NaN'
    if e.get('msssim_perview') is not None:
        s += f' MSSSIM {e["msssim_perview"]:.4f}/{e["msssim_shared"]:.4f}'
    return s + f' ({e.get("s", 0):.0f}s)'


# ---------------- tuning (seed 1) ----------------------------------------------

def run_tuning(gt):
    OUT = 'bilateral_results_seed1_tuning.json'
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    geo = _geo()
    if 'dot_test' not in res:
        res['dot_test'] = check_adjoint(geo, ANG, (DS, DS, DS))
        save(res, OUT)
    proj = proj_cache(1, gt)
    x0 = input_cache(1)
    foj1 = foj_cache(1, x0)
    print('[tune] input PSNR %.2f, foj guide ready' % metrics.psnr(gt, np.clip(x0, 0, 1)), flush=True)

    # --- post-hoc FBS tuning on FoJ guide (lam is dimensionless) ---
    for ss in SS_GRID:
        for sr in SR_GRID:
            grid = B.BilateralGrid3D(foj1, ss, sr, device=DEV)
            print(f'[tune-fbs] ss={ss} sr={sr}: K={grid.K} dims={grid.dims}',
                  flush=True)
            del grid
            for lam in LAM_FBS:
                key = f'fbs_ss{ss}_sr{sr}_lf{lam}'
                if key in res:
                    continue
                t0 = time.time()
                out = fbs(x0, foj1, lam, ss, sr)
                res[key] = dict(s=time.time() - t0, lam=lam,
                                **psnr_entry(gt, out))
                print(f'  {key}: {fmt(res[key])}', flush=True)
                save(res, OUT)
    # --- guided-recon tuning on FoJ guide ---
    for ss in SS_GRID:
        for sr in SR_GRID:
            grid = B.BilateralGrid3D(foj1, ss, sr, device=DEV)
            lap = make_lapply(grid)
            scale = lam_guided_scale(x0, proj, geo, lap)
            for lf in LAM_FRAC_GUIDED:
                key = f'guided_ss{ss}_sr{sr}_lf{lf}'
                if key in res:
                    continue
                t0 = time.time()
                out = guided_recon(x0, proj, geo, lap, lam=lf * scale,
                                   iters=CG_ITERS_TUNE)
                res[key] = dict(s=time.time() - t0, lam=lf * scale,
                                **psnr_entry(gt, out))
                print(f'  {key}: {fmt(res[key])}', flush=True)
                save(res, OUT)
            del grid
    # --- TV baseline tuning ---
    for tv in TV_GRID_ABS:
        key = f'tv_{tv}'
        if key in res:
            continue
        try:
            t0 = time.time()
            vol, desc = tv_recon(proj, geo, ANG, tv)
            res[key] = dict(s=time.time() - t0, desc=desc,
                            **psnr_entry(gt, vol))
        except Exception as e:
            res[key] = dict(error=str(e))
        print(f'  {key}: {res[key]}', flush=True)
        save(res, OUT)
    # --- pick best (selection metric: clipped psnr3d, matching the 18.86
    # FoJ reference) + loud warning if a lam optimum sits at a grid end ---
    def best(prefix, grid_vals):
        cands = [(k, v) for k, v in res.items()
                 if k.startswith(prefix) and v.get('psnr3d') is not None]
        bk, bv = max(cands, key=lambda kv: kv[1]['psnr3d'])
        lf = float([p[2:] for p in bk.split('_') if p.startswith('lf')][0])
        at_edge = lf == min(grid_vals) or lf == max(grid_vals)
        if at_edge:
            print(f'WARNING best at grid boundary: {bk} (lf={lf})', flush=True)
        return bk, at_edge
    def parse(k):
        parts = k.split('_')
        ss = int([p[2:] for p in parts if p.startswith('ss')][0])
        sr = float([p[2:] for p in parts if p.startswith('sr')][0])
        lf = float([p[2:] for p in parts if p.startswith('lf')][0])
        return ss, sr, lf
    bf, bf_edge = best('fbs_', LAM_FBS)
    bg, bg_edge = best('guided_', LAM_FRAC_GUIDED)
    tvs = [(k, v) for k, v in res.items()
           if k.startswith('tv_') and v.get('psnr3d') is not None]
    bt = max(tvs, key=lambda kv: kv[1]['psnr3d'])[0] if tvs else None
    res['best'] = dict(fbs=parse(bf), guided=parse(bg),
                       tv=float(bt[3:]) if bt else None,
                       fbs_key=bf, guided_key=bg, tv_key=bt,
                       fbs_at_boundary=bf_edge, guided_at_boundary=bg_edge)
    print('[tune] best:', res['best'], flush=True)
    save(res, OUT)
    return res['best']


# ---------------- test (seed 0) -------------------------------------------------

_VOLS = {}


def run_test(gt, best):
    OUT = 'bilateral_results_seed0.json'
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    geo = _geo()
    if 'dot_test' not in res:
        res['dot_test'] = check_adjoint(geo, ANG, (DS, DS, DS))
        save(res, OUT)
    proj = proj_cache(0, gt)
    x0 = input_cache(0)
    score = make_scorer(gt)

    def entry(key, fn):
        if key in res:
            return np.asarray(_VOLS.get(key)) if key in _VOLS else None
        t0 = time.time()
        vol = fn()
        res[key] = dict(score(vol), s=time.time() - t0)
        _VOLS[key] = vol
        np.save(f'bil_out_{key}.npy', vol.astype(np.float16))
        print(f'  {key}: {fmt(res[key])}', flush=True)
        save(res, OUT)
        return vol

    if 'input' not in res:
        res['input'] = score(x0)
        save(res, OUT)
    print('[test] input: ' + fmt(res['input']), flush=True)

    foj0 = foj_cache(0, x0)
    if 'foj_r12' not in res:
        res['foj_r12'] = dict(score(foj0), s=0.0)
        print('  foj_r12: ' + fmt(res['foj_r12']), flush=True)
        save(res, OUT)
    g_gauss = gaussian_filter(x0, 1.5).astype(np.float32)
    g_foj = np.ascontiguousarray(foj0, np.float32)
    g_gt = gt.astype(np.float32)
    np.save('bil_vol_input.npy', x0.astype(np.float16))
    np.save('bil_vol_guide_foj.npy', g_foj.astype(np.float16))

    if best.get('tv') is not None:
        entry('tv', lambda: tv_recon(proj, geo, ANG, best['tv'])[0])

    ss_f, sr_f, lf_f = best['fbs']
    ss_g, sr_g, lf_g = best['guided']
    for gname, guide in (('gauss', g_gauss), ('foj', g_foj), ('gt', g_gt)):
        entry(f'fbs_{gname}', lambda g=guide: fbs(x0, g, lf_f, ss_f, sr_f))

    grids = {}
    for gname, guide in (('gauss', g_gauss), ('foj', g_foj), ('gt', g_gt)):
        key = f'guided_{gname}'
        grids[gname] = B.BilateralGrid3D(guide, ss_g, sr_g, device=DEV)
        if key not in res:
            lap = make_lapply(grids[gname])
            scale = lam_guided_scale(x0, proj, geo, lap)
            entry(key, lambda lap=lap, sc=scale:
                  guided_recon(x0, proj, geo, lap, lam=lf_g * sc))

    if 'alternation' not in res:
        gkeys = [k for k in res if k.startswith('guided_')
                 and res[k].get('psnr3d') is not None]
        bk = max(gkeys, key=lambda k: res[k]['psnr3d'])
        print(f'[test] alternation from {bk}', flush=True)
        g2_path = 'bil_foj_r12_alt_s0.npy'
        if os.path.exists(g2_path):
            g2 = np.load(g2_path)
        else:
            prev_path = f'bil_out_{bk}.npy'
            prev = np.load(prev_path) if os.path.exists(prev_path) else _VOLS[bk]
            g2, _ = chunked_foj.chunked_foj(
                prev.astype(np.float32),      # unclamped protocol
                mkopts(12), chunk_patches=13, halo=0, verbose=True,
                clip_blocks=False, clip_output=False,
                checkpoint_path='ckpt_bil_foj_alt.npz', ckpt_every=50)
            np.save(g2_path, g2.astype(np.float32))
        grid2 = B.BilateralGrid3D(g2, ss_g, sr_g, device=DEV)
        lap2 = make_lapply(grid2)
        scale = lam_guided_scale(x0, proj, geo, lap2)
        entry('alternation',
              lambda: guided_recon(x0, proj, geo, lap2, lam=lf_g * scale))
    print('DONE', flush=True)
    return res


# ---------------- smoke ---------------------------------------------------------

def smoke():
    print('[smoke] 64^3 sanity run', flush=True)
    gt = load_gt('engine')
    s = gt.shape[0] // 2
    vol = gt[s - 32:s + 32, s - 32:s + 32, s - 32:s + 32]
    geo = _geo(vol.shape[0], 64)
    ang = np.linspace(0, 2 * np.pi, 7)[:-1].astype(np.float32)
    check_adjoint(geo, ang, vol.shape)
    proj = ct_pipeline.add_noise(ax_np(vol, geo, ang), 50, 0)
    import tigre.algorithms as algs
    t0 = time.time()
    v = algs.cgls(np.ascontiguousarray(proj[:, ::-1, :], np.float32), geo, ang, 10)
    v = v[0] if isinstance(v, tuple) else v
    x0 = np.ascontiguousarray(np.transpose(v, (2, 1, 0))).astype(np.float32)
    print(f'[smoke] cgls10 done ({time.time() - t0:.0f}s), '
          f'psnr {metrics.psnr(vol, np.clip(x0, 0, 1)):.2f}', flush=True)
    grid = B.BilateralGrid3D(vol, 4, 0.05, device=DEV)
    out = fbs(x0, vol, lam=1.0, ss=4, sr=0.05, iters=20)
    print(f'[smoke] fbs psnr {metrics.psnr(vol, np.clip(out, 0, 1)):.2f}', flush=True)
    lap = make_lapply(grid)
    scale = lam_guided_scale(x0, proj, geo, lap, ang=ang)
    gout = guided_recon(x0, proj, geo, lap, lam=1.0 * scale, iters=15, ang=ang)
    print(f'[smoke] guided psnr {metrics.psnr(vol, np.clip(gout, 0, 1)):.2f}',
          flush=True)
    try:
        tv, desc = tv_recon(proj, geo, ang, 0.1, niter=5)
        print(f'[smoke] tv via {desc}: psnr {metrics.psnr(vol, np.clip(tv, 0, 1)):.2f}',
              flush=True)
    except Exception as e:
        print(f'[smoke] tv failed: {e}', flush=True)
    print('SMOKE DONE', flush=True)


if __name__ == '__main__':
    if args.smoke:
        smoke()
        sys.exit(0)
    gt = load_gt('engine')
    if args.seed in (1, -1):
        best = run_tuning(gt)
    else:
        t = json.load(open('bilateral_results_seed1_tuning.json'))
        best = t['best']
    if args.seed in (0, -1):
        run_test(gt, best)
