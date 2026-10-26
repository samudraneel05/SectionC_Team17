"""3D FoJ on MRI volumes — two experiments, incremental JSON output.

  E1  Rician denoising, BrainWeb T1 (181x217x181, scaled to [0,1]).
      sigma in {3,5,7,9,13}% of max intensity (standard MRI protocol).
      Methods: noisy, gaussian, median, TV, BM4D, FoJ R in {6,8,12}.

  E2  Accelerated-MRI reconstruction (plug-and-play, single-coil).
      Fully sample kx (dim 0), variable-density random mask on (ky,kz)
      with a calibration disc, acceleration targets {4x, 8x}.
      Recon = alternating [denoiser] / [hard k-space data consistency],
      T=4 rounds; denoisers: none (zero-fill), TV, BM4D, FoJ R=8.

Writes mri_results.json after EVERY entry (survives kernel preemption).
Run on Kaggle: sys.path gets the code via the 'foj3d-code' dataset.
Phantom expected at */t1_icbm_normal_1mm_pn0_rf0.rawb under /kaggle/input
(or ./brainweb_phantom/ locally).
"""
import glob
import json
import os
import sys
import time

import numpy as np

for cand in (glob.glob('/kaggle/input/**/chunked_foj.py', recursive=True) +
             glob.glob('/content/3D-Field-of-Junctions/chunked_foj.py') +
             [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chunked_foj.py')]):
    d = os.path.dirname(cand)
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)
        break

print('kaggle/input:', os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'n/a', flush=True)
import torch  # noqa: E402
import chunked_foj  # noqa: E402
import metrics  # noqa: E402
from strict_protocol import mkopts  # noqa: E402

OUT = 'mri_results.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
SMOKE = '--smoke' in sys.argv


def save():
    json.dump(res, open(OUT, 'w'), indent=2)


def fin(x):
    return float(x) if np.isfinite(x) else None


def entry(gt, vol):
    v = np.clip(vol, 0, 1)
    return dict(psnr3d=fin(metrics.psnr(gt, v)),
                psnr3d_unclipped=fin(metrics.psnr(gt, vol)),
                ssim3d=fin(metrics.ssim3d(gt, v)))


def run(key, fn, save_vol=None):
    if key in res:
        print(f'  {key}: cached {res[key]}', flush=True)
        return
    t0 = time.time()
    out = fn()
    res[key] = dict(s=time.time() - t0, **entry(gt, out))
    print(f'  {key}: {res[key]}', flush=True)
    save()
    if save_vol:
        np.save(save_vol, out.astype(np.float16))


def load_phantom():
    cands = (glob.glob('/kaggle/input/**/t1_icbm*.rawb', recursive=True) +
             glob.glob('**/t1_icbm*.rawb', recursive=True))
    if not cands:
        raise FileNotFoundError('BrainWeb rawb not found under /kaggle/input or cwd')
    v = np.fromfile(cands[0], dtype=np.uint8).reshape(181, 217, 181, order='F')
    return (v.astype(np.float32) / 255.0)


def rician(x, sigma, rng):
    """Magnitude-Rician corruption: |x + sigma*(n1 + i n2)|."""
    n1 = rng.normal(0, sigma, x.shape)
    n2 = rng.normal(0, sigma, x.shape)
    return np.sqrt((x + n1) ** 2 + n2 ** 2).astype(np.float32)


# ---------------- denoisers -------------------------------------------------

def den_gauss(y):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(y, 1.0).astype(np.float32)


def den_median(y):
    from scipy.ndimage import median_filter
    return median_filter(y, size=3).astype(np.float32)


try:
    from skimage.restoration import denoise_tv_chambolle as _tv_chambolle
except Exception:
    _tv_chambolle = None
try:
    import bm4d as _bm4d
except Exception:
    _bm4d = None


def den_tv(y, weight):
    return _tv_chambolle(y.astype(np.float64), weight=weight).astype(np.float32)


def den_bm4d(y, sigma):
    return np.asarray(_bm4d.bm4d(y, sigma), np.float32)


def den_foj(y, R, tag):
    out, _ = chunked_foj.chunked_foj(
        y.astype(np.float32), mkopts(R), chunk_patches=13, halo=0,
        verbose=True, clip_blocks=False, clip_output=False,
        checkpoint_path=f'ckpt_mri_{tag}.npz', ckpt_every=50)
    return out


# ---------------- E1: Rician denoising --------------------------------------

NOISES = [0.03, 0.05, 0.07, 0.09, 0.13]
RS = [6, 8, 12]
TV_W = [0.05, 0.1, 0.2]


def e1(gt):
    rng = np.random.default_rng(0)
    for s in NOISES:
        if SMOKE and s != 0.05:
            continue
        tag = f's{int(s * 100)}'
        y = rician(gt, s, rng)
        np.save(f'mri_noisy_{tag}.npy', y.astype(np.float16))
        run(f'e1_noisy_{tag}', lambda y=y: y)
        run(f'e1_gauss_{tag}', lambda y=y: den_gauss(y))
        run(f'e1_median_{tag}', lambda y=y: den_median(y))
        if _tv_chambolle is not None:
            for w in TV_W:
                run(f'e1_tv_w{w}_{tag}', lambda y=y, w=w: den_tv(y, w))
        else:
            print('  skimage unavailable: skipping TV', flush=True)
        if _bm4d is not None:
            try:
                run(f'e1_bm4d_{tag}', lambda y=y, s=s: den_bm4d(y, s))
            except Exception as e:
                res[f'e1_bm4d_{tag}'] = dict(error=str(e))
                save()
        else:
            res[f'e1_bm4d_{tag}'] = dict(error='bm4d not installed')
            save()
        for R in ([6] if SMOKE else RS):
            run(f'e1_foj_R{R}_{tag}', lambda y=y, R=R, tag=tag:
                den_foj(y, R, f'e1_{tag}_R{R}'),
                save_vol=f'mri_foj_R{R}_{tag}.npy')


# ---------------- E2: accelerated-MRI plug-and-play -------------------------

def vd_mask(H, W, accel, acs_radius, rng):
    """Variable-density mask on (ky,kz): calibration disc + prob(r)~linear decay,
    outer probability scaled by bisection to hit `accel` in expectation."""
    cy, cz = np.fft.fftshift(np.arange(H)) - H // 2, np.fft.fftshift(np.arange(W)) - W // 2
    r = np.sqrt(np.add.outer(cy ** 2, cz ** 2))
    r_n = r / r.max()
    base = np.clip(1.0 - r_n, 0.0, 1.0) ** 1.5     # denser toward centre
    acs = r <= acs_radius                          # calibration disc, pixels
    want = (H * W) / accel
    lo, hi = 0.0, 1.0 / max(base[~acs].mean(), 1e-9)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        p = np.clip(base * mid, 0, 1)
        exp_total = acs.sum() + p[~acs].sum()
        if exp_total < want:
            lo = mid
        else:
            hi = mid
    p = np.clip(base * hi, 0, 1)
    mask = acs | (rng.random((H, W)) < p)
    return mask, (H * W) / mask.sum()


def fft3(x):
    return np.fft.fftn(x.astype(np.float64), norm='ortho')


def ifft3(k):
    return np.abs(np.fft.ifftn(k, norm='ortho')).astype(np.float32)


def pnp(K, mask, denoise, T):
    """Alternating [denoise]/[hard data-consistency] POCS-style loop.
    denoise(x, t) is called with the round index (checkpoint tags differ)."""
    x = ifft3(K * mask)
    for t in range(T):
        x = np.clip(denoise(x, t).astype(np.float32), 0, None)
        k = fft3(x)
        k[mask] = K[mask]
        x = ifft3(k)
        print(f'    pnp round {t + 1}/{T} done', flush=True)
    return x


def e2(gt):
    if SMOKE:
        accels, T = [(4.0, 'a4')], 1
    else:
        accels, T = [(4.0, 'a4'), (8.0, 'a8')], 4
    K = fft3(gt)
    D, H, W = gt.shape
    rng = np.random.default_rng(1)
    for target, tag in accels:
        mkey = f'mask_{tag}'
        if mkey in res:
            mask = np.load(f'mri_mask_{tag}.npy')
        else:
            mask, actual = vd_mask(H, W, target, 8, rng)
            np.save(f'mri_mask_{tag}.npy', mask)
            res[mkey] = dict(target=target, actual=round(float(actual), 3))
            save()
            print(f'[e2] mask {tag}: target {target}x, actual {actual:.2f}x', flush=True)
        m3 = np.broadcast_to(mask[None, :, :], (D, H, W))

        run(f'e2_zerofill_{tag}', lambda m3=m3: ifft3(K * m3))

        if _tv_chambolle is not None:
            run(f'e2_tv_{tag}', lambda m3=m3:
                pnp(K, m3, lambda x, t: den_tv(x, 0.1), T))
        if _bm4d is not None:
            try:
                run(f'e2_bm4d_{tag}', lambda m3=m3:
                    pnp(K, m3, lambda x, t: den_bm4d(x, 0.02), T))
            except Exception as e:
                res[f'e2_bm4d_{tag}'] = dict(error=str(e))
                save()
        R = 6 if SMOKE else 8
        run(f'e2_foj_{tag}', lambda m3=m3, tag=tag:
            pnp(K, m3, lambda x, t: den_foj(x, R, f'e2_{tag}_it{t}'), T),
            save_vol=f'mri_pnp_foj_{tag}.npy')


if __name__ == '__main__':
    print('device:', 'cuda' if torch.cuda.is_available() else 'cpu', flush=True)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0), flush=True)
    gt = load_phantom()
    if SMOKE:
        gt = gt[60:124, 80:144, 60:124]          # 64^3 smoke volume
        print('[smoke] 64^3 volume', flush=True)
    np.save('mri_gt.npy', gt.astype(np.float16))
    res['gt_shape'] = list(gt.shape)
    save()
    e1(gt)
    e2(gt)
    print('MRI DONE', flush=True)
