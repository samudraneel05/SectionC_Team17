"""3D FoJ on digital-rock volumes — denoising comparison.

GT volumes (piecewise-constant phases with sharp interfaces — the structure
the wedge prior assumes):
  'sphere' : data/SpherePack_300x300x300.raw  (uint8, synthetic sphere pack)
  'b1m1'   : data/B1M1_CT_binary_8.02um_8bu_748x748x1234.raw (uint8, real
             segmented Berea micro-CT; central 192^3 crop used for speed)
Both come from Zenodo record 15237327 (data.zip), downloaded in-kernel.

Noise: additive Gaussian sigma {0.15, 0.30} — FoJ's low-SNR niche.
Methods: noisy, gauss(s1), median3, tv{0.05,0.1,0.2}, bm4d, FoJ R{6,8,12}.
Metrics: PSNR (clipped), SSIM3D, boundary-F1 (edge preservation).
Writes rock_results.json after EVERY entry.
"""
import glob
import json
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np

for cand in (glob.glob('/kaggle/input/**/exp_rock.py', recursive=True) +
             glob.glob('/content/**/exp_rock.py', recursive=True) +
             [os.path.abspath(__file__)]):
    d = os.path.dirname(cand)
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)
        break

import torch  # noqa: E402
import chunked_foj  # noqa: E402
import metrics  # noqa: E402
from strict_protocol import mkopts  # noqa: E402

# bm4d ships as bm4d_pkg.zip inside the code dataset (Kaggle flattens dirs)
for z in glob.glob('/kaggle/input/**/bm4d_pkg.zip', recursive=True):
    dest = '/kaggle/working/bm4d_pkg'
    if not os.path.isdir(dest):
        zipfile.ZipFile(z).extractall(dest)
    if dest not in sys.path:
        sys.path.insert(0, dest)
try:
    import bm4d as _bm4d
except Exception:
    _bm4d = None
try:
    from skimage.restoration import denoise_tv_chambolle as _tv_chambolle
except Exception:
    _tv_chambolle = None

OUT = 'rock_results.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
SMOKE = '--smoke' in sys.argv


def save():
    json.dump(res, open(OUT, 'w'), indent=2)


def fin(x):
    return float(x) if np.isfinite(x) else None


def entry(gt, vol):
    v = np.clip(vol, 0, 1)
    f1, pr, rc = metrics.boundary_fscore(gt, v)
    return dict(psnr3d=fin(metrics.psnr(gt, v)),
                ssim3d=fin(metrics.ssim3d(gt, v)),
                bf1=fin(f1), bprec=fin(pr), brec=fin(rc))


def run(key, fn, save_vol=None):
    if key in res:
        print(f'  {key}: cached', flush=True)
        return
    t0 = time.time()
    out = fn()
    res[key] = dict(s=round(time.time() - t0, 1), **entry(gt, out))
    print(f'  {key}: {res[key]}', flush=True)
    save()
    if save_vol:
        np.save(save_vol, out.astype(np.float16))


# ---------------- data -------------------------------------------------------

ZIP_URL = 'https://zenodo.org/api/records/15237327/files/data.zip/content'
MEMBERS = {
    'sphere': ('data/SpherePack_300x300x300.raw', (300, 300, 300)),
    'b1m1': ('data/B1M1_CT_binary_8.02um_8bu_748x748x1234.raw', (748, 748, 1234)),
}


def get_zip(path='/kaggle/working/data.zip'):
    if not os.path.exists(path) or os.path.getsize(path) < 3e8:
        print('[rock] downloading data.zip (~322MB)...', flush=True)
        urllib.request.urlretrieve(ZIP_URL, path)
    return zipfile.ZipFile(path)


def load_gt(zf, name):
    member, shape = MEMBERS[name]
    v = np.frombuffer(zf.read(member), dtype=np.uint8).reshape(shape)
    if name == 'sphere':
        v = v[54:246, 54:246, 54:246]          # 192^3 crop
    else:
        v = v[278:470, 278:470, 480:672]       # 192^3 crop inside the volume
    v = v.astype(np.float32)
    v = v / v.max()                          # -> [0,1] (binary or grayscale)
    return v


# ---------------- denoisers --------------------------------------------------

def den_gauss(y):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(y, 1.0).astype(np.float32)


def den_median(y):
    from scipy.ndimage import median_filter
    return median_filter(y, size=3).astype(np.float32)


def den_tv(y, w):
    return _tv_chambolle(y.astype(np.float64), weight=w).astype(np.float32)


def den_foj(y, R, tag):
    out, _ = chunked_foj.chunked_foj(
        y.astype(np.float32), mkopts(R), chunk_patches=13, halo=0,
        verbose=True, clip_blocks=False, clip_output=False,
        checkpoint_path=f'ckpt_rock_{tag}.npz', ckpt_every=50)
    return out


# ---------------- experiment -------------------------------------------------

NOISES = [0.15, 0.30]
RS = [6, 8, 12]
TV_W = [0.05, 0.1, 0.2]


def sweep(gt, vol_name):
    rng = np.random.default_rng(0)
    for s in NOISES:
        tag = f'{vol_name}_s{int(s * 100)}'
        y = (gt + rng.normal(0, s, gt.shape)).astype(np.float32)
        np.save(f'rock_noisy_{tag}.npy', y.astype(np.float16))
        run(f'noisy_{tag}', lambda y=y: y)
        run(f'gauss_{tag}', lambda y=y: den_gauss(y))
        run(f'median_{tag}', lambda y=y: den_median(y))
        if _tv_chambolle is not None:
            for w in TV_W:
                run(f'tv_w{w}_{tag}', lambda y=y, w=w: den_tv(y, w))
        if _bm4d is not None:
            try:
                run(f'bm4d_{tag}', lambda y=y, s=s: np.asarray(_bm4d.bm4d(y, s), np.float32))
            except Exception as e:
                res[f'bm4d_{tag}'] = dict(error=str(e))
                save()
        else:
            res[f'bm4d_{tag}'] = dict(error='bm4d not installed')
            save()
        for R in ([6] if SMOKE else RS):
            run(f'foj_R{R}_{tag}', lambda y=y, R=R, tag=tag:
                den_foj(y, R, f'{tag}_R{R}'),
                save_vol=f'rock_foj_R{R}_{tag}.npy')


if __name__ == '__main__':
    print('device:', 'cuda' if torch.cuda.is_available() else 'cpu', flush=True)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0), flush=True)
    print('bm4d:', _bm4d is not None, ' skimage:', _tv_chambolle is not None, flush=True)
    zf = get_zip()
    print('zip entries:', zf.namelist(), flush=True)
    names = ['sphere'] if SMOKE else ['sphere', 'b1m1']
    for name in names:
        gpath = f'rock_gt_{name}.npy'
        if os.path.exists(gpath):
            gt = np.load(gpath).astype(np.float32)
        else:
            gt = load_gt(zf, name)
            np.save(gpath, gt.astype(np.float16))
        print(f'[rock] {name}: shape {gt.shape} uniq {np.unique(gt[::8, ::8, ::8])[:8]}', flush=True)
        sweep(gt, name)
    print('ROCK DONE', flush=True)
