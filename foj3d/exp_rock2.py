"""3D FoJ on REAL grayscale rock micro-CT — Bentheim sandstone, 5.854um
(Zenodo 5548582, file '1MPa 1_5854um.zip' = 1007 TIFF slices, ~1.2GB).

The previous binary volumes had features BELOW patch scale (~2 vox granules)
so FoJ collapsed to the mean. Bentheim grains are ~10-35 voxels — the proper
test of whether FoJ works on piecewise-constant rock at the right scale.

GT = 192 central slices x central 192x192 crop -> 192^3, normalized by
percentile to [0,1].  Noise: additive Gaussian sigma {0.15, 0.30}.
Methods: noisy, gauss, median, tv{0.05,0.1,0.2}, bm4d, FoJ R{6,8,12}.
Metrics: PSNR, SSIM3D, boundary-F1.  rock2_results.json after every entry.
"""
import glob
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np

for cand in (glob.glob('/kaggle/input/**/exp_rock2.py', recursive=True) +
             glob.glob('/content/**/exp_rock2.py', recursive=True) +
             [os.path.abspath(__file__)]):
    d = os.path.dirname(cand)
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)
        break

import torch  # noqa: E402
import chunked_foj  # noqa: E402
import metrics  # noqa: E402
from strict_protocol import mkopts  # noqa: E402

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
try:
    import tifffile as _tifffile
except Exception:
    _tifffile = None

OUT = 'rock2_results.json'
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

ZIP_URL = ('https://zenodo.org/api/records/5548582/files/'
           '1MPa%201_5854um.zip/content')
CROP = 192
Z0, Z1 = 400, 592          # central 192 slices


def read_tiff(buf):
    if _tifffile is not None:
        return _tifffile.imread(io.BytesIO(buf))
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(buf)))


def load_gt():
    zpath = '/kaggle/working/bentheim.zip'
    if not os.path.exists(zpath) or os.path.getsize(zpath) < 1e9:
        print('[rock2] downloading bentheim zip (~1.2GB)...', flush=True)
        urllib.request.urlretrieve(ZIP_URL, zpath)
    zf = zipfile.ZipFile(zpath)
    names = sorted(n for n in zf.namelist()
                   if n.lower().endswith(('.tiff', '.tif')))
    names = names[Z0:Z1] if not SMOKE else names[Z0:Z0 + 64]
    print(f'[rock2] {len(names)} slices, first: {names[0]}', flush=True)
    first = read_tiff(zf.read(names[0]))
    print('[rock2] slice dtype', first.dtype, 'shape', first.shape, flush=True)
    H, W = first.shape[:2]
    h0, w0 = (H - CROP) // 2, (W - CROP) // 2
    if SMOKE:
        h0, w0 = (H - 64) // 2, (W - 64) // 2
        c = 64
    else:
        c = CROP
    vol = np.empty((len(names), c, c), np.float32)
    for i, n in enumerate(names):
        im = read_tiff(zf.read(n))
        if im.dtype == np.uint16:
            im = im.astype(np.float32) / 65535.0
        else:
            im = im.astype(np.float32) / 255.0
        vol[i] = im[h0:h0 + c, w0:w0 + c]
        if i % 64 == 0:
            print(f'  slice {i + 1}/{len(names)}', flush=True)
    v = vol / np.percentile(vol, 99.9)
    return np.clip(v, 0, 1).astype(np.float32)


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
        checkpoint_path=f'ckpt_rock2_{tag}.npz', ckpt_every=50)
    return out


NOISES = [0.40, 0.55]
RS = [6, 8, 12]
TV_W = [0.05, 0.1, 0.2]


def sweep(gt):
    rng = np.random.default_rng(0)
    for s in NOISES:
        tag = f's{int(s * 100)}'
        y = (gt + rng.normal(0, s, gt.shape)).astype(np.float32)
        np.save(f'rock2_noisy_{tag}.npy', y.astype(np.float16))
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
                save_vol=f'rock2_foj_R{R}_{tag}.npy')


if __name__ == '__main__':
    print('device:', 'cuda' if torch.cuda.is_available() else 'cpu', flush=True)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0), flush=True)
    print('bm4d:', _bm4d is not None, ' skimage:', _tv_chambolle is not None,
          ' tifffile:', _tifffile is not None, flush=True)
    gpath = 'rock2_gt.npy'
    if os.path.exists(gpath):
        gt = np.load(gpath).astype(np.float32)
    else:
        gt = load_gt()
        np.save(gpath, gt.astype(np.float16))
    print('[rock2] gt', gt.shape, 'mean', float(gt.mean()), 'std', float(gt.std()), flush=True)
    sweep(gt)
    print('ROCK2 DONE', flush=True)
