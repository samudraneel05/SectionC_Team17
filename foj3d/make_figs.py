"""Generate report figures from the downloaded Kaggle outputs.

Usage:
  python make_figs.py --vols /tmp/kg_rock2_out --out ../report/figures

Produces:
  fig_slices.png   mid-slice panels: GT | noisy | Gauss | Median | TV | FoJ-R6
  fig_curve.png    PSNR vs sigma for all methods (the crossover figure)
  fig_collapse.png binary-rock collapse illustration (GT vs FoJ output)
  fig_domains.png  PSNR-vs-noise summary across MRI / rock / CT regimes
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def panels(vdir, sig_tag, out):
    gt = np.load(f'{vdir}/rock2_gt.npy').astype(np.float32)
    no = np.load(f'{vdir}/rock2_noisy_{sig_tag}.npy').astype(np.float32)
    foj = np.load(f'{vdir}/rock2_foj_R6_{sig_tag}.npy').astype(np.float32)
    from scipy.ndimage import gaussian_filter, median_filter
    from skimage.restoration import denoise_tv_chambolle
    tv = denoise_tv_chambolle(np.clip(no, 0, 1).astype(np.float64), weight=0.2)
    ims = [('Ground truth', gt), (f'Noisy $\\sigma$={int(sig_tag[1:])}%',
            np.clip(no, 0, 1)),
           ('Gaussian $\\sigma$=1', np.clip(gaussian_filter(no, 1.0), 0, 1)),
           ('Median 3$^3$', np.clip(median_filter(no, 3), 0, 1)),
           ('TV $w$=0.2', np.clip(tv, 0, 1)), ('FoJ $R$=6', np.clip(foj, 0, 1))]
    z = gt.shape[0] // 2
    fig, axs = plt.subplots(1, 6, figsize=(13.2, 2.4))
    for ax, (t, im) in zip(axs, ims):
        ax.imshow(im[z], cmap='gray', vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('wrote', out)


def curve(mri_json, rock2_json, out):
    """PSNR vs sigma on Bentheim (all methods)."""
    d = json.load(open(rock2_json))
    methods = [('noisy', 'Noisy', 'k--o'), ('gauss', 'Gaussian', 'C0-s'),
               ('median', 'Median', 'C1-^'), ('tv_w0.2', 'TV $w$=0.2', 'C2-d'),
               ('tv_w0.05', 'TV $w$=0.05', 'C2:d'),
               ('foj_R6', 'FoJ $R$=6', 'C3-o'), ('foj_R12', 'FoJ $R$=12', 'C7-x')]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    sigs = [15, 30, 40, 55]
    for key, lab, sty in methods:
        ys = [d.get(f'{key}_s{s}', {}).get('psnr3d') for s in sigs]
        xs = [s for s, y in zip(sigs, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        if ys:
            ax.plot(xs, ys, sty, label=lab, ms=5, lw=1.5)
    ax.set_xlabel('Gaussian noise $\\sigma$ (%)')
    ax.set_ylabel('PSNR (dB)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('wrote', out)


def collapse(vdir, out):
    gt = np.load(f'{vdir}/rock_gt_b1m1.npy').astype(np.float32)
    foj = np.load(f'{vdir}/rock_foj_R6_b1m1_s15.npy').astype(np.float32)
    z = gt.shape[0] // 2
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (t, im, vm, vx) in zip(axs, [('Binary GT', gt, 0, 1),
                                       ('FoJ $R$=6 output', foj, 0, 1),
                                       ('FoJ output (zoomed)', foj, foj.min(),
                                        foj.max())]):
        ax.imshow(im[z], cmap='gray', vmin=vm, vmax=vx)
        ax.set_title(t, fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('wrote', out)


def domains(mri_json, rock_json, rock2_json, out):
    """FoJ-vs-best-classical PSNR gap vs input noise level, all domains."""
    m = json.load(open(mri_json))
    r2 = json.load(open(rock2_json))

    def best_classical(d, pfx, tag):
        cands = [d.get(f'{pfx}{p}_{tag}', {}).get('psnr3d')
                 for p in ('gauss', 'median', 'tv_w0.05', 'tv_w0.1', 'tv_w0.2')]
        return max((x for x in cands if x is not None), default=None)

    pts = []
    for s in [3, 5, 7, 9, 13]:
        k = m.get(f'e1_foj_R6_s{s}', {}).get('psnr3d')
        b = best_classical(m, 'e1_', f's{s}')
        nz = m.get(f'e1_noisy_s{s}', {}).get('psnr3d')
        if k and b and nz:
            pts.append((nz, k - b, 'MRI (BrainWeb)'))
    for s in [15, 30, 40, 55]:
        k = r2.get(f'foj_R6_s{s}', {}).get('psnr3d')
        b = best_classical(r2, '', f's{s}')
        nz = r2.get(f'noisy_s{s}', {}).get('psnr3d')
        if k and b and nz:
            pts.append((nz, k - b, 'Micro-CT (Bentheim)'))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for dom in sorted(set(p[2] for p in pts)):
        xy = sorted([(x, y) for x, y, d2 in pts if d2 == dom])
        ax.plot([x for x, _ in xy], [y for _, y in xy], 'o-', label=dom)
    ax.axhline(0, color='k', lw=1, ls='--')
    ax.set_xlabel('Noisy-input PSNR (dB)')
    ax.set_ylabel('FoJ minus best classical (dB)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--vols', default='/tmp/kg_rock2_out')
    ap.add_argument('--rockvols', default='/tmp/kg_rock_out')
    ap.add_argument('--mri', default='/tmp/kg_res/mri_results.json')
    ap.add_argument('--rock', default='/tmp/kg_rock_out/rock_results.json')
    ap.add_argument('--rock2', default='/tmp/kg_rock2_out/rock2_results.json')
    ap.add_argument('--sig', default='s30')
    ap.add_argument('--out', default='../report/figures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    curve(a.mri, a.rock2, f'{a.out}/fig_curve.png')
    domains(a.mri, a.rock, a.rock2, f'{a.out}/fig_domains.png')
    if os.path.exists(f'{a.vols}/rock2_gt.npy'):
        panels(a.vols, a.sig, f'{a.out}/fig_slices.png')
    if os.path.exists(f'{a.rockvols}/rock_gt_b1m1.npy'):
        collapse(a.rockvols, f'{a.out}/fig_collapse.png')
