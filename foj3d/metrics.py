"""Evaluation metrics — implemented with numpy/scipy so no new deps are needed.

PSNR, slice-wise SSIM-3D, boundary F-score, and per-noise-zone breakdowns.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


def psnr(ref, est, data_range=1.0):
    mse = np.mean((ref - est) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))


def ssim3d(ref, est, win=7, data_range=1.0):
    """Volumetric SSIM via 3D Gaussian-weighted local statistics."""
    ref = np.asarray(ref, np.float64)
    est = np.asarray(est, np.float64)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    f = lambda x: gaussian_filter(x, 1.5, truncate=(win - 1) / 6)
    mu_r, mu_e = f(ref), f(est)
    mu_r2, mu_e2, mu_re = mu_r * mu_r, mu_e * mu_e, mu_r * mu_e
    s_r = f(ref * ref) - mu_r2
    s_e = f(est * est) - mu_e2
    s_re = f(ref * est) - mu_re
    ssim_map = ((2 * mu_re + C1) * (2 * s_re + C2)) / ((mu_r2 + mu_e2 + C1) * (s_r + s_e + C2))
    return float(ssim_map.mean())


def boundary_fscore(ref, est, thresh=None, tol=1):
    """F1 over gradient-magnitude edge voxels (within `tol` voxel dilation)."""
    from scipy.ndimage import sobel, maximum_filter

    def edges(v):
        g = np.sqrt(sum(sobel(v, axis=a) ** 2 for a in range(3)))
        if thresh is None:
            thresh_ = np.percentile(g, 97.5)
        else:
            thresh_ = thresh
        return g > thresh_

    e_ref = maximum_filter(edges(ref), size=2 * tol + 1)
    e_est = maximum_filter(edges(est), size=2 * tol + 1)
    t_ref, t_est = edges(ref), edges(est)
    tp_p = (t_est & e_ref).sum()
    tp_r = (t_ref & e_est).sum()
    prec = tp_p / max(t_est.sum(), 1)
    rec = tp_r / max(t_ref.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return float(f1), float(prec), float(rec)


def per_zone_metrics(clean, est, P_field, zones=(3.0, 10.0, 30.0)):
    """PSNR/SSIM restricted to voxels whose photon level matches each zone."""
    rows = {}
    for z in zones:
        m = P_field == z
        if m.sum() < 100:
            continue
        rows[f'P{z}'] = {
            'psnr': psnr(clean[m], est[m]),
            'frac_voxels': float(m.mean()),
        }
    return rows
