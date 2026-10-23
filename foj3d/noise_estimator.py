"""Local noise estimation: patch-wise MAD of a high-pass residual.

sigma_i = 1.4826 * median(|r_i - median(r_i)|),  r = V - G_sigma * V

Also produces a per-patch-grid sigma map aligned with the FoJ patch lattice,
which is what the adaptive config actually consumes.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter


def estimate_sigma_voxel(volume, filt_sigma=1.0, win=9):
    """Per-voxel robust noise estimate: MAD inside a win^3 neighborhood."""
    vol = volume[..., 0] if volume.ndim == 4 else volume
    r = vol - gaussian_filter(vol, filt_sigma)
    med = median_filter(r, size=win)
    mad = median_filter(np.abs(r - med), size=win)
    return (1.4826 * mad).astype(np.float32)


def sigma_to_patch_grid(sigma_map, R, stride):
    """Aggregate per-voxel sigma onto the FoJ patch grid [D',H',W'].

    Patch (i,j,k) covers voxels [i*s, i*s+R) x ... ; sigma_i = mean of the
    voxel estimates inside that footprint.
    """
    D, H, W = sigma_map.shape
    Dp, Hp, Wp = (D - R) // stride + 1, (H - R) // stride + 1, (W - R) // stride + 1
    out = np.zeros((Dp, Hp, Wp), np.float32)
    for i in range(Dp):
        for j in range(Hp):
            for k in range(Wp):
                blk = sigma_map[i * stride:i * stride + R,
                                j * stride:j * stride + R,
                                k * stride:k * stride + R]
                out[i, j, k] = blk.mean()
    return out


def normalize01(x, lo=None, hi=None):
    lo = np.percentile(x, 2) if lo is None else lo
    hi = np.percentile(x, 98) if hi is None else hi
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1).astype(np.float32)
