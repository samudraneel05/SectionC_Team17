"""Spatially-varying photon fields P(x) and voxel-domain Poisson noise.

Noise model identical to the authors' demo: noisy = Poisson(clean * P) / P.
Only difference: P is a per-voxel field instead of a scalar.

CALIBRATION NOTE: paper's P50/P100/P1000 are photons per DETECTOR pixel in
projection space. In voxel space the equivalent severe->mild range is
~P in {3, 10, 30} photons/voxel (input ~13-25 dB on engine-64^3).
We label zones SEVERE=3, MODERATE=10, MILD=30 throughout.
"""
import numpy as np

SEVERE, MODERATE, MILD = 3.0, 10.0, 30.0
ZONES = (SEVERE, MODERATE, MILD)


def photon_field(shape, pattern, seed=None):
    """Return per-voxel photon-count field P(x) for `shape` (D,H,W)."""
    D, H, W = shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
    zz /= max(D - 1, 1)
    yy /= max(H - 1, 1)
    xx /= max(W - 1, 1)

    if pattern == 'uniform3':
        P = np.full(shape, SEVERE, np.float32)
    elif pattern == 'uniform10':
        P = np.full(shape, MODERATE, np.float32)
    elif pattern == 'uniform30':
        P = np.full(shape, MILD, np.float32)
    elif pattern == 'quadrants':
        # 2x2 across y/x in the central z-slab: SEVERE | MODERATE / MODERATE | MILD
        P = np.full(shape, MODERATE, np.float32)
        slab = (zz > 0.25) & (zz < 0.75)
        P[slab & (yy < 0.5) & (xx < 0.5)] = SEVERE
        P[slab & (yy < 0.5) & (xx >= 0.5)] = MODERATE
        P[slab & (yy >= 0.5) & (xx < 0.5)] = MODERATE
        P[slab & (yy >= 0.5) & (xx >= 0.5)] = MILD
    elif pattern == 'quadrants4':
        # four distinct levels: SEVERE, MILD, MODERATE, MILD-variant
        P = np.full(shape, MODERATE, np.float32)
        slab = (zz > 0.25) & (zz < 0.75)
        P[slab & (yy < 0.5) & (xx < 0.5)] = SEVERE
        P[slab & (yy < 0.5) & (xx >= 0.5)] = MILD
        P[slab & (yy >= 0.5) & (xx < 0.5)] = MODERATE
        P[slab & (yy >= 0.5) & (xx >= 0.5)] = SEVERE
    elif pattern == 'gradient_x':
        P = (SEVERE + (MILD - SEVERE) * xx).astype(np.float32)
    elif pattern == 'radial':
        c = 0.5
        r = np.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
        r /= r.max()
        P = (MILD - (MILD - SEVERE) * np.clip(r / 0.6, 0, 1)).astype(np.float32)
    elif pattern == 'half':
        P = np.where(xx < 0.5, SEVERE, MILD).astype(np.float32)
    else:
        raise ValueError(f'unknown pattern {pattern}')
    return P


def true_sigma(clean, P):
    """Per-voxel std of Poisson(clean*P)/P — ground-truth noise map."""
    return np.sqrt(np.clip(clean, 0, None) / P).astype(np.float32)


def add_noise(clean, P, seed=0):
    rng = np.random.default_rng(seed)
    noisy = rng.poisson(clean.astype(np.float64) * P) / P
    return np.clip(noisy, 0, 1).astype(np.float32)
