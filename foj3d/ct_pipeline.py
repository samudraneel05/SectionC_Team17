"""CT simulation + CGLS following the R2-Gaussian data generator
(data_generator/synthetic_dataset/generate_data.py, utils/ct_utils.py):

  geometry : cone, DSD=7 (source-detector), DSO=5, sVoxel=2, nVoxel=256^3,
             sDetector=4 x 4, accuracy=0.5; nDetector=det (paper: 256)
  forward  : tigre.Ax(vol.transpose(2,1,0), geo, angles)[:, ::-1, :]
  noise    : TIGRE CTnoise.add formula, Poisson only (paper: "expected photon
             count of n per pixel (in air)"):
               m = max(p); I ~ Poisson(P * exp(-p/m)); I[I<=0] = 1e-6;
               p_noisy = -log(I/P) * m;  then p_noisy[p_noisy<0] = 0
             (re-implemented with a seeded numpy RNG for reproducibility)
  recon    : tigre.algorithms.cgls(p[:, ::-1, :], geo, angles, niter)
             -> transpose(2,1,0); R2-Gaussian's standard uses niter=60, no clamp.

backend='torch' uses foj3d/projector.py (validated against analytic chords)
with the same noise formula, for machines without TIGRE.
"""
import numpy as np
import torch

DSD, DSO, SDET, SVOX = 7.0, 5.0, 4.0, 2.0


def tigre_geometry(det=256, nvox=256):
    import tigre
    geo = tigre.geometry(mode='cone')
    geo.DSD, geo.DSO = DSD, DSO
    geo.nDetector = np.array([det, det])
    geo.sDetector = np.array([SDET, SDET])
    geo.dDetector = geo.sDetector / geo.nDetector
    geo.nVoxel = np.array([nvox] * 3)
    geo.sVoxel = np.array([SVOX] * 3)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.array([0, 0, 0])
    geo.offDetector = np.array([0, 0, 0])
    geo.accuracy = 0.5
    geo.filter = None
    return geo


def add_noise(proj, P, seed):
    rng = np.random.default_rng(seed)
    m = float(proj.max())
    I = rng.poisson(P * np.exp(-proj / m)).astype(np.float64)
    I[I <= 0] = 1e-6
    out = (-np.log(I / P) * m).astype(np.float32)
    out[out < 0] = 0.0
    return out


def simulate(vol, angles, P, seed, det=256, backend='tigre'):
    """vol [D,H,W] numpy in R2-Gaussian layout -> noisy projections (backend layout)."""
    if backend == 'tigre':
        import tigre
        geo = tigre_geometry(det, vol.shape[0])
        proj = tigre.Ax(np.transpose(vol, (2, 1, 0)).copy(), geo,
                        np.asarray(angles, np.float32))[:, ::-1, :]
        proj = np.ascontiguousarray(proj)
    else:
        import projector as pj
        v = torch.from_numpy(np.ascontiguousarray(vol, np.float32))[None, None].cuda()
        proj = pj.forward_project(v, angles, det_hw=det, det_half=SDET / 2).cpu().numpy()
    return add_noise(proj, P, seed)


def cgls(proj, angles, niters, nvox=256, det=256, backend='tigre'):
    """Returns {k: unclamped volume after k iterations} for k in niters."""
    out = {}
    if backend == 'tigre':
        import tigre.algorithms as algs
        geo = tigre_geometry(det, nvox)
        p = np.ascontiguousarray(proj[:, ::-1, :])
        a = np.asarray(angles, np.float32)
        for k in sorted(niters):
            res = algs.cgls(p, geo, a, k, computel2=True)
            vol = res[0] if isinstance(res, tuple) else res
            out[k] = np.ascontiguousarray(np.transpose(vol, (2, 1, 0))).astype(np.float32)
    else:
        import projector as pj
        b = torch.from_numpy(proj).cuda()
        snaps = pj.cgls(b, angles, (nvox,) * 3, det_hw=det, det_half=SDET / 2,
                        n_iters=max(niters), verbose=False, snapshots=set(niters))
        out = {k: x[0, 0].numpy().astype(np.float32) for k, x in snaps.items()}
    return out
