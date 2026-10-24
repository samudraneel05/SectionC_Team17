"""Shared reproduction protocol: 256^3 volume, det 256^2, 20 views,
projection-domain Poisson noise (TIGRE CTnoise formula), CGLS.
Simulation/recon live in ct_pipeline.py (backend 'tigre' or 'torch').

angles: 'jitter' = 20 stratified-jitter non-uniform views (seed 0; the paper
says "non-uniform" without listing them), 'uniform' = R2-Gaussian linspace."""
import os
from types import SimpleNamespace

import numpy as np

import ct_pipeline

GT_PATH = 'examples/image_of_engine/cone_ntrain_25_angle_360/2_engine_cone/vol_gt.npy'
DS, DET, N_VIEW = 256, 256, 20

BASE = dict(stride=2, eta=0.01, delta=0.05, lr_angles=0.005, lr_x0y0z0=0.05,
            lambda_boundary_final=0.1, lambda_color_final=0.001, nvals=5,
            num_initialization_iters=1, num_refinement_iters=1,
            greedy_step_every_iters=50, parallel_mode=True)


def mkopts(R, **kw):
    o = dict(BASE); o['R'] = R; o.update(kw)
    return SimpleNamespace(**o)


def get_angles(mode='jitter'):
    if mode == 'uniform':
        return np.linspace(0, 2 * np.pi, N_VIEW + 1)[:-1]
    rng = np.random.default_rng(0)
    base = np.linspace(0, 2 * np.pi, N_VIEW, endpoint=False)
    jit = rng.uniform(-0.45, 0.45, N_VIEW) * (2 * np.pi / N_VIEW)
    return np.sort((base + jit) % (2 * np.pi))


angles20 = get_angles

OBJECTS = {'engine': GT_PATH,
           'foot': 'gt/foot/vol_gt.npy',
           'jaw': 'gt/jaw/vol_gt.npy',
           'pepper': 'gt/pepper/vol_gt.npy',
           'teapot': 'gt/teapot/vol_gt.npy'}


def load_gt(obj='engine'):
    return np.load(OBJECTS[obj]).astype(np.float32)


def recon_snapshots(iters, P_n=50.0, seed=0, obj='engine', backend='tigre',
                    angles='jitter',
                    cache_fmt='recon_{b}_{a}_{o}_P{P}_s{s}_cg{k}.npy'):
    """Unclamped CGLS volumes after k iterations for k in `iters`, cached."""
    paths = {k: cache_fmt.format(b=backend, a=angles, o=obj, P=int(P_n), s=seed, k=k)
             for k in iters}
    if all(os.path.exists(p) for p in paths.values()):
        return {k: np.load(p) for k, p in paths.items()}
    ang = get_angles(angles)
    proj = ct_pipeline.simulate(load_gt(obj), ang, P_n, seed, det=DET, backend=backend)
    out = ct_pipeline.cgls(proj, ang, iters, nvox=DS, det=DET, backend=backend)
    for k, v in out.items():
        np.save(paths[k], v)
    return out
