"""Oracle analysis: does spatially-varying patch size have headroom?

Loads foj256_P50_R{6,10,12}.npy + clean GT. For every voxel computes which
R was locally best (per-voxel MSE argmin is too noisy -> evaluate per
*patch cell* of the R6/s2 grid: mean SE of each output inside each cell).

Outputs: oracle upper bound, per-cell best-R map stats, spatial coherence.
"""
import json, sys
import numpy as np

sys.path.insert(0, '/content/3D-Field-of-Junctions')
import metrics

clean = np.load('examples/image_of_engine/cone_ntrain_25_angle_360/2_engine_cone/vol_gt.npy').astype(np.float32)
outs = {R: np.load(f'foj256_P50_R{R}.npy') for R in (6, 10, 12)}

# cell size ~ patch grid cell of R6/stride2 -> 6^3 voxel cells is too local;
# use 16^3 blocks (~patch scale) for reliable per-cell comparison
B = 16
D = clean.shape[0]
nb = D // B
res = {}
stack = np.stack([outs[R] for R in (6, 10, 12)])           # [3,D,D,D]
se = (stack - clean[None]) ** 2
# per-block mean SE -> [3, nb, nb, nb]
se_blk = se.reshape(3, nb, B, nb, B, nb, B).mean(axis=(2, 4, 6))
best = np.argmin(se_blk, axis=0)                            # [nb,nb,nb] in {0,1,2}
Rvals = np.array([6, 10, 12])
bestR = Rvals[best]

oracle_blk = np.take_along_axis(
    stack.reshape(3, nb, B, nb, B, nb, B), best[None, :, None, :, None, :, None],
    axis=0)[0].reshape(D, D, D)
res['oracle_block_psnr'] = metrics.psnr(clean, oracle_blk)
res['per_R_psnr'] = {f'R{R}': metrics.psnr(clean, outs[R]) for R in (6, 10, 12)}
res['bestR_hist'] = {str(R): int((bestR == R).sum()) for R in Rvals}
res['n_blocks'] = int(nb ** 3)

# spatial coherence: mean best-R in coarse 4x4x4 super-blocks should vary
# smoothly if adaptivity is spatial (not per-block noise)
from scipy.ndimage import uniform_filter
sm = uniform_filter(bestR.astype(np.float32), size=2)
res['bestR_spatial_std'] = float(sm.std())
res['bestR_overall_std'] = float(bestR.astype(np.float32).std())
res['coherence_ratio'] = res['bestR_spatial_std'] / max(res['bestR_overall_std'], 1e-9)

# zone check: near-surface (object boundary band) vs interior?
g = np.abs(np.gradient(clean)[0]) + np.abs(np.gradient(clean)[1]) + np.abs(np.gradient(clean)[2])
gb = g.reshape(nb, B, nb, B, nb, B).mean(axis=(1, 3, 5))
hi, lo = gb > np.median(gb), gb <= np.median(gb)
res['bestR_mean_at_edges'] = float(bestR[hi].mean())
res['bestR_mean_in_flat'] = float(bestR[lo].mean())

print(json.dumps(res, indent=2))
json.dump(res, open('oracle256.json', 'w'), indent=2)
np.save('bestR_map.npy', bestR)
np.save('oracle256.npy', oracle_blk.astype(np.float32))
print('DONE')
