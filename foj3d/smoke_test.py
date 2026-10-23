"""Smoke test: 64^3 cube + Poisson noise -> FoJ3D on MPS/CPU.

Verifies installation and measures per-iteration cost so we can
extrapolate full-volume runtimes on this machine.
"""
import sys, time
import numpy as np
import torch
from types import SimpleNamespace

sys.path.insert(0, '.')
import field_of_junctions3d as foj_module

print('device:', foj_module.dev)

N = 64
rng = np.random.default_rng(0)
clean = np.zeros((N, N, N, 1), dtype=np.float32)
clean[16:48, 16:48, 16:48, 0] = 1.0          # solid cube
clean[30:40, 30:56, 10:30, 0] = 0.5         # internal plate

P = 3.0                                      # severe noise regime
noisy = np.clip(rng.poisson(clean * P) / P, 0, 1).astype(np.float32)
mse = ((noisy - clean) ** 2).mean()
print(f'input PSNR: {-10*np.log10(mse):.2f} dB')

opts = SimpleNamespace(
    R=8, stride=2,
    eta=0.01, delta=0.05,
    lr_angles=0.005, lr_x0y0z0=0.05,
    lambda_boundary_final=0.1, lambda_color_final=0.001,
    nvals=5,
    num_initialization_iters=1, num_refinement_iters=1,
    greedy_step_every_iters=50,
    parallel_mode=True,
)

t0 = time.time()
foj = foj_module.FieldOfJunctions3D(noisy, opts)
print(f'setup: {time.time()-t0:.1f}s  patches={foj.D_patches}^3')

for i in range(foj.num_iters):
    t0 = time.time()
    foj.step(i)
    kind = 'init' if i < opts.num_initialization_iters else 'refine'
    print(f'step {i} ({kind}): {time.time()-t0:.1f}s')

out = foj.global_volume[0].permute(1, 2, 3, 0).detach().cpu().numpy()
mse = ((out - clean) ** 2).mean()
print(f'output PSNR: {-10*np.log10(mse):.2f} dB')
if foj_module.dev.type == 'mps':
    print(f'peak MPS alloc: {torch.mps.current_allocated_memory()/1e9:.2f} GB')
