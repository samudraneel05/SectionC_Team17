"""Chunked FoJ3D driver — runs a dense stride-`s` patch grid as many small,
independent FoJ optimizations and coverage-weighted-blends the results.

Adapted from upstream `scripts/generate_demo_volume.py` (authors' own MPS-safe
driver), plus a **halo** so chunk borders keep full patch coverage:
each block extends `halo` voxels past its assigned patch grid, neighbouring
blocks overlap, and coverage-weighted accumulation blends them seamlessly.

`opts_fn` may return a different config per chunk (e.g., per-chunk patch size
R from a noise map) — the entry point for the adaptive extension.
"""
import gc
import time
from types import SimpleNamespace

import numpy as np
import os
import torch

import field_of_junctions3d as foj_module


def foj_on_block(volume_hwc, opts, clip=True):
    """Run FoJ on one [D,H,W,C] numpy block; returns (regions, boundaries, coverage)."""
    model = foj_module.FieldOfJunctions3D(volume_hwc, opts)
    for iteration in range(model.num_iters):
        model.step(iteration)
    regions = model.global_volume[0, 0].detach().cpu().numpy()
    if clip:
        regions = np.clip(regions, 0, 1)
    boundaries = np.clip(model.global_boundaries[0, 0].detach().cpu().numpy(), 0, 1)
    coverage = model.num_patches.detach().cpu().numpy()
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return regions, boundaries, coverage


def chunked_foj(volume, opts_template, chunk_patches=26, roi=None,
                opts_fn=None, halo=None, verbose=True,
                checkpoint_path=None, ckpt_every=50, clip_blocks=True,
                clip_output=True):
    """
    volume        : [D,H,W] or [D,H,W,1] float32 array, values ~[0,1]
    opts_template : SimpleNamespace with the FoJ fields (R, stride, eta, ...)
    chunk_patches : patch-grid cells per chunk axis (26 in the authors' demo)
    roi           : optional ((z0,z1),(y0,y1),(x0,x1)) voxel box to optimize
    opts_fn       : optional fn(d_patch0, h_patch0, w_patch0) -> opts
                    (may return a modified config, e.g. different R)
    halo          : context voxels added on every side of each block
                    (default = opts_template.R; border coverage fix)
    checkpoint_path : optional .npz — saves accumulators + done-count every
                    `ckpt_every` chunks and resumes from it if present
                    (survives Colab session preemption)
    returns       : (denoised_full_volume, boundary_volume)
    """
    if volume.ndim == 3:
        volume = volume[..., None]
    volume = volume.astype(np.float32)

    if roi is None:
        roi = tuple((0, s) for s in volume.shape[:3])
    roi_slices = tuple(slice(a, b) for a, b in roi)
    target = volume[roi_slices]
    TD, TH, TW = target.shape[:3]

    R0, stride = opts_template.R, opts_template.stride
    if halo is None:
        halo = R0

    axis_grids = []
    for length in target.shape[:3]:
        n_patches = max(0, (length - R0) // stride + 1)
        starts = list(range(0, n_patches, chunk_patches))
        axis_grids.append((n_patches, starts))

    total = int(np.prod([len(g[1]) for g in axis_grids]))
    acc_regions = np.zeros(target.shape[:3], np.float32)
    acc_bounds = np.zeros(target.shape[:3], np.float32)
    acc_weight = np.zeros(target.shape[:3], np.float32)
    n_done = 0
    skip_upto = -1
    if checkpoint_path and os.path.exists(checkpoint_path):
        ck = np.load(checkpoint_path)
        acc_regions, acc_bounds, acc_weight = ck['acc_regions'], ck['acc_bounds'], ck['acc_weight']
        skip_upto = int(ck['n_done']) - 1       # chunks 0..skip_upto already done
        n_done = skip_upto + 1
        print(f'[ckpt] resuming after chunk {n_done}', flush=True)
    t_start = time.time()

    chunk_idx = -1

    for pz0 in axis_grids[0][1]:
        nz = min(chunk_patches, axis_grids[0][0] - pz0)
        z0, z1 = pz0 * stride, pz0 * stride + R0 + (nz - 1) * stride
        for py0 in axis_grids[1][1]:
            ny = min(chunk_patches, axis_grids[1][0] - py0)
            y0, y1 = py0 * stride, py0 * stride + R0 + (ny - 1) * stride
            for px0 in axis_grids[2][1]:
                nx = min(chunk_patches, axis_grids[2][0] - px0)
                x0, x1 = px0 * stride, px0 * stride + R0 + (nx - 1) * stride
                chunk_idx += 1
                if chunk_idx <= skip_upto:
                    continue
                n_done = chunk_idx + 1

                opts = opts_fn(pz0, py0, px0) if opts_fn else opts_template
                R = opts.R

                # haloed block extents (clamped to the ROI)
                z0h, z1h = max(0, z0 - halo), min(TD, z1 + halo)
                y0h, y1h = max(0, y0 - halo), min(TH, y1 + halo)
                x0h, x1h = max(0, x0 - halo), min(TW, x1 + halo)
                block = target[z0h:z1h, y0h:y1h, x0h:x1h]
                if min(block.shape[:3]) < R:
                    # haloed block smaller than a patch — skip (shouldn't happen)
                    continue

                regions, bounds, cover = foj_on_block(block, opts, clip=clip_blocks)
                acc_regions[z0h:z1h, y0h:y1h, x0h:x1h] += regions * cover
                acc_bounds[z0h:z1h, y0h:y1h, x0h:x1h] += bounds * cover
                acc_weight[z0h:z1h, y0h:y1h, x0h:x1h] += cover
                if checkpoint_path and n_done % ckpt_every == 0:
                    np.savez_compressed(checkpoint_path, acc_regions=acc_regions,
                                        acc_bounds=acc_bounds,
                                        acc_weight=acc_weight, n_done=n_done)
                if verbose:
                    el = time.time() - t_start
                    print(f'chunk {n_done}/{total} R={opts.R} '
                          f'z{z0h}:{z1h} y{y0h}:{y1h} x{x0h}:{x1h} '
                          f'({el:.0f}s, ~{el / n_done * (total - n_done):.0f}s left)',
                          flush=True)

    if np.any(acc_weight == 0):
        nmiss = int((acc_weight == 0).sum())
        print(f'WARNING: {nmiss} voxels uncovered (copying noisy input there)')
        missing = acc_weight == 0
        acc_regions[missing] = target[..., 0][missing]
        acc_weight[missing] = 1.0

    out = acc_regions / acc_weight
    if clip_output:
        out = np.clip(out, 0, 1)
    bounds_full = acc_bounds / acc_weight

    full = volume[..., 0].copy()
    full[roi_slices] = out
    bvol = np.zeros(volume.shape[:3], np.float32)
    bvol[roi_slices] = bounds_full
    return full, bvol
