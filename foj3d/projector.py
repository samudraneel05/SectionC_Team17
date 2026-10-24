"""Cone-beam CT projector + CGLS in pure PyTorch (CUDA/MPS/CPU).

Geometry (from R2-Gaussian meta_data.json, normalized [-1,1] volume coords):
    source at distance DSO from origin, detector plane at distance DSD on the
    far side, detector half-extent det_half, nDetector pixels, views rotate
    about the z axis (axial slices share projection geometry).

Layered cone-beam model: for a view angle theta, rotate the sampling coords
so the beam direction is the volume's x axis. A voxel at beam-depth x_rot maps
to detector lateral position u_det = u_vol * (DSO+DSD)/(DSO+x_rot). Building
that per-depth magnification grid and trilinear-summing gives the forward
projection; the exact adjoint comes from autograd.

Noise model (paper protocol): detector counts y ~ Poisson(P_n * exp(-p)),
noisy line integrals phat = -log(max(y,1)/P_n).
"""
import numpy as np
import torch
import torch.nn.functional as F


def _view_grid(theta, vol_shape, det_hw, DSO, DSD, det_half, device,
               z_chunk=32):
    """Yield sampling grids for z-batches of one view.

    Returns list of grids each [1, Zc, det, det, 3] in grid_sample (x,y,z)
    convention, plus the per-z obliquity weights.
    """
    D, H, W = vol_shape
    ct, st = np.cos(theta), np.sin(theta)
    # beam direction in the rotated frame = +x_rot. TIGRE convention: source
    # at x_rot = -DSO, detector plane at x_rot = DSD - DSO (DSD = source-det).
    u = torch.linspace(-det_half, det_half, det_hw, device=device)   # det col
    v = torch.linspace(-det_half, det_half, det_hw, device=device)   # det row
    n_depth = int(np.ceil(D * np.sqrt(2)))       # rotated cube spans ±sqrt(2)
    zvol = torch.linspace(-np.sqrt(2), np.sqrt(2), n_depth, device=device)
    scale = DSD / (DSO + zvol)      # magnification of depth plane x_rot
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    oblique = torch.sqrt(1 + (uu ** 2 + vv ** 2) / DSD ** 2)  # ds/dx along ray
    grids, weights = [], []
    for z0 in range(0, n_depth, z_chunk):
        z1 = min(z0 + z_chunk, n_depth)
        zr = zvol[z0:z1]
        sc = scale[z0:z1].view(-1, 1, 1)
        # rotated volume coords of detector pixel (u,v) at depth z_rot
        xr = zr.view(-1, 1, 1).expand(-1, det_hw, det_hw)
        yr = uu.unsqueeze(0) / sc
        zr2 = vv.unsqueeze(0) / sc
        # rotate (x_rot,y_rot,z_rot) back to volume coords about z axis
        xv = ct * xr - st * yr
        yv = st * xr + ct * yr
        zv = zr2
        g = torch.stack([xv, yv, zv], dim=-1).unsqueeze(0)  # [1,Zc,det,det,3]
        grids.append(g)
        weights.append(oblique.unsqueeze(0))
    return grids, weights


def forward_project(vol, angles, det_hw=None, DSO=5.0, DSD=7.0, det_half=2.0,
                    z_chunk=32):
    """vol [1,1,D,H,W] -> projections [n_views, det, det] (line integrals)."""
    dev = vol.device
    D, H, W = vol.shape[2:]
    det_hw = det_hw or H * 2
    projs = []
    n_depth = int(np.ceil(D * np.sqrt(2)))
    dz = 2.0 * np.sqrt(2) / (n_depth - 1)   # world step along beam axis
    for th in angles:
        acc = None
        grids, weights = _view_grid(th, (D, H, W), det_hw, DSO, DSD,
                                    det_half, dev, z_chunk)
        for g, w in zip(grids, weights):
            s = F.grid_sample(vol, g, mode='bilinear', padding_mode='zeros',
                              align_corners=True)
            s = (s[0, 0] * w).sum(0)  # [det,det]
            acc = s if acc is None else acc + s
        projs.append(acc * dz)
    return torch.stack(projs)


def apply_At(y, vol_template, angles, det_hw, DSO=5.0, DSD=7.0, det_half=2.0):
    """Exact adjoint A^T y via autograd, one view at a time to bound VRAM."""
    out = torch.zeros_like(vol_template)
    for i, th in enumerate(angles):
        v = torch.zeros_like(vol_template, requires_grad=True)
        p = forward_project(v, [th], det_hw, DSO, DSD, det_half)
        out += torch.autograd.grad(p, v, y[i:i + 1])[0]
    return out


def cgls(b, angles, vol_shape, det_hw=None, n_iters=15, DSO=5.0, DSD=7.0,
         det_half=2.0, verbose=True, snapshots=None):
    """CGLS for min ||Ax-b||^2. b: [n,det,det] noisy line integrals.
    snapshots: optional iterable of iteration counts -> returns
    {k: unclamped x after k iterations (cpu)} instead of the clamped final x."""
    snaps = {}
    dev = b.device
    det_hw = det_hw or b.shape[1]
    x = torch.zeros(1, 1, *vol_shape, device=dev)
    r = b - forward_project(x, angles, det_hw, DSO, DSD, det_half)
    s = apply_At(r, x, angles, det_hw, DSO, DSD, det_half)
    p = s.clone()
    gamma = (s * s).sum()
    for it in range(n_iters):
        Ap = forward_project(p, angles, det_hw, DSO, DSD, det_half)
        alpha = gamma / (Ap * Ap).sum().clamp_min(1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        s = apply_At(r, x, angles, det_hw, DSO, DSD, det_half)
        gamma_new = (s * s).sum()
        p = s + (gamma_new / gamma.clamp_min(1e-12)) * p
        gamma = gamma_new
        if verbose:
            res = (r * r).sum().sqrt().item()
            print(f'  cgls it{it}: ||r||={res:.3f}', flush=True)
        if snapshots is not None and (it + 1) in snapshots:
            snaps[it + 1] = x.detach().cpu().clone()
    return snaps if snapshots is not None else torch.clamp(x, 0, 1)


def add_poisson_noise(proj_clean, P_n, seed=0):
    """Paper noise model: y ~ Poisson(P_n*exp(-p)) -> phat=-log(y/P_n).
    proj_clean [n,det,det] line integrals, P_n scalar or [n,det,det] field.
    Returns (noisy_integrals, photon_counts)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    P = torch.as_tensor(P_n, dtype=torch.float32)        # scalar or field
    expected = torch.exp(-proj_clean.cpu()) * P
    y = torch.poisson(expected, generator=g)
    phat = -torch.log(y.clamp_min(1.0) / P)
    return phat.to(proj_clean.device), y.to(proj_clean.device)
