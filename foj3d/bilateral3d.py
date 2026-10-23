"""4D bilateral-space operators on 3D volumes (pure PyTorch).

Implements the Barron & Poole "Fast Bilateral Solver" (ECCV 2016) grid for
volumetric data: voxels are splatted into the occupied cells of a sparse 4D
bilateral grid (z, y, x, intensity of a *guide* volume), blurred with a
separable normalized [1,2,1] kernel, bistochastized (20 Sinkhorn iterations),
and used as a graph-Laplacian regularizer.

Operators
  S       : splat  (voxel -> cell scatter-add, compact occupied-cell index)
  S^T     : slice  (cell  -> voxel gather)
  B       : normalized separable blur on the dense 4D grid of occupied cells
  m = S 1 : splatted mass (voxel count per cell)
  n       : bistochastization factor, n * (B n) ~= m

Voxel-space regularizer: the FBS filter applied to the splatted signal is
    W_hat = S^T Dm^-1 Dn B Dn Dm^-1 S     (row-stochastic after bistoch.)
and the voxel graph Laplacian is
    L = I - W_hat
    apply_L(v) = v - S^T( n * B(n * (S v) / m) / m )
Eigenvalues in [0, 2]; nullspace = constants (exactly, in the
bistochastized limit). A voxel vector that is zero-mean inside every cell
has S v = 0, so L v = v (such fluctuations are fully penalized).
"""

import numpy as np
import torch
import torch.nn.functional as F


def _scatter(src, index, size):
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def _blur_axis(t, ax):
    """Normalized [1,2,1] convolution along one axis, zero padding."""
    nd = t.ndim
    pad = [0, 0] * nd
    i = 2 * (nd - 1 - ax)          # F.pad consumes last dim first
    pad[i], pad[i + 1] = 1, 1
    tp = F.pad(t, tuple(pad))
    sl = [slice(None)] * nd
    sl[ax] = slice(0, -2)
    a = tp[tuple(sl)]
    sl[ax] = slice(1, -1)
    b = tp[tuple(sl)]
    sl[ax] = slice(2, None)
    c = tp[tuple(sl)]
    return 0.25 * (a + c) + 0.5 * b


class BilateralGrid3D:
    """Sparse 4D bilateral grid built from a 3D guide volume.

    Cell coordinates: (round(z/ss), round(y/ss), round(x/ss),
    round((g - g_min)/sr)). Only occupied cells are stored.
    """

    def __init__(self, guide, sigma_s, sigma_r, bistoch_iters=20, device=None):
        g = torch.as_tensor(guide, dtype=torch.float32)
        if device is not None:
            g = g.to(device)
        self.device = g.device
        D, H, W = g.shape
        self.shape = g.shape
        gmin = float(g.min())
        self.gmin, self.gmax = gmin, float(g.max())

        zz, yy, xx = torch.meshgrid(
            torch.arange(D, device=g.device),
            torch.arange(H, device=g.device),
            torch.arange(W, device=g.device), indexing='ij')
        cz = torch.round(zz.float() / sigma_s).long()
        cy = torch.round(yy.float() / sigma_s).long()
        cx = torch.round(xx.float() / sigma_s).long()
        ci = torch.round((g - gmin) / sigma_r).clamp_min(0).long()

        nz, ny, nx = int(cz.max()) + 1, int(cy.max()) + 1, int(cx.max()) + 1
        nr = int(ci.max()) + 1
        self.dims = (nz, ny, nx, nr)

        lin = (((cz * ny + cy) * nx + cx) * nr + ci).reshape(-1)
        uniq, inv = torch.unique(lin, return_inverse=True)
        self.cell_index = inv                # [N] compact cell id per voxel
        self.lin_cells = uniq                # [K] dense flat index per cell
        self.K = int(uniq.numel())
        self.N = int(inv.numel())

        self.m = torch.bincount(inv, minlength=self.K).to(torch.float32)

        n = torch.ones(self.K, device=g.device)
        for _ in range(bistoch_iters):
            n = torch.sqrt(n * self.m / self.blur(n).clamp_min(1e-9))
        self.n = n

        # diagonal of Dm - Dn B Dn (B self-weight = (1/2)^4 = 1/16)
        self.lap_diag = self.m - self.n * self.n / 16.0

    def splat(self, v):
        """S v: voxel field -> cell field."""
        return _scatter(v.reshape(-1).to(torch.float32), self.cell_index, self.K)

    def slice_(self, y):
        """S^T y: cell field -> voxel field."""
        return y[self.cell_index].reshape(self.shape)

    def blur(self, vals):
        """B v: normalized separable blur over the dense 4D grid of cells."""
        dense = torch.zeros(int(np.prod(self.dims)), dtype=torch.float32,
                            device=self.device)
        dense[self.lin_cells] = vals.to(torch.float32)
        t = dense.reshape(self.dims)
        for ax in range(4):
            t = _blur_axis(t, ax)
        return t.reshape(-1)[self.lin_cells]

    def apply_What(self, v):
        """W_hat v = S^T Dm^-1 Dn B Dn Dm^-1 S v (row-stochastic filter)."""
        sv = self.splat(v)
        return self.slice_(self.n * self.blur(self.n * sv / self.m) / self.m)

    def apply_L(self, v):
        """L v = v - W_hat v (voxel-space graph Laplacian)."""
        return v - self.apply_What(v)


def _cg(Aop, b, M_inv_diag, iters, x0=None, rtol=0.0):
    """Preconditioned conjugate gradient on Aop y = b (flat tensors)."""
    y = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - Aop(y)
    z = r * M_inv_diag
    p = z.clone()
    rz = float((r * z).sum())
    nb = float((b * b).sum().sqrt())
    for it in range(iters):
        Ap = Aop(p)
        alpha = rz / max(float((p * Ap).sum()), 1e-30)
        y = y + alpha * p
        r = r - alpha * Ap
        if rtol > 0 and float((r * r).sum().sqrt()) < rtol * nb:
            break
        z = r * M_inv_diag
        rz_new = float((r * z).sum())
        p = z + (rz_new / max(rz, 1e-30)) * p
        rz = rz_new
    return y


def bilateral_solve(target, confidence, guide, lam, sigma_s, sigma_r,
                    iters=50, device=None):
    """Fast Bilateral Solver denoiser in grid space.

    Solves  (lam * (Dm - Dn B Dn) + diag(S c)) y = S (c * t)
    by Jacobi-preconditioned CG; returns x = S^T y on the voxel grid.

    target/confidence/guide: [D,H,W] numpy or torch arrays.
    """
    if device is None:
        if torch.is_tensor(guide):
            device = guide.device
        elif torch.is_tensor(target):
            device = target.device
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    grid = BilateralGrid3D(guide, sigma_s, sigma_r, device=device)
    t = torch.as_tensor(target, dtype=torch.float32, device=device).reshape(-1)
    c = torch.as_tensor(confidence, dtype=torch.float32,
                        device=device).reshape(-1)

    S_c = _scatter(c, grid.cell_index, grid.K)
    S_ct = _scatter(c * t, grid.cell_index, grid.K)

    def Aop(y):
        return lam * (grid.m * y - grid.n * grid.blur(grid.n * y)) + S_c * y

    # Jacobi preconditioner: diag(A) = lam*(m - n^2/16) + S_c
    diag = lam * grid.lap_diag + S_c
    M_inv = 1.0 / diag.clamp_min(diag.max().item() * 1e-8 + 1e-30)

    y = _cg(Aop, S_ct, M_inv, iters)
    return grid.slice_(y)
