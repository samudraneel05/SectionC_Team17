"""Unit tests for bilateral3d (CPU/MPS, 48^3-128^3 volumes)."""
import time

import numpy as np
import torch

import bilateral3d as B


def phantom(n=64):
    z, y, x = np.mgrid[:n, :n, :n].astype(np.float32)
    v = 0.05 + 0.1 * z / n                      # background ramp
    v[8:24, 8:24, 8:24] = 0.4                    # box
    r2 = (z - n * 0.66) ** 2 + (y - n * 0.66) ** 2 + (x - n * 0.5) ** 2
    v[r2 < (n * 0.2) ** 2] = 0.9                 # sphere
    v[n // 2 - 4:n // 2 + 4, 40:n - 12, 12:n - 24] = 0.7   # slab
    return v.astype(np.float32)


def main():
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.manual_seed(0)
    n = 64
    g = torch.from_numpy(phantom(n)).to(dev)
    rng = torch.Generator(device='cpu').manual_seed(0)

    grid = B.BilateralGrid3D(g, sigma_s=4, sigma_r=0.05, device=dev)
    print(f'device={dev}  grid dims={grid.dims}  occupied cells={grid.K}')

    # 1. bistochastization residual  ||n * B n - m|| / ||m||
    res = (grid.n * grid.blur(grid.n) - grid.m).norm() / grid.m.norm()
    print(f'[1] bistoch ||nBn-m||/||m|| = {res.item():.3e}   '
          f'{"PASS" if res < 1e-3 else "FAIL"}')

    # 2. symmetry  <u, Lv> vs <Lu, v>
    u = torch.randn(g.shape, generator=rng).to(dev)
    v = torch.randn(g.shape, generator=rng).to(dev)
    Lv = grid.apply_L(v)
    Lu = grid.apply_L(u)
    a = float((u * Lv).sum())
    b = float((Lu * v).sum())
    rel = abs(a - b) / (float(u.norm()) * float(Lv.norm()))
    print(f'[2] symmetry |<u,Lv>-<Lu,v>|/(||u||||Lv||) = {rel:.3e}   '
          f'{"PASS" if rel < 1e-5 else "FAIL"}')

    # 3. PSD: min Rayleigh quotient over 20 random vectors
    worst = float('inf')
    for _ in range(20):
        w = torch.randn(g.shape, generator=rng).to(dev)
        q = float((w * grid.apply_L(w)).sum()) / float((w * w).sum())
        worst = min(worst, q)
    print(f'[3] PSD min <v,Lv>/||v||^2 = {worst:.3e}   '
          f'{"PASS" if worst >= -1e-6 else "FAIL"}')

    # 4. constants: L must annihilate the all-ones vector
    ones = torch.ones(g.shape, device=dev)
    L1 = grid.apply_L(ones)
    v2 = torch.randn(g.shape, generator=rng).to(dev)
    Lv2 = grid.apply_L(v2)
    c_res = float(L1.norm()) / float(ones.norm())
    v_res = float(Lv2.norm()) / float(v2.norm())
    rat = c_res / v_res
    print(f'[4] ||L1||/||1|| = {c_res:.3e}, ||Lv||/||v|| = {v_res:.3e}, '
          f'ratio = {rat:.3e}   {"PASS" if rat < 1e-3 else "FAIL"}')

    # 4a. row-stochasticity of W_hat: ||W_hat 1 - 1||_inf < 1e-2
    W1 = grid.apply_What(ones)
    wres = float((W1 - 1).abs().max())
    print(f'[4a] ||W_hat 1 - 1||_inf = {wres:.3e}   '
          f'{"PASS" if wres < 1e-2 else "FAIL"}')

    # 4b. within-cell zero-mean fluctuation is NOT in the nullspace
    v0 = torch.randn(g.shape, generator=rng).to(dev)
    cellmean = grid.slice_(grid.splat(v0) / grid.m)
    vz = v0 - cellmean                       # zero mean inside every cell
    q = float((vz * grid.apply_L(vz)).sum()) / float((vz * vz).sum())
    print(f'[4b] <vz,Lvz>/||vz||^2 = {q:.3f} (expect ~1)   '
          f'{"PASS" if q > 0.5 else "FAIL"}')

    # 4c. max eigenvalue via 30 power iterations (must be <= ~2)
    w = torch.randn(g.shape, generator=rng).to(dev)
    w = w / w.norm()
    for _ in range(30):
        w = grid.apply_L(w)
        w = w / w.norm()
    lmax = float((w * grid.apply_L(w)).sum()) / float((w * w).sum())
    print(f'[4c] lambda_max(L) ~ {lmax:.3f}   '
          f'{"PASS" if lmax <= 2 + 1e-3 else "FAIL"}')

    # 4d. tiny CG check: solve (I + lam L) x = t, guide = clean
    noisy = g + 0.2 * torch.randn(g.shape, generator=rng).to(dev)

    def psnr(ref, est):
        mse = float(((ref - est) ** 2).mean())
        return -10 * np.log10(mse)

    tflat = noisy.reshape(-1)
    x = B._cg(lambda vv: vv + 2.0 * grid.apply_L(vv.reshape(g.shape)).reshape(-1),
              tflat, torch.ones_like(tflat), iters=30).reshape(g.shape)
    p_in, p_out = psnr(g, noisy), psnr(g, x)
    print(f'[4d] CG (I+2L)x=t: PSNR {p_in:.2f} -> {p_out:.2f}   '
          f'{"PASS" if p_out > p_in else "FAIL"}')

    # 5. toy FBS denoise: clean-guide and self-guide
    p0 = p_in
    out_gt = B.bilateral_solve(noisy, torch.ones_like(noisy), g,
                               lam=1.0, sigma_s=4, sigma_r=0.05, iters=50)
    p1 = psnr(g, out_gt)
    out_self = B.bilateral_solve(noisy, torch.ones_like(noisy), noisy,
                                 lam=1.0, sigma_s=4, sigma_r=0.05, iters=50)
    p2 = psnr(g, out_self)
    ok1 = p1 - p0 > 6
    ok2 = p2 > p0
    print(f'[5] denoise PSNR in {p0:.2f} -> guide=GT {p1:.2f} '
          f'({"PASS" if ok1 else "FAIL"}) | guide=noisy {p2:.2f} '
          f'({"PASS" if ok2 else "FAIL"})')

    # timing apply_L at 128^3 (and 256^3 if memory allows)
    for size in (128, 256):
        try:
            gN = torch.rand(size, size, size, device=dev)
            gridN = B.BilateralGrid3D(gN, sigma_s=4, sigma_r=0.05, device=dev)
            vv = torch.rand(size, size, size, device=dev)
            gridN.apply_L(vv)                  # warmup
            if dev == 'mps':
                torch.mps.synchronize()
            t0 = time.time()
            for _ in range(10):
                gridN.apply_L(vv)
            if dev == 'mps':
                torch.mps.synchronize()
            dt = (time.time() - t0) / 10
            print(f'[6] apply_L at {size}^3 on {dev}: {dt * 1000:.1f} ms/call '
                  f'({gridN.K} cells, dims {gridN.dims})')
        except (RuntimeError, MemoryError) as e:
            print(f'[6] apply_L at {size}^3 on {dev}: SKIPPED ({e})')


if __name__ == '__main__':
    main()
