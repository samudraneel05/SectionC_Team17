#!/usr/bin/env python3
"""Build demo/foj_demo.html — a single-file interactive 3D FoJ demo.

Volumes live in demo/volumes/*.npy (128^3 float32, produced by
demo/make_volumes.py): clean engine, sigma=0.50 Gaussian-noise input,
the 3D FoJ fit, and its global boundary map. Each is quantized to uint8
and embedded as base64 inside one self-contained HTML file.
Double-click the output to run — no server needed.
"""
import base64
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = 128

STATES = ["gt", "noisy", "foj", "bnd"]


def b64(v):
    u = np.round(np.clip(v, 0, 1) * 255).astype(np.uint8)
    return base64.b64encode(u.tobytes()).decode()


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    vols = {k: np.load(HERE / "volumes" / f"{k}.npy").astype(np.float32)
            for k in STATES}
    assert all(v.shape == (N, N, N) for v in vols.values())
    lo, hi = (float(x) for x in np.percentile(vols["gt"], [0.5, 99.7]))
    html = (HERE / "template.html").read_text()
    for k in STATES:
        html = html.replace(f"__{k.upper()}__", b64(vols[k]))
    html = (html
            .replace("__LO__", f"{lo:.6f}")
            .replace("__HI__", f"{hi:.6f}")
            .replace("__PSNR_NOISY__", f"{psnr(vols['noisy'], vols['gt']):.2f}")
            .replace("__PSNR_FOJ__", f"{psnr(vols['foj'], vols['gt']):.2f}"))
    out = HERE / "foj_demo.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"window [{lo:.3f}, {hi:.3f}]  "
          f"PSNR noisy {psnr(vols['noisy'], vols['gt']):.2f}  "
          f"foj {psnr(vols['foj'], vols['gt']):.2f}")


if __name__ == "__main__":
    main()
