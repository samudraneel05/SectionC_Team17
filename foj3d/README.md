# FoJ3D — local working copy

Reconstructed from `github.com/voilalab/3D-Field-of-Junctions` @ main (arXiv:2603.02149, ECCV 2026).
Files were fetched verbatim via the GitHub raw endpoint; not a git checkout.

## Files
- `field_of_junctions3d.py` — upstream core. **Patched:** device selection now `cuda → mps → cpu`.
- `unfold3d.py` — upstream verbatim (grouped `conv3d`; works on MPS).
- `fold3d.py` — upstream + **MPS patch**: `F.conv_transpose3d` is unsupported on MPS, so on
  `device.type == 'mps'` we take a mathematically identical `index_add` scatter path
  (verified bit-exact vs `conv_transpose3d`). CUDA/CPU path unchanged.
- `smoke_test.py` — 64³ cube + voxel-domain Poisson noise, FoJ denoise, PSNR in/out.

## How to run on this Mac
torch is NOT in a venv — it lives in `~/Library/Python/3.12/lib/python/site-packages`
(torch 2.6.0, MPS-enabled). Run with pyenv 3.12.11 + PYTHONPATH:

```bash
cd foj3d
PYTHONPATH=$HOME/Library/Python/3.12/lib/python/site-packages \
  ~/.pyenv/versions/3.12.11/bin/python smoke_test.py
```

Everything needed except `scikit-image` already imports this way
(torch 2.6.0, numpy, scipy, matplotlib, tqdm, cv2). Install skimage later:
`~/.pyenv/versions/3.12.11/bin/python -m pip install --user scikit-image` (lands in
`~/.local/lib/python3.12/site-packages`, picked up automatically).

## Measured on Apple M2 / 8 GB (MPS)
- 64³, R=8, s=2, nvals=5, 1 init + 1 refine iter: **~7 min total, peak 1.68 GB**
- init step ≈ 4× a refine step (coordinate-descent over 12 param groups × nvals)
- Rough extrapolation: 128³ ≈ 1.5–2 h/run; 256³ needs the authors' chunked driver
  (`scripts/generate_demo_volume.py` upstream) or Colab T4 (~1 h/run).

## Upstream assets not yet downloaded (network was down)
- `examples/image_of_engine/.../vol_gt.npy` — engine clean volume (67 MB, in repo)
- `docs/static/data/engine-ct-p{20,50,100}-256.bin.gz` — authors' demo I/O
- `scripts/generate_demo_volume.py` — chunked MPS driver (content captured in PLAN notes)
- `examples/foj_3D_Demo.ipynb`
- Other R²-Gaussian volumes → Google Drive `1YZ3w87XrCNyjDRos6gkY8zgT5hESl-PN`
  (fetch `vol_gt.npy` per case on Colab, not here).
