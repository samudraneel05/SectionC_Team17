# 3D Field of Junctions: Evaluation on New Denoising Domains

IPCV course project — Section C, Team 17.

We take the publicly released **3D Field of Junctions (3D FoJ)**
implementation (Moeini et al., ECCV 2026, arXiv:2603.02149) and evaluate
it on volumetric denoising domains the original paper did not test:
grayscale porous-media micro-CT (Bentheim sandstone), an MRI phantom
(BrainWeb T1), and two binary porous-media volumes. On micro-CT, FoJ
outperforms Gaussian, median, and TV filtering once the input drops
below ~11 dB PSNR (+3.2 dB at the highest noise tested); the other two
domains fail because their structure is smooth or below the patch scale.

## Reports

- **`report/applications.pdf`** — the focused project report (4–5 pages:
  Abstract, Introduction, Methodology, Results and Discussion,
  Conclusion). Build: `cd report && tectonic applications.tex`
- **`report/report.pdf`** — the extended study (mechanism analysis +
  CT-pipeline reproduction). Build: `cd report && tectonic report.tex`
- `research/separability_analysis.md` — code-level notes behind the
  mechanism claims in the extended report.

## Repository layout

```
foj3d/      code: patched release copy + experiment drivers
report/     LaTeX sources, compiled PDFs, figures
data/       result JSONs + DATASETS.md (links to all volumes)
research/   notes and run logs
```

## Implementation details

- `foj3d/field_of_junctions3d.py`, `foj3d/fold3d.py` — the authors'
  released FoJ model (device selection patched for Apple-silicon MPS;
  upstream originals kept in `foj3d/upstream/` for diffing).
- `foj3d/chunked_foj.py` — halo-padded chunked FoJ driver; splits the
  volume into overlapping chunks so 256³ volumes fit in GPU memory.
  Output is identical to a global run to ~3e-7 (the estimator is
  separable — patches never interact).
- `foj3d/ct_pipeline.py`, `foj3d/projector.py`, `foj3d/strict_protocol.py`
  — CT forward model: TIGRE cone-beam geometry (R²-Gaussian scanner
  parameters), Poisson projection noise (P50 = 50 photons/pixel),
  CGLS-60 reconstruction, then FoJ.
- `foj3d/exp_*.py` — one driver per experiment (below).
- `foj3d/make_figs.py` — regenerates report figures from `data/*.json`.

## Datasets

No volumes are committed. All sources and the ground-truth files each
script expects are listed in **`data/DATASETS.md`** (Bentheim Zenodo,
BrainWeb, Digital Rocks Portal, R²-Gaussian CT volumes on Google Drive).

## How to run

Requirements: Python ≥3.10, `torch`, `numpy`, `scikit-image`,
`matplotlib`. GPU strongly recommended (runs take 2–20 min per FoJ call
on T4-class hardware; the CT pipeline additionally needs TIGRE + CUDA).
Most drivers accept `--smoke` for a quick CPU-scale test.

```bash
# 1. Bentheim micro-CT denoising sweep (main result)
#    auto-downloads the 1.2 GB Bentheim zip on first run;
#    appends metrics to data/rock2_all.json, saves FoJ outputs as .npy
python foj3d/exp_rock2.py            # or: python foj3d/exp_rock2.py --smoke

# 2. Binary porous-media volumes (sphere pack + Berea) — the collapse test
python foj3d/exp_rock.py

# 3. BrainWeb MRI under Rician noise — the smooth-texture control
python foj3d/exp_mri.py

# 4. CT ablation reproduction (engine, P50, R=4..12) — needs TIGRE and the
#    engine volume at data/engine/vol_gt.npy
python foj3d/exp_recon256.py

# 5. Regenerate all report figures from data/*.json
python foj3d/make_figs.py
```

Every driver writes its metrics into a JSON under `data/` as it runs, so
a preempted run resumes where it left off. Note the scripts were
developed against a Kaggle/T4 environment; absolute paths such as
`/kaggle/working/...` fall back to the script's own directory when run
locally.
