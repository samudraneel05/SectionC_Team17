# Separability of the released 3D FoJ implementation

Technical notes supporting Section 3 of `report/report.tex`. The paper
states the findings; this file records the code-level evidence for them.
All line numbers refer to `field_of_junctions3d.py` in
<https://github.com/voilalab/3D-Field-of-Junctions>.

## Claim

At the default configuration (`N_init = N_ref = 1`), the 3D FoJ
objective reduces to a sum of independent per-patch problems. Each patch
is fitted on its own and overlapping patch renderings are averaged.
Nothing couples the patches.

## 1. Consistency weights are zero at defaults (lines 121–128)

```python
if self.opts.num_refinement_iters <= 1:
    factor = 0.0
else:
    factor = max(0, (iteration - self.opts.num_initialization_iters) /
                    (self.opts.num_refinement_iters - 1))
lmbda_boundary = factor * self.opts.lambda_boundary_final
lmbda_color    = factor * self.opts.lambda_color_final
```

`factor` is hard-coded to `0.0` whenever `num_refinement_iters <= 1`.
At the default `N_ref = 1`, the ramp never leaves zero, so
`λ_B = λ_C = 0` at every iteration and the boundary/color consistency
terms contribute no gradient.

## 2. The single refinement step cannot change the output (lines 208–227)

```python
params = torch.cat([self.angles, self.x0y0z0], dim=1)
dists, colors, patches = self.get_dists_and_patches_3d(params, lmbda_color)
loss = self.get_loss_3d(dists, colors, patches, lmbda_boundary, lmbda_color).mean()
...
optimizer.step()
dists, colors, patches = self.get_dists_and_patches_3d(params, lmbda_color)
self.global_volume = self.local2global_3d(patches)
```

`torch.cat` allocates a new tensor. `optimizer.step()` updates the leaf
tensors `self.angles` and `self.x0y0z0` in place — not the `params`
copy. The re-render after `step()` therefore uses the pre-update values,
and `global_volume` is recomputed from exactly the parameters that
initialization produced. The update lands on the leaf tensors but would
only be read by a *second* refinement iteration, which does not run.

## Consequence

The estimator is separable: each patch solves

    theta_p* = argmin_theta || y_hat_p(theta) - y|_{Omega_p} ||_2^2

independently, and the output volume is the average of overlapping patch
renderings. It is therefore partition-invariant: any split of the patch
set into batches or spatial chunks gives the same output up to
floating-point tie-breaking in the discrete `argmin`.

## Verification

- Chunked execution (halo-padded chunks, renderings accumulated
  globally) reproduces the global implementation to ~3e-7 RMSE on a
  CGLS reconstruction crop for R ∈ {6, 8, 12}; deviations at
  R ∈ {4, 10} are near-ties in the discrete search.
- Toggling the refinement step changes the output by at most 3.6e-7.
- All full-volume results in the report are within 0.01 dB of an exact
  global run.

## Consistent evidence from the paper itself

- The batch-size ablation reports identical results for batch 1–6 —
  expected if patches never interact.
- The refinement ablation (appendix) finds that more iterations, which
  activate the consistency ramp, degrade quality.

## Why it matters

A separable wedge fit followed by averaging cannot:
1. remove a systematic input bias (region intensities are local means);
2. represent structure that is not piecewise-flat at the patch scale;
3. help when noise is not the dominant error (the wedge model's
   approximation bias must be paid for by the noise removed).

The experiments in the report test each condition in turn.
