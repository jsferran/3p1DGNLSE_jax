# GNLSE Solver — Technical Notes for Collaborators

`gnlse_solver_noisy.py`: a GPU-accelerated, **differentiable**, stochastic
(3+1)D Generalized Nonlinear Schrödinger solver in JAX. One code path covers
1+1D (temporal), 2+1D (CW transverse), and 3+1D (full x,y,t), with and without
noise, forward-only or with exact end-to-end gradients.

---

## 1. What is integrated

Envelope A(x, y, t; z), split-step Fourier in z (Strang, 2nd order). One z-step:

```
1. exp(i D dz/2)          spectral (kx, ky, Ω): diffraction + dispersion
2. PML · Nprop            real space (x, y): absorber + waveguide potential
3. nonlinear step         (x, y, t): Kerr (exact exp) + Raman/gain (Heun RK2)
                          + self-steepening shock term
4. PML · Nprop            (mirror of 2)
5. exp(i D dz/2)          (mirror of 1)
6. [loss, noise, temporal absorber]   — stochastic/boundary bookkeeping
```

The state between steps lives in **k-space** (`field_kwo`, indices kx, ky, Ω);
each step does the FFT round trips internally.

Key operator details:

- **Diffraction is non-paraxial**: `D = i(sqrt(β_eff² − kx² − ky²) − β0 − β1 Ω + ½β2 Ω²)`.
  The sqrt term carries full angular-spectrum propagation and any material
  dispersion in n(ω); β0, β1 subtract the carrier phase and group delay
  (co-moving frame); β2 is an *additional* explicit GVD knob.
- **Waveguide**: transverse index profile n(x, y) enters as the exact phase
  `exp(i β_eff/2 ((n/n_eff)² − 1) dz)` per half-step (`Nprop_half`). This is how
  the GRIN guide is implemented — no mode approximation, full field.
- **Kerr** is an exact exponential `exp(i γ(1−f_R)|A|² dz)` (no step error in the
  SPM phase); **Raman** (Blow–Wood h(t), f_R fraction) and **saturable gain**
  integrate via Heun substeps; **self-steepening** applies the shock factor
  (Ω/ω0) to the full nonlinear polarization in ω-space.
- A **band-limit mask** (Matsushima) zeroes k-components that would alias for
  the given dz; a no-op for our fine steps.

## 2. Units and conventions

- **|A|² is intensity [W/m²]**, not power. γ = n₂ω₀/c₀ [m/W] is the *bulk*
  coefficient. For 1+1D temporal runs you must convert power → intensity with an
  effective area (P = I·A_eff). Powers quoted in the GRIN work are grid sums
  `Σ|A|² dx dy`.
- **β2 sign: standard (Agrawal) convention** — β2 < 0 anomalous (bright
  solitons with n₂>0), β2 > 0 normal. *Historical note*: releases before
  2026-07 had this flipped and old scripts carried `beta2_solver = -beta2_phys`
  workarounds; all removed. Any comment claiming "solver convention is
  opposite" is stale.
- **Frame group delay must match the medium.** With a non-dispersive n(x,y),
  the group index *is* the phase index: pass `beta1 = n_core/c0` with the exact
  `c0 = 299792458.0` (the module exports `C0`). A mismatched β1 (e.g. a
  literature group index, or a rounded c0) makes the pulse walk across the
  periodic time window and wrap — it looks like an instability but is a frame
  error.
- FFTs are unnormalized numpy-convention; fields reconstruct as
  `A(t) = Σ Ã e^{+iΩt}`, so ∂/∂t ↔ +iΩ. All sign choices above are stated in
  that convention.
- The module enables **JAX x64 at import** (fp64 default precision;
  `precision='fp32'` for speed). Requesting fp64 with x64 disabled raises
  instead of silently truncating.

## 3. Noise model (reparameterized — the differentiability trick)

Per z-step, *after* the deterministic step (Itô order — damping before noise):

```
A *= exp(-loss_coeff · dz)                        # optional FDT damping
A += σ · sqrt(dz) · H(ω) · ε_step                  # additive (ASE), or
A += σ · sqrt(dz) · sqrt(|A|) · ε_step             # shot noise (multiplicative)
```

- ε ~ CN(0,1) is **pre-sampled** (or derived from a per-window PRNG key); σ
  only scales it. Because ε is fixed, everything is a deterministic
  differentiable function of (A0, σ) — gradients through noisy trajectories
  are exact ("reparameterization trick").
- `loss_coeff = σ²/2` gives a stationary fluctuation–dissipation balance
  (additive white noise only; the multiplicative/shot case has no such closed
  form).
- Coloured noise: spectral filter H with mean|H|² = 1, so σ means the same
  total power for any filter shape.
- Shot noise uses `sqrt(max(|A|,1e-30))` — the floor keeps the backward pass
  finite in dark regions.
- Noiseless runs pass `eps=None`; a static `use_noise=False` branch means the
  ε array is **never materialized** (this used to allocate multi-GB zero
  tensors on fine grids).

## 4. The checkpointing method (how gradients through 15k steps fit in memory)

The problem: reverse-mode AD through N split-steps naively stores every
intermediate of every step (≈10 FFT-sized buffers/step). At N ≈ 15,000 that is
hopeless. Two nested ideas fix it:

**Level 1 — rematerialization inside a window (`jax.checkpoint` on the scan
body).** The propagation is a `lax.scan` whose carry is just the field. Wrapping
the step body in `jax.checkpoint` tells JAX: during backward, store only the
*carries* (one field per step) and **recompute** the step's internals (FFTs,
Kerr products) on the fly. Memory: steps × field. Compute: ~2× forward.

**Level 2 — windowed two-level checkpointing (`windowed_grad_noisy`).** Even
one-field-per-step is too much for long runs (3+1D: 15k × 134 MB ≈ 2 TB). So
split the N steps into W windows (~50–400 steps each):

```
FORWARD:   A0 → A1 → A2 → ... → A_W          run window-by-window;
           store ONLY the W+1 boundary fields, offloaded to HOST RAM (numpy).

LOSS:      value_and_grad of loss(A_W) → cotangent dL/dA_W
           (loss_domain='xyt' folds the k-space→real-space transform into it)

BACKWARD:  for w = W-1 ... 0:
               reload checkpoint A_w to GPU
               re-run window w forward under jax.vjp   (level-1 remat inside)
               dL/dA_w = VJP_w(dL/dA_{w+1})            (chain the cotangent)
```

Costs: **GPU memory = one window** (steps_per_window × field + working set);
host RAM = W × field; compute ≈ 3× a plain forward (forward + re-forward +
backward). The result is the *exact* adjoint — VJPs compose exactly, and the
per-window noise slices ε_w are reused bit-identically in the recomputation.
Verified against central finite differences: rel. err. ~10⁻⁴ (CW, fp32,
84 nonlinear steps at 2.5× P_cr) and 4×10⁻⁵ (3+1D with GVD + Raman +
self-steepening + temporal absorber).

Practical notes:
- `make_windowed_context_noisy(args, Nt)` pre-builds the JIT'd window
  propagator; **always pass `ctx=` inside an optimization loop** or every
  iteration recompiles.
- Gradient is returned w.r.t. the k-space initial field `A0_kwo`; chain to
  your actual parameters (e.g. mode coefficients) with one extra `jax.vjp` of
  `params → field → fftn`. Using *real* parameter vectors sidesteps JAX's
  complex-conjugation conventions entirely.
- The forward-only siblings: `GNLSE3D_propagate_noisy` (snapshots along z),
  `windowed_forward_noisy` (host checkpoints, final field). A separate
  scan-based propagator (`GNLSE3D_propagate`) supports event-triggered early
  stopping with its own 'segments'/'tree' remat strategies.

## 5. The gradients are exact — and can still lie (chaos horizon)

Measured on the GRIN system at 2.5× P_cr: at 84 steps the adjoint gradient
matches finite differences to 10⁻⁴; at ~2100 steps the *local* adjoint slope is
~10³× the finite-scale slope. This is not a bug: linearized perturbations grow
like e^{λz} through strongly nonlinear propagation (λ ≈ 0.004/step here), so
the local derivative explodes and decorrelates from the landscape at any
usable step size — the classic chaotic-adjoint problem. Consequences for any
optimization built on this solver:

- Below the horizon (≲1500–2500 steps): quasi-Newton on exact gradients
  (L-BFGS) is excellent.
- Beyond it: use finite-scale estimators (ES/SPSA probes) and/or guarded
  (acceptance-tested) steps; and prefer **continuation** (extend solutions
  from shorter lengths) — note also that gradients *through a completed
  collapse* vanish (power-based losses saturate), so you cannot gradient-ascend
  out of a collapsed configuration.
- The loss landscape at long z has chaotic micro-texture (~10⁻³ in J at 30 cm);
  acceptance tolerances must exceed it.

## 6. Boundaries (three independent systems)

- **Spatial PML/CAP** (x,y edges): sin² real absorber, `exp(−W dz/2)` per
  half-step. Wmax auto-calibrated from the PML *depth* (40 dB across d_pml),
  not the propagation length. `pml_eta = 0` is mandatory (a complex CAP phase
  diffracts high-k ghosts). Verified: a guided LP01 loses only 0.84% over
  35 cm / 15k steps (uniform discretization shedding, not PML erosion).
- **Temporal absorber** (t edges): sin² sponge `exp(−α(t) dz)`, for pulses that
  broaden toward the window edge; honoured on **all** propagator paths.
- **Band-limit mask**: automatic anti-aliasing of the transfer function.

## 7. Performance & parallelism

- Everything is `lax.scan` under `jit`; per-step cost ≈ a few FFTs of
  (Nx·Ny·Nt). Multi-GPU shards the **time axis** (NamedSharding); on one GPU it
  degenerates to a no-op. CW 256²: ~10³ steps/s on a desktop GPU.
- fp32 is the workhorse (search/optimization); fp64 for final verification.
  fp32 J has ~10⁻⁶ roundoff texture; gradients through ~10³ steps stay usable.
- Memory knobs: `n_windows` (GPU peak), host RAM = checkpoints, and the
  static `use_noise` flag (noiseless runs allocate no ε).

## 8. Discretization guidance (measured on the 50 µm GRIN @ 2.5× P_cr)

- Transverse 256² (dx = 0.625 µm): converged at short z (0.00% vs 512²) but
  **−7.8% in core-delivered power at 30 cm** — fine grids resolve *more* of a
  metastable state's slow radiative bleed. Validate long-z quantitative claims
  at 2× resolution (`finecheck` tooling in `pns_grin_sfopt.py`).
- z-step L_beat/24 (≈24 µm): **+4.5% at dz/2** even at 0.5 cm (splitting error
  in the strongly-bleeding transient). Production: L_beat/48.
- The two errors have opposite signs; quote 256²/24 curves with ~±5–8%
  discretization uncertainty, or fine-check them.

## 9. Validation suite (run these before trusting a modified solver)

- `test_solver_fixes.py` — 17 regression checks: temporal absorber on all
  paths, (1−f_R) Kerr weight, β2 soliton-sign convention (N=1 sech preserved at
  β2<0, broadens at β2>0), loss-domain consistency, boundary guards, CW edge
  cases.
- `test_selfsteepening.py` — solver vs independent RK4 integration of the pure
  Kerr+shock PDE: rel. L2 error 2×10⁻⁸.
- `python pns_grin_sfopt.py audit [--nt 64]` — end-to-end FD gradient checks
  (CW + 3+1D), 35 cm boundary-erosion floor, LP01/Huang collapse cross-check,
  optimizer monotonicity from a collapsed init.
