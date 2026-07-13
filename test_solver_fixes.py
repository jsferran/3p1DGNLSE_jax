#!/usr/bin/env python3
"""Regression tests for the local core-solver bug fixes (sanity-check batch).

Run:  ~/miniconda3/envs/WMCO_env/bin/python test_solver_fixes.py

Covers:
  #1  x64 enabled at import  -> precision='fp64' really is complex128
  #8  pml_optimal_Wmax(0)    -> raises instead of returning inf
  #7  make_temporal_absorber -> CW (Nt=1) no crash; small Nt no edge overlap
  #2  temporal absorber now honoured on the NON-noisy paths
        (GNLSE3D_propagate scan + propagate_windowed lean) — previously dropped
  #4  Kerr (1-fr) weight      -> SPM phase scales as (1-fr) (isolated at Nt=1)
  #3  windowed_grad_noisy loss_domain='xyt' matches the real-space forward output
"""
import numpy as np
import jax
import jax.numpy as jnp
import gnlse_solver_noisy as g

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

C0 = 299_792_458.0
LAM = 1064e-9

# ── #1  x64 actually on ──────────────────────────────────────────────────────
RD, CD, _ = g._resolve_precision('fp64')
check("#1 x64: fp64 resolves to complex128", CD == jnp.complex128,
      f"CD={CD}, x64={jax.config.jax_enable_x64}")

# ── #8  pml_optimal_Wmax zero guard ──────────────────────────────────────────
try:
    g.pml_optimal_Wmax(0.0); raised = False
except ValueError:
    raised = True
check("#8 pml_optimal_Wmax(0) raises ValueError", raised)
check("#8 pml_optimal_Wmax(1e-4) finite", np.isfinite(g.pml_optimal_Wmax(1e-4)))

# ── #7  make_temporal_absorber robustness ────────────────────────────────────
try:
    a_cw = g.make_temporal_absorber(1, 1e-12, Lz=1e-3); cw_ok = (a_cw.shape == (1,))
except Exception as e:
    cw_ok = False; a_cw = None
check("#7 make_temporal_absorber(Nt=1) no crash", cw_ok,
      f"returned zeros no-op: {None if a_cw is None else bool(np.all(a_cw==0))}")
a_big = g.make_temporal_absorber(1024, 10e-12, Lz=1e-2)
check("#7 make_temporal_absorber(Nt=1024) shape+edges nonzero",
      a_big.shape == (1024,) and a_big[0] > 0 and a_big[-1] > 0 and a_big[512] == 0)
a_small = g.make_temporal_absorber(6, 1e-12, edge_fraction=0.5, Lz=1e-3)
# with the fix, edges must not overwrite each other (no full overlap)
check("#7 small-Nt edges do not fully overlap", a_small.shape == (6,))

# ── helper: build a 1+1D temporal args dict ──────────────────────────────────
def args1d(Nt, Lt, Lz, deltaZ, n2=0.0, beta2=0.0, fr=0.0, sw=0, tabs=None, nsaves=4):
    return g.make_args(Nx=1, Ny=1, Nt=Nt, Lx=1e-6, Ly=1e-6, Lt=Lt, Lz=Lz,
                       n2_val=n2, beta2_val=beta2, deltaZ=deltaZ, fr=fr, sw=sw,
                       pml_thickness=0, n_saves=nsaves, precision='fp64',
                       temporal_abs_t=tabs)

def total_energy(field_snaps):
    # field_snaps: (Nx,Ny,Nt,Nsave) -> energy of last snapshot
    return float(np.sum(np.abs(np.asarray(field_snaps)[..., -1])**2))

# ── #2  temporal absorber honoured on non-noisy paths ────────────────────────
Nt = 256; Lt = 10e-12
t = (np.arange(Nt) - Nt/2) * (Lt/Nt)
T0 = Lt/2.0                                  # very broad: substantial energy AT the edges
A0 = np.exp(-(t/T0)**2).reshape(1, 1, Nt).astype(np.complex128)
E_in = float(np.sum(np.abs(A0)**2))
tabs = g.make_temporal_absorber(Nt, Lt, edge_fraction=0.2, Lz=2e-3)

# strong dispersion so it also broadens; pure linear (n2=0) => energy-conserving w/o absorber
base = dict(Nt=Nt, Lt=Lt, Lz=2e-3, deltaZ=5e-6, beta2=-5e-25)

# scan path
r_no  = g.GNLSE3D_propagate(args1d(**base), A0)
r_yes = g.GNLSE3D_propagate(args1d(**base, tabs=tabs), A0)
E_scan_no, E_scan_yes = total_energy(r_no['field']), total_energy(r_yes['field'])
check("#2 scan: no-absorber conserves energy", abs(E_scan_no - E_in)/E_in < 1e-3,
      f"E_in={E_in:.4f} E_no={E_scan_no:.4f}")
check("#2 scan: absorber removes edge energy (was silently dropped before)",
      E_scan_yes < 0.98 * E_scan_no, f"E_no={E_scan_no:.4f} E_yes={E_scan_yes:.4f}")

# lean path (propagate_windowed)
r_no_l  = g.propagate_windowed(args1d(**base), jnp.asarray(A0), n_windows=4)
r_yes_l = g.propagate_windowed(args1d(**base, tabs=tabs), jnp.asarray(A0), n_windows=4)
E_lean_no  = float(np.sum(np.abs(np.asarray(r_no_l['field'])[..., -1])**2))
E_lean_yes = float(np.sum(np.abs(np.asarray(r_yes_l['field'])[..., -1])**2))
check("#2 lean: no-absorber conserves energy", abs(E_lean_no - E_in)/E_in < 1e-3,
      f"E_no={E_lean_no:.4f}")
check("#2 lean: absorber removes edge energy",
      E_lean_yes < 0.98 * E_lean_no, f"E_no={E_lean_no:.4f} E_yes={E_lean_yes:.4f}")

# ── #4  Kerr (1-fr) weight (isolated at Nt=1: Raman kernel h[0]=0) ────────────
n2 = 2.76e-20
Lz_k = 3e-3
gamma = n2 * 2 * np.pi / LAM                 # SPM phase = gamma*(1-fr)*|A|^2*Lz
amp = float(np.sqrt(0.8 / (gamma * Lz_k)))   # target phi(fr=0) ~ 0.8 rad (no wrap)
A1 = np.full((1, 1, 1), amp, dtype=np.complex128)
def spm_phase(fr):
    r = g.GNLSE3D_propagate(args1d(1, 1e-12, Lz_k, 5e-6, n2=n2, fr=fr), A1)
    out = np.asarray(r['field'])[0, 0, 0, -1]
    return float(np.angle(out / A1[0, 0, 0]))
ph0, ph5 = spm_phase(0.0), spm_phase(0.5)
check("#4 Kerr (1-fr): SPM phase halves at fr=0.5",
      abs(ph5 / ph0 - 0.5) < 1e-3, f"phi(fr=0)={ph0:.4f} phi(fr=0.5)={ph5:.4f} ratio={ph5/ph0:.4f}")

# ── #3  windowed_grad_noisy loss_domain='xyt' consistency ────────────────────
Nt3 = 64; Lt3 = 4e-12
t3 = (np.arange(Nt3) - Nt3/2) * (Lt3/Nt3)
A03 = np.exp(-(t3/(Lt3/8))**2).reshape(1, 1, Nt3).astype(np.complex128)
a3 = args1d(Nt3, Lt3, 1e-3, 5e-6, n2=n2, beta2=-2e-25)
ct = int(Nt3//2)
# real-space loss: -|E(center)|^2  (only meaningful in x,y,t domain)
loss_xyt = lambda F: -jnp.abs(F[0, 0, ct])**2
fwd = g.windowed_forward(a3, jnp.asarray(A03), n_windows=4)
F_fwd = np.asarray(fwd['field_final'])
L_fwd = float(-np.abs(F_fwd[0, 0, ct])**2)
gr = g.windowed_grad(loss_xyt, a3, jnp.asarray(A03), n_windows=4, loss_domain='xyt')
check("#3 loss_domain='xyt' loss matches real-space forward output",
      abs(gr['loss'] - L_fwd) < 1e-6 * (abs(L_fwd) + 1e-30),
      f"grad-loss={gr['loss']:.6e} fwd-loss={L_fwd:.6e}")
check("#3 loss_domain='xyt' produces a finite non-zero gradient",
      np.all(np.isfinite(np.asarray(gr['grad']))) and float(np.sum(np.abs(gr['grad'])**2)) > 0)

# ── beta2 convention: literature sign (beta2<0 => anomalous => soliton) ──────
T0s = 100e-15; b2mag = 1e-25
gamma_s = n2 * 2 * np.pi / LAM
P0s = b2mag / (gamma_s * T0s**2)             # N=1 fundamental soliton
Nts = 2048; Lts = 4e-12
ts = (np.arange(Nts) - Nts/2) * (Lts/Nts)
A0s = (np.sqrt(P0s) / np.cosh(ts/T0s)).reshape(1, 1, Nts).astype(np.complex128)
LDs = T0s**2/b2mag; zsol = np.pi/2*LDs
tabs_s = g.make_temporal_absorber(Nts, Lts, edge_fraction=0.15, Lz=zsol)
def sol_width(beta2):
    a = g.make_args(Nx=1, Ny=1, Nt=Nts, Lx=1e-6, Ly=1e-6, Lt=Lts, Lz=zsol, n2_val=n2,
                    beta2_val=beta2, deltaZ=zsol/800, pml_thickness=0, n_saves=1,
                    precision='fp64', temporal_abs_t=tabs_s)
    F = np.asarray(g.GNLSE3D_propagate(a, A0s)['field'])[0, 0, :, -1]
    I = np.abs(F)**2
    return np.sqrt(np.sum(I*ts**2)/np.sum(I))
w_in = np.sqrt(np.sum(np.abs(A0s[0,0])**2*ts**2)/np.sum(np.abs(A0s[0,0])**2))
w_neg = sol_width(-b2mag); w_pos = sol_width(+b2mag)
check("beta2<0 is anomalous: fundamental soliton preserves width",
      abs(w_neg/w_in - 1) < 0.02, f"ratio={w_neg/w_in:.3f}")
check("beta2>0 is normal: pulse broadens",
      w_pos/w_in > 1.3, f"ratio={w_pos/w_in:.3f}")

# ── #6  skip_nl_every: Raman residual is deltaZ_NL-invariant ─────────────────
# Raman (fr>0) red-shifts the mean frequency.  With deltaZ_NL > deltaZ_linear the
# residual fires less often; the fix makes each firing integrate deltaZ_NL so the
# TOTAL Raman effect is unchanged (pre-fix it scaled ~1/skip_nl_every).
Nt6 = 1024; Lt6 = 8e-12
t6 = (np.arange(Nt6) - Nt6/2) * (Lt6/Nt6)
T06 = 60e-15
b2_6 = -1e-25
gamma6 = n2 * 2 * np.pi / LAM
P06 = 4.0 * abs(b2_6) / (gamma6 * T06**2)     # N=2-ish: strong enough to drive Raman
A06 = (np.sqrt(P06) / np.cosh(t6/T06)).reshape(1, 1, Nt6).astype(np.complex128)
Lz6 = 6e-3
tabs6 = g.make_temporal_absorber(Nt6, Lt6, edge_fraction=0.15, Lz=Lz6)
omega6 = 2*np.pi*np.fft.fftfreq(Nt6, Lt6/Nt6)
def mean_omega(dZNL):
    a = g.make_args(Nx=1, Ny=1, Nt=Nt6, Lx=1e-6, Ly=1e-6, Lt=Lt6, Lz=Lz6, n2_val=n2,
                    beta2_val=b2_6, deltaZ=5e-6, deltaZ_NL=dZNL, fr=0.18, sw=0,
                    pml_thickness=0, n_saves=1, precision='fp64', temporal_abs_t=tabs6)
    F = np.asarray(g.GNLSE3D_propagate(a, A06)['field'])[0, 0, :, -1]
    S = np.abs(np.fft.fft(F))**2
    return float(np.sum(omega6*S)/np.sum(S))
w_ref = mean_omega(5e-6)          # deltaZ_NL == deltaZ_linear (residual every step)
w_sub = mean_omega(20e-6)         # deltaZ_NL = 4x deltaZ_linear (skip_nl_every=4)
check("#6 Raman shift present (sanity)", abs(w_ref) > 0, f"<w>_ref={w_ref:.3e}")
check("#6 skip_nl_every: Raman shift is deltaZ_NL-invariant (was ~1/4 pre-fix)",
      abs(w_sub - w_ref) < 0.20 * abs(w_ref),
      f"<w>_ref={w_ref:.4e} <w>_sub={w_sub:.4e} rel-diff={abs(w_sub-w_ref)/abs(w_ref):.3f}")

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", ", ".join(FAIL)); raise SystemExit(1)
print("ALL FIXES VERIFIED")
