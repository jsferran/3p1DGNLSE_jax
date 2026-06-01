#!/usr/bin/env python3
"""
Temporal-axis resolution convergence test for the 3+1D GRIN fiber forward pass.

Runs noiseless forward propagation at Nt in {128, 256, 512, 1024, 2048} with
identical physical parameters. Saves output metrics and a summary figure to
compare simulation quality as a function of time-axis resolution.

Usage (interactive GPU node or via grin3d_nt_convergence_submit.sh):
    python grin3d_nt_convergence.py --code-dir . --mode-folder grin_modes_300_50um
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',    type=str, default=str(Path(__file__).resolve().parent))
ap.add_argument('--mode-folder', type=str, default='grin_modes_300_50um')
ap.add_argument('--n-modes',     type=int, default=100)
ap.add_argument('--n-windows',   type=int, default=220)
ap.add_argument('--dz-frac',     type=float, default=0.05)
ap.add_argument('--lz-cm',       type=float, default=30.0)
ap.add_argument('--p-mult',      type=float, default=2.0)
ap.add_argument('--outdir',      type=str, default='grin3d_nt_convergence')
cli = ap.parse_args()

CODE_DIR  = Path(cli.code_dir)
MODE_PATH = (Path(cli.mode_folder) if Path(cli.mode_folder).is_absolute()
             else CODE_DIR / cli.mode_folder)
OUTDIR = Path(cli.outdir)
OUTDIR.mkdir(exist_ok=True)

sys.path.insert(0, str(CODE_DIR))
try:
    from gnlse_solver_noisy_new import (
        make_args, make_temporal_absorber, pml_optimal_Wmax,
        make_windowed_context_noisy,
    )
except ImportError:
    from gnlse_solver_noisy import (
        make_args, make_temporal_absorber, pml_optimal_Wmax,
        make_windowed_context_noisy,
    )

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)
print(f"JAX devices: {jax.devices()}", flush=True)

# ── Fixed physics (matches grin3d_cluster_run.py exactly) ─────────────────────
C0      = 2.99792458e8
LAMBDA0 = 1030e-9
N_CORE  = 1.453
NA      = 0.20
R_CORE  = 50e-6
N2      = 2.76e-20
BETA2   = 22e-27
T_FWHM  = 100e-15
FR, SW  = 0.18, 1

NX = NY = 300
LX = LY = 350e-6
LT      = 3e-12
PML_PIX = 30
T_ABS_FRAC = 0.10

n_clad = float(np.sqrt(N_CORE**2 - NA**2))
Delta  = (N_CORE**2 - n_clad**2) / (2.0 * N_CORE**2)
omega0 = 2.0 * np.pi * C0 / LAMBDA0
k0     = 2.0 * np.pi / LAMBDA0
w0     = float(np.sqrt(R_CORE / (k0 * N_CORE * np.sqrt(2.0 * Delta))))
L_beat = np.pi * R_CORE / np.sqrt(2.0 * Delta)
gamma  = N2 * omega0 / C0
P_cr   = np.pi * w0**2 / (2.0 * gamma * L_beat)
P_peak = cli.p_mult * P_cr
T0     = T_FWHM / (2.0 * np.sqrt(np.log(2.0)))

N_BEAT      = int(np.ceil(cli.lz_cm * 1e-2 / L_beat))
Lz          = N_BEAT * L_beat
deltaZ      = cli.dz_frac * L_beat
steps_total = int(np.ceil(Lz / deltaZ))
E_PULSE     = P_peak * np.sqrt(np.pi) * T0

print(f"steps={steps_total}  Lz={Lz*100:.1f}cm  L_beat={L_beat*1e3:.3f}mm", flush=True)
print(f"P_peak={P_peak:.2e}W  E_pulse={E_PULSE*1e9:.2f}nJ", flush=True)

# ── Fixed spatial grid ────────────────────────────────────────────────────────
from gnlse_medium import make_polynomial_n
x  = np.linspace(-LX/2, LX/2, NX, endpoint=False)
y  = np.linspace(-LY/2, LY/2, NY, endpoint=False)
dx = float(x[1] - x[0])
X, Y = np.meshgrid(x, y, indexing='ij')
n_xy = make_polynomial_n(X, Y, N_CORE, n_clad, R_CORE, alpha=2).astype(np.float32)

d_pml    = PML_PIX * dx
pml_Wmax = pml_optimal_Wmax(d_pml, target_db=40.0)

# ── Load modes (fixed coefficients across all Nt) ────────────────────────────
N_MODES = cli.n_modes
B = np.zeros((NX * NY, N_MODES), dtype=np.complex64)
for m in range(N_MODES):
    fpath = MODE_PATH / f"mode_{m:04d}.npz"
    if not fpath.exists():
        N_MODES = m; B = B[:, :m]; break
    data = np.load(str(fpath))
    B[:, m] = data['field'].ravel().astype(np.complex64)
print(f"Loaded {N_MODES} modes", flush=True)

rng = np.random.default_rng(42)
c_complex = (rng.standard_normal(N_MODES) + 1j * rng.standard_normal(N_MODES)).astype(np.complex64)
c_complex /= np.linalg.norm(c_complex)
f_spatial = (B @ c_complex).reshape(NX, NY)  # (NX, NY)

def _window_steps(total, nw):
    base, rem = divmod(total, nw)
    return [base + (1 if i < rem else 0) for i in range(nw)]

# ── NT sweep ──────────────────────────────────────────────────────────────────
NT_LIST = [128, 256, 512, 1024, 2048]
results = []

for NT in NT_LIST:
    dt = LT / NT
    f_nyq_thz = 1.0 / (2.0 * dt) * 1e-12
    print(f"\n{'='*60}", flush=True)
    print(f"NT={NT}  dt={dt*1e15:.1f}fs  f_Nyq={f_nyq_thz:.1f}THz  "
          f"T0/dt={T0/dt:.1f}", flush=True)

    t = np.linspace(-LT/2, LT/2, NT, endpoint=False)

    TEMPORAL = np.exp(-t**2 / (2.0 * T0**2)).astype(np.float32)
    NORM_T2  = float(np.sum(TEMPORAL**2) * dt)
    scale    = np.sqrt(E_PULSE / (float(np.sum(np.abs(f_spatial)**2) * dx**2) * NORM_T2))
    A0 = (f_spatial * scale)[:, :, None] * TEMPORAL[None, None, :]

    temp_abs = make_temporal_absorber(NT, LT, edge_fraction=T_ABS_FRAC, Lz=Lz)
    args = make_args(
        Nx=NX, Ny=NY, Nt=NT,
        Lx=LX, Ly=LY, Lt=LT, Lz=Lz,
        n_xyomega=n_xy[:, :, None],
        n_ref=N_CORE, n2_val=N2, lambda0=LAMBDA0,
        beta0_val=N_CORE * k0,
        beta1_val=N_CORE / C0,
        beta2_val=BETA2,
        deltaZ=deltaZ, fr=FR, sw=SW,
        pml_thickness=PML_PIX, pml_Wmax=pml_Wmax,
        n_saves=1,
        precision='fp32',
        temporal_abs_t=temp_abs,
    )

    print("  Compiling ...", flush=True)
    t0 = time.time()
    ctx = make_windowed_context_noisy(args, NT, use_shot_noise=False)
    jnp.zeros(1).block_until_ready()
    print(f"  Compiled in {time.time()-t0:.1f}s", flush=True)

    prop_lean   = ctx['prop_lean']
    lean_kw     = ctx['lean_kw']
    ws          = _window_steps(steps_total, cli.n_windows)
    filter_flat = jnp.ones(NT, dtype=jnp.float32)
    CD          = jnp.complex64

    def _forward(A0f, _ws=ws, _prop=prop_lean, _kw=lean_kw, _ff=filter_flat):
        field = jnp.fft.fftn(jnp.asarray(A0f, dtype=CD), axes=(0, 1, 2))
        for nw_here in _ws:
            eps_t = jnp.zeros((nw_here, 1, 1, 1), dtype=CD)
            field = _prop(field, eps_t, 0.0, _ff,
                          use_noise_filter=False, **_kw, steps_total=nw_here)
        return jnp.fft.ifftn(field, axes=(0, 1, 2))

    A0_jax = jnp.asarray(A0, dtype=CD)
    # warmup
    _ = _forward(A0_jax); jax.block_until_ready(_)

    t0 = time.time()
    A_out = _forward(A0_jax)
    jax.block_until_ready(A_out)
    t_fwd = time.time() - t0
    print(f"  Forward pass: {t_fwd:.2f}s", flush=True)

    A_out_np = np.asarray(A_out)
    I_out    = np.abs(A_out_np)**2

    peak_power = float(I_out.max())
    energy_out = float(I_out.sum() * dx**2 * dt)
    energy_in  = float(np.abs(A0)**2.sum() * dx**2 * dt)

    ix, iy = NX // 2, NY // 2
    t_profile = I_out[ix, iy, :]
    t_peak_idx = int(np.argmax(t_profile))
    xy_profile = I_out[:, :, t_peak_idx]

    print(f"  peak_P={peak_power:.4e}W  E_out={energy_out*1e12:.3f}pJ  "
          f"E_in={energy_in*1e12:.3f}pJ", flush=True)

    results.append(dict(
        NT=NT, dt=dt, t=t, t_fwd=t_fwd,
        peak_power=peak_power, energy_out=energy_out, energy_in=energy_in,
        t_profile=t_profile, xy_profile=xy_profile,
    ))

    np.savez(OUTDIR / f"nt_{NT:04d}.npz",
             NT=NT, dt=dt, t=t,
             A_out_center_t=A_out_np[ix, iy, :],
             A_out_xy_peak=A_out_np[:, :, t_peak_idx],
             peak_power=peak_power, energy_out=energy_out, energy_in=energy_in,
             t_fwd=t_fwd)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*74, flush=True)
print(f"{'NT':>6}  {'dt(fs)':>8}  {'t_fwd(s)':>9}  "
      f"{'peak_P(W)':>12}  {'E_out(pJ)':>10}  {'E_in(pJ)':>10}", flush=True)
for r in results:
    print(f"{r['NT']:>6}  {r['dt']*1e15:>8.1f}  {r['t_fwd']:>9.1f}  "
          f"{r['peak_power']:>12.4e}  {r['energy_out']*1e12:>10.3f}  "
          f"{r['energy_in']*1e12:>10.3f}", flush=True)

ref = results[-1]
print(f"\nRelative to Nt={ref['NT']}:", flush=True)
for r in results[:-1]:
    dp = abs(r['peak_power'] - ref['peak_power']) / max(ref['peak_power'], 1e-30) * 100
    de = abs(r['energy_out'] - ref['energy_out']) / max(ref['energy_out'], 1e-30) * 100
    print(f"  Nt={r['NT']:4d}: delta_peak={dp:.2f}%  delta_energy={de:.2f}%", flush=True)

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))

ax = axes[0, 0]
ax.plot([r['NT'] for r in results], [r['peak_power'] for r in results], 'o-', color='steelblue')
ax.set_xscale('log', base=2)
ax.set_xlabel('Nt')
ax.set_ylabel('Peak output power (W)')
ax.set_title('Peak power vs Nt')
ax.set_xticks([r['NT'] for r in results])
ax.set_xticklabels([str(r['NT']) for r in results])
ax.axhline(ref['peak_power'], ls='--', color='gray', lw=0.8)

ax = axes[0, 1]
ax.plot([r['NT'] for r in results], [r['energy_out'] * 1e12 for r in results], 'o-', color='tomato')
ax.set_xscale('log', base=2)
ax.set_xlabel('Nt')
ax.set_ylabel('Output energy (pJ)')
ax.set_title('Output energy vs Nt')
ax.set_xticks([r['NT'] for r in results])
ax.set_xticklabels([str(r['NT']) for r in results])
ax.axhline(ref['energy_out'] * 1e12, ls='--', color='gray', lw=0.8)

ax = axes[1, 0]
for r, c in zip(results, colors):
    t_ps = r['t'] * 1e12
    ax.plot(t_ps, r['t_profile'] / max(ref['peak_power'], 1e-30),
            label=f"Nt={r['NT']}", color=c, lw=1.2, alpha=0.9)
ax.set_xlabel('t (ps)')
ax.set_ylabel('Intensity / ref peak (norm.)')
ax.set_title('Output temporal profile at beam center')
ax.legend(fontsize=8)
t_win = 5 * T0 * 1e12  # ±5 T0 around center for zoom
ax.set_xlim(-t_win, t_win)

ax = axes[1, 1]
ax.plot([r['NT'] for r in results], [r['t_fwd'] for r in results], 's-', color='seagreen')
ax.set_xscale('log', base=2)
ax.set_xlabel('Nt')
ax.set_ylabel('Forward pass time (s)')
ax.set_title('Wall time vs Nt')
ax.set_xticks([r['NT'] for r in results])
ax.set_xticklabels([str(r['NT']) for r in results])

fig.tight_layout()
fig_path = OUTDIR / 'nt_convergence.png'
fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {fig_path}", flush=True)

np.savez(OUTDIR / 'nt_convergence_summary.npz',
         NT_list=[r['NT'] for r in results],
         peak_power=[r['peak_power'] for r in results],
         energy_out=[r['energy_out'] for r in results],
         energy_in=[r['energy_in'] for r in results],
         t_fwd=[r['t_fwd'] for r in results])
print("Done.", flush=True)
