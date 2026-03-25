# ======================================================================================
# gnlse_solver_noisy.py
#
# Self-contained GNLSE 3+1D solver with optional per-step additive Gaussian noise.
#
# This module merges gnlse_solver_new.py (core physics) with stochastic extensions.
# It is the single public API — no other solver file needs to be imported.
#
# Noise model
# -----------
# At the end of each split-step (in the x,y,t domain):
#
#   A_xyt  +=  noise_sigma * sqrt(deltaZ) * noise_step(t)
#
# with optional spectral colour filter H(omega) and FDT loss balance.
# Pass eps=None and noise_sigma=0 to recover fully noiseless behaviour.
#
# Public API — noiseless
# ----------------------
#   GNLSE3D_propagate        — scan-based forward with snapshots + event stopping
#   GNLSE3D_propagate_lean   — lean-carry forward with snapshots
#   propagate_windowed        — forward with snapshots in visualisation format
#   make_windowed_context    — build reusable JIT context (noiseless)
#   windowed_forward         — forward pass broken into windows (noiseless)
#   windowed_grad            — dL/dA0 via per-window VJP (noiseless)
#   make_args                — build args dict (+ noise_sigma, noise_filter_w, temporal_abs_t)
#
# Public API — noisy / stochastic
# --------------------------------
#   make_noise_filter         — spectral colour filter factory
#   make_noise_samples        — draw pre-sampled noise array
#   GNLSE3D_propagate_noisy   — forward with per-step reparameterized noise
#   make_windowed_context_noisy — reusable JIT context with FDT loss
#   windowed_forward_noisy    — windowed forward with noise
#   windowed_grad_noisy       — windowed gradient with noise
# ======================================================================================

import time
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

C0 = 299_792_458.0
LN10_OVER_10 = np.log(10) / 10


# --------------------------------------------------------------------------------------
# Precision policy
# --------------------------------------------------------------------------------------
def _resolve_precision(precision: str | None):
    p = (precision or "fp64").lower()
    if p in ("fp32", "32", "single"):
        return jnp.float32, jnp.complex64, np.float32
    elif p in ("fp64", "64", "double"):
        return jnp.float64, jnp.complex128, np.float64
    else:
        raise ValueError("precision must be 'fp32' or 'fp64'")


# --------------------------------------------------------------------------------------
# Raman kernel (fft of causal response)
# --------------------------------------------------------------------------------------
def _make_hrw(Nt: int, dt: float, t1=12.2e-15, t2=32.0e-15, *, real_dtype=jnp.float64):
    complex_dtype = jnp.complex64 if real_dtype == jnp.float32 else jnp.complex128
    t = (dt * jnp.arange(Nt, dtype=real_dtype))
    h = ((t1**2 + t2**2) / (t1 * t2**2)) * jnp.exp(-t/real_dtype(t2)) * jnp.sin(t/real_dtype(t1))
    H = jnp.fft.fft(h).astype(complex_dtype) * real_dtype(dt)
    return H


# --------------------------------------------------------------------------------------
# Residual NL (Raman + self-steepening + gain/saturation)
# --------------------------------------------------------------------------------------
def _dA_dz_NL_rest(A_xy_t,
                   *, dt: float, f0: float, fr: float, sw: int, gamma: float,
                   omega_vec: jnp.ndarray, hrw: jnp.ndarray, gain_term: jnp.ndarray,
                   saturation_intensity: float, use_gain: bool) -> jnp.ndarray:
    CD = A_xy_t.dtype
    RD = jnp.float32 if CD == jnp.complex64 else jnp.float64
    ONEJ = jax.lax.complex(RD(0.0), RD(1.0))

    absA2 = jnp.abs(A_xy_t)**2
    NL = jnp.zeros_like(A_xy_t, dtype=CD)

    # Raman
    if fr != 0.0:
        Iw = jnp.fft.fft(absA2, axis=2).astype(CD)
        Hw = hrw[None, None, :].astype(CD)
        Raman = jnp.fft.ifft(Hw * Iw, axis=2).astype(CD)
        Raman = jnp.nan_to_num(Raman, nan=0.0, posinf=0.0, neginf=0.0).astype(CD)
        NL = NL + (fr * Raman).astype(CD) * A_xy_t

    # Self-steepening
    if sw == 1:
        NLw = jnp.fft.fft(NL, axis=2).astype(CD)
        NLw = NLw * (RD(1.0) + omega_vec[None, None, :] / (RD(2.0) * jnp.pi * RD(f0)))
        NL = jnp.fft.ifft(NLw, axis=2).astype(CD)

    dA = (ONEJ * RD(gamma)).astype(CD) * NL

    # Saturable gain (optional)
    def _add_gain(args):
        dA_in, A_in, absA2_in = args
        power_xy  = jnp.sum(absA2_in, axis=2) * RD(dt)
        gain_pref = RD(1.0) / (RD(1.0) + power_xy / RD(saturation_intensity))
        gain_pref = gain_pref[:, :, None].astype(RD)
        Aw = jnp.fft.fft(A_in, axis=2).astype(CD)
        A_gain_w = gain_term.astype(CD) * (gain_pref.astype(CD) * Aw)
        return dA_in + jnp.fft.ifft(A_gain_w, axis=2).astype(CD)

    dA = jax.lax.cond(use_gain, _add_gain, lambda args: dA, (dA, A_xy_t, absA2))
    return dA.astype(CD)


# --------------------------------------------------------------------------------------
# Precompute propagation operators
# --------------------------------------------------------------------------------------
def _prepare_propagation(args, A0, *, precision: str = "fp64"):
    RD, CD, NPD = _resolve_precision(precision)

    Lx, Ly, Lz, Lt = args["Lx"], args["Ly"], args["Lz"], args["Lt"]
    Nx, Ny, Nt     = args["Nx"], args["Ny"], args["Nt"]
    dx, dy, dt     = Lx/Nx, Ly/Ny, Lt/Nt

    deltaZ_linear = float(args["deltaZ"])
    deltaZ_NL     = float(args["deltaZ_NL"])
    steps_total   = int(round(Lz / deltaZ_linear))

    # Save indices
    save_at_m = np.asarray(args["save_at"], dtype=np.float64)
    save_idx  = np.unique(np.clip(np.rint(save_at_m / deltaZ_linear).astype(np.int32),
                                  0, max(0, steps_total-1)))
    save_idx  = jnp.asarray(save_idx, dtype=jnp.int32)
    save_n    = int(save_idx.size)

    # Physics constants
    c0      = 2.997_924_58e8
    lambda0 = float(args["lambda0"])
    f0      = c0 / lambda0
    omega0  = RD(2) * jnp.pi * RD(f0)
    n2      = float(args["n2"])
    gamma   = RD(n2) * omega0 / RD(c0)

    beta0, beta1, beta2 = float(args["beta0"]), float(args["beta1"]), float(args["beta2"])
    gain_coeff, gain_fwhm = float(args["gain_coeff"]), float(args["gain_fwhm"])
    use_gain = bool(gain_coeff != 0.0)
    t1, t2 = float(args["t1"]), float(args["t2"])

    # k / omega grids
    omega = RD(2) * jnp.pi * jnp.fft.fftfreq(Nt, RD(dt)).astype(RD)
    kx    = RD(2) * jnp.pi * jnp.fft.fftfreq(Nx, RD(dx)).astype(RD)
    ky    = RD(2) * jnp.pi * jnp.fft.fftfreq(Ny, RD(dy)).astype(RD)

    KX, KY, OMEGA = kx[:, None, None], ky[None, :, None], omega[None, None, :]

    # Material index (x,y,omega)
    n_xyomega = jnp.asarray(args["n_xyomega"], dtype=RD)
    n_xyomega = jnp.broadcast_to(n_xyomega, (Nx, Ny, Nt))
    n_eff_omega = n_xyomega[Nx//2, Ny//2, :]
    beta_eff = (n_eff_omega[None,None,:] * (omega0 + OMEGA) / RD(c0)).astype(RD)

    # Linear spectral propagator
    ONEJ = jax.lax.complex(RD(0.0), RD(1.0))
    rad = beta_eff**2 - KX**2 - KY**2
    sqrt_term = jnp.sqrt(rad.astype(CD))
    D = ONEJ * (sqrt_term - RD(beta0) - RD(beta1)*OMEGA - RD(0.5)*RD(beta2)*OMEGA**2).astype(CD)
    D_half = jnp.exp(D * RD(deltaZ_linear/2)).astype(CD)

    # PML (disable on singleton axes)
    pml_thickness = int(args["pml_thickness"])
    pml_Wmax      = float(args["pml_Wmax"])

    idx = jnp.arange(Nx); idy = jnp.arange(Ny)
    d_x = jnp.minimum(idx, (Nx-1)-idx)
    d_y = jnp.minimum(idy, (Ny-1)-idy)

    if Nx > 1:
        ramp_x = jnp.where(d_x < pml_thickness, RD(pml_Wmax)*((RD(pml_thickness)-d_x)/RD(pml_thickness))**2, RD(0))
    else:
        ramp_x = jnp.zeros(Nx, dtype=RD)

    if Ny > 1:
        ramp_y = jnp.where(d_y < pml_thickness, RD(pml_Wmax)*((RD(pml_thickness)-d_y)/RD(pml_thickness))**2, RD(0))
    else:
        ramp_y = jnp.zeros(Ny, dtype=RD)

    W2d = (ramp_x[:,None] + ramp_y[None,:]).astype(RD)
    PML_half = jnp.exp(-W2d * RD(deltaZ_linear/2)).astype(RD)

    # Waveguide mode coupling
    Nprop_half = jnp.exp(
        (ONEJ * beta_eff / RD(2)) * ((n_xyomega/n_eff_omega[None,None,:])**2 - RD(1.0)) * RD(deltaZ_linear/2)
    ).astype(CD)

    # Raman kernel
    hrw = _make_hrw(Nt, dt, t1, t2, real_dtype=RD)

    # Gain spectral envelope
    if use_gain:
        omega_fwhm = RD(2) * jnp.pi * RD(gain_fwhm)
        omega_mid  = omega_fwhm / (RD(2) * jnp.sqrt(jnp.log(RD(2))))
        g0 = RD(gain_coeff)/RD(2)
        gain_term = (g0 * jnp.exp(-(OMEGA**2)/(RD(2)*omega_mid**2))).astype(CD)
        use_gain_flag = True
    else:
        gain_term = jnp.zeros((1,1,Nt), dtype=CD)
        use_gain_flag = False

    # NL stepping strategy
    if deltaZ_NL <= deltaZ_linear:
        nl_outer_subcycles = int(max(1, round(deltaZ_linear / deltaZ_NL)))
        skip_nl_every = 1
    else:
        nl_outer_subcycles = 1
        skip_nl_every = int(max(1, round(deltaZ_NL / deltaZ_linear)))

    m_nl = int(args.get("m_nl_substeps", 1))

    return dict(
        steps_total=int(steps_total),
        save_idx=save_idx, save_n=int(save_n),
        dt=RD(dt), dx=RD(dx), dy=RD(dy),
        omega_vec=omega.astype(RD),
        f0=RD(C0/float(args["lambda0"])),
        gamma=RD(gamma),
        D_half=D_half, PML_half=PML_half, Nprop_half=Nprop_half,
        hrw=hrw, gain_term=gain_term,
        fr=float(args["fr"]), sw=int(args["sw"]),
        deltaZ_linear=RD(deltaZ_linear),
        deltaZ_NL=RD(deltaZ_NL),
        use_gain=use_gain_flag,
        m_nl_substeps=m_nl,
        nl_outer_subcycles=nl_outer_subcycles,
        skip_nl_every=skip_nl_every,
    )


# --------------------------------------------------------------------------------------
# Sharding utilities
# --------------------------------------------------------------------------------------
def _best_1d_factor(n_devices: int, n_axis: int) -> int:
    best = 1
    for d in range(1, n_devices + 1):
        if (n_devices % d == 0) and (n_axis % d == 0):
            best = d
    return best


def _make_mesh_for_time_axis(Nt: int):
    devs = jax.devices()
    ndev = len(devs)
    if ndev == 0:
        raise RuntimeError("No JAX devices available.")
    n_shards_t = _best_1d_factor(ndev, Nt)
    mesh_arr = np.array(devs[:max(1, n_shards_t)]).reshape((max(1, n_shards_t),))
    mesh = Mesh(mesh_arr, axis_names=('t',))
    shard_t   = NamedSharding(mesh, P(None, None, 't'))
    replicate = NamedSharding(mesh, P(None, None, None))
    return mesh, shard_t, replicate


# --------------------------------------------------------------------------------------
# Split-step kernel (one z-step)
# --------------------------------------------------------------------------------------
@partial(
    jax.jit,
    static_argnames=('fr','sw','use_gain','m_nl_substeps','nl_outer_subcycles','shard_t','replicate'),
    donate_argnums=(0,)
)
def split_step_sharded(field_kwo, *,
                       shard_t, replicate,
                       dt, f0, fr, sw,
                       deltaZ_linear, deltaZ_NL,
                       gamma, omega_vec,
                       D_half, Nprop_half, PML_half,
                       hrw, gain_term, saturation_intensity, use_gain,
                       m_nl_substeps=1,
                       nl_outer_subcycles=1,
                       apply_nl=True,
                       ):

    CD = field_kwo.dtype
    RD = jnp.float32 if CD == jnp.complex64 else jnp.float64
    ONEJ = jax.lax.complex(RD(0.0), RD(1.0))

    # 1) half linear (spectral)
    field_kwo = jax.lax.with_sharding_constraint(field_kwo, shard_t)
    field_kwo = field_kwo * D_half

    # 2) half spatial linear (PML + waveguide)
    field_xyw = jnp.fft.ifftn(field_kwo, axes=(0,1))
    field_xyw = field_xyw * (PML_half[:, :, None].astype(RD)) * Nprop_half

    def _do_nl(field_xyw_local, apply_residual: bool):
        field_xyw_rep = jax.lax.with_sharding_constraint(field_xyw_local, replicate)
        field_xyt = jnp.fft.ifft(field_xyw_rep, axis=2)

        def kerr_half(A, dz):
            return A * jnp.exp(ONEJ * RD(gamma) * RD(0.5) * RD(dz) * jnp.abs(A)**2)

        def residual_heun(A, dz):
            h = RD(dz) / RD(m_nl_substeps)
            def step(a,_):
                k1 = _dA_dz_NL_rest(a, dt=dt, f0=f0, fr=fr, sw=sw, gamma=gamma,
                                    omega_vec=omega_vec, hrw=hrw, gain_term=gain_term,
                                    saturation_intensity=saturation_intensity, use_gain=use_gain)
                a1 = a + h*k1
                k2 = _dA_dz_NL_rest(a1, dt=dt, f0=f0, fr=fr, sw=sw, gamma=gamma,
                                    omega_vec=omega_vec, hrw=hrw, gain_term=gain_term,
                                    saturation_intensity=saturation_intensity, use_gain=use_gain)
                return a + RD(0.5)*h*(k1+k2), None
            A_out, _ = jax.lax.scan(step, A, xs=None, length=m_nl_substeps)
            return A_out

        m  = int(nl_outer_subcycles)
        dz = deltaZ_linear / m

        def one_cycle(A,_):
            A = kerr_half(A, dz)
            A = jax.lax.cond(apply_residual,
                            lambda a: residual_heun(a, dz),
                            lambda a: a,
                            A)
            A = kerr_half(A, dz)
            return A, None

        field_xyt, _ = jax.lax.scan(one_cycle, field_xyt, xs=None, length=m)
        return jnp.fft.fft(field_xyt, axis=2)

    field_xyw = _do_nl(field_xyw, apply_residual=apply_nl)
    field_xyw = jax.lax.with_sharding_constraint(field_xyw, shard_t)

    # 3) finish spatial linear + back to spectral
    field_xyw = field_xyw * (PML_half[:, :, None].astype(RD)) * Nprop_half
    field_kwo = jnp.fft.fftn(field_xyw, axes=(0,1))

    # 4) finish spectral linear
    field_kwo = field_kwo * D_half
    return field_kwo


# ======================================================================================
# ORIGINAL SCAN-BASED PROPAGATOR (with snapshots + events in carry)
# ======================================================================================
def make_propagate_scan_sharded_checkpointed(
    shard_t, replicate,
    *,
    event_fn=None,
    stop_on_event=True,
    event_check_every: int = 1,
    strategy: str = "segments",
    segment_len: int = 16,
    tree_depth: int = 2,
    base_len: int = 32,
):
    use_event   = bool(callable(event_fn) and stop_on_event)
    check_every = int(max(1, event_check_every))

    def _materialize_xyt_from_kwo(field_kwo):
        field_kwo = jax.lax.with_sharding_constraint(field_kwo, replicate)
        field_xyw = jnp.fft.ifftn(field_kwo, axes=(0,1))
        field_xyt = jnp.fft.ifft(field_xyw, axis=2)
        return field_xyt

    @partial(
        jax.jit,
        static_argnames=('steps_total','save_n','fr','sw','use_gain',
                         'm_nl_substeps','nl_outer_subcycles','skip_nl_every',
                         'strategy','segment_len','tree_depth','base_len',
                         'save_as_fp32'),
        donate_argnums=(0,),
    )
    def _propagate_scan_ckpt(
        A0_kwo: jnp.ndarray,
        *,
        payload,
        steps_total: int,
        save_idx: jnp.ndarray, save_n: int,
        dt: float, f0: float,
        fr: float, sw: int,
        deltaZ_linear: float,
        deltaZ_NL: float,
        gamma: float,
        omega_vec: jnp.ndarray,
        D_half: jnp.ndarray,
        PML_half: jnp.ndarray,
        Nprop_half: jnp.ndarray,
        hrw: jnp.ndarray,
        gain_term: jnp.ndarray,
        saturation_intensity: float,
        use_gain: bool,
        m_nl_substeps: int,
        nl_outer_subcycles: int,
        skip_nl_every: int,
        strategy: str = "segments",
        segment_len: int = 16,
        tree_depth: int = 2,
        base_len: int = 32,
        save_as_fp32: bool = False,
    ):
        CD_sim = A0_kwo.dtype
        CD_save = (jnp.complex64 if (save_as_fp32 and CD_sim == jnp.complex128) else CD_sim)

        Nx, Ny, Nt = A0_kwo.shape
        field_kwo0 = jax.lax.with_sharding_constraint(A0_kwo, shard_t)

        save_buf0  = jnp.zeros((Nx, Ny, Nt, save_n), dtype=CD_save) if save_n > 0 else jnp.zeros((Nx,Ny,Nt,0), dtype=CD_save)
        save_ptr0  = jnp.array(0, dtype=jnp.int32)
        i0         = jnp.array(0, dtype=jnp.int32)
        done0      = jnp.array(False)
        z_event0   = jnp.array(jnp.nan, dtype=(jnp.float32 if CD_sim == jnp.complex64 else jnp.float64))

        def one_step(state, _):
            i, field_kwo, save_ptr, save_buf, done, z_event = state
            apply_nl = ((i + 1) % jnp.asarray(skip_nl_every, dtype=i.dtype)) == 0

            field_kwo = split_step_sharded(
                field_kwo,
                shard_t=shard_t, replicate=replicate,
                dt=dt, f0=f0, fr=fr, sw=sw,
                deltaZ_linear=deltaZ_linear, deltaZ_NL=deltaZ_NL,
                gamma=gamma, omega_vec=omega_vec,
                D_half=D_half, Nprop_half=Nprop_half, PML_half=PML_half,
                hrw=hrw, gain_term=gain_term,
                saturation_intensity=saturation_intensity,
                use_gain=use_gain,
                m_nl_substeps=m_nl_substeps,
                nl_outer_subcycles=nl_outer_subcycles,
                apply_nl=apply_nl,
            )

            z_here = (i + 1).astype(z_event.dtype) * deltaZ_linear

            # Save path
            can_save   = save_ptr < save_n
            want_index = jnp.where(can_save, save_idx[save_ptr], -1)
            save_now   = jnp.logical_and(can_save, i == want_index)

            def do_save(args):
                field_kwo_in, save_buf_in, save_ptr_in = args
                xyt = _materialize_xyt_from_kwo(field_kwo_in).astype(CD_save)
                save_buf_out = save_buf_in.at[..., save_ptr_in].set(xyt)
                return (save_buf_out, save_ptr_in + 1)

            save_buf, save_ptr = jax.lax.cond(
                save_now,
                do_save,
                lambda args: (args[1], args[2]),
                (field_kwo, save_buf, save_ptr),
            )

            # Event path
            if use_event:
                check_now = ((i + 1) % jnp.asarray(check_every, dtype=i.dtype)) == 0

                def do_event(args):
                    field_kwo_in, z_val = args
                    xyt = _materialize_xyt_from_kwo(field_kwo_in)
                    return event_fn(xyt, z_val, payload)

                triggered = jax.lax.cond(
                    check_now,
                    do_event,
                    lambda _: jnp.array(False),
                    (field_kwo, z_here),
                )
            else:
                triggered = jnp.array(False)

            done    = jnp.logical_or(done, triggered)
            z_event = jnp.where(jnp.logical_and(triggered, jnp.isnan(z_event)), z_here, z_event)

            return (i + 1, field_kwo, save_ptr, save_buf, done, z_event), None

        # Runners
        def run_segment(state, n: int, remat: bool):
            body = one_step
            if remat:
                body = jax.checkpoint(body, prevent_cse=False)
            state, _ = jax.lax.scan(body, state, xs=None, length=int(n))
            return state

        def run_none(state):
            return run_segment(state, int(steps_total), remat=False)

        def run_segments(state):
            N = int(steps_total); S = int(max(1, segment_len)); k = 0
            while k < N:
                nseg = min(S, N - k)
                state = run_segment(state, nseg, remat=True)
                k += nseg
            return state

        def run_tree(state):
            N = int(steps_total); D = int(tree_depth); B = int(base_len)
            def build(n, d):
                if (n <= B) or (d <= 0): return [n]
                n1 = n // 2; n2 = n - n1
                return build(n1, d-1) + build(n2, d-1)
            for nseg in build(N, D):
                state = run_segment(state, int(nseg), remat=True)
            return state

        state0 = (i0, field_kwo0, save_ptr0, save_buf0, done0, z_event0)
        if strategy == "none":
            state_end = run_none(state0)
        elif strategy == "segments":
            state_end = run_segments(state0)
        elif strategy == "tree":
            state_end = run_tree(state0)
        else:
            raise ValueError("strategy must be one of: 'none', 'segments', 'tree'")

        i_end, field_end_kwo, save_ptr_end, save_buf, done_end, z_event = state_end

        # Tail save
        def _tail_save(args):
            save_buf_in, field_end_kwo_in, save_ptr_in = args
            xyt_end = _materialize_xyt_from_kwo(field_end_kwo_in).astype(CD_save)
            return save_buf_in.at[..., save_ptr_in].set(xyt_end)

        pred_tail = save_ptr_end < jnp.int32(save_n)
        save_buf = jax.lax.cond(
            pred_tail,
            _tail_save,
            lambda x: x[0],
            (save_buf, field_end_kwo, save_ptr_end),
        )

        n_saved = jax.lax.select(pred_tail, save_ptr_end + jnp.int32(1), save_ptr_end)

        meta = dict(
            steps_executed=i_end,
            stopped_early=done_end,
            z_event=z_event,
            n_saved=n_saved,
        )
        return save_buf, meta

    return _propagate_scan_ckpt


# --------------------------------------------------------------------------------------
# GNLSE3D_propagate — original public entry (forward with snapshots + events)
# --------------------------------------------------------------------------------------
def GNLSE3D_propagate(
    args, A0,
    *,
    event_fn=None,
    event_payload=None,
    stop_on_event=True,
    event_check_every: int = 1e10,
    ckpt_strategy: str | None = None,
    ckpt_segment_len: int | None = None,
    ckpt_tree_depth: int | None = None,
    ckpt_base_len: int | None = None,
    precision: str | None = None,
    save_as_fp32: bool = True,
):
    prec = precision or args.get("precision", "fp64")
    RD, CD, _ = _resolve_precision(prec)

    prep = _prepare_propagation(args, A0, precision=prec)

    A0_kwo = jnp.fft.fftn(jnp.asarray(A0, dtype=CD), axes=(0,1,2))
    _, shard_t, replicate = _make_mesh_for_time_axis(A0_kwo.shape[2])

    strategy  = ckpt_strategy   or args.get("ckpt_strategy",   "segments")
    seglen    = ckpt_segment_len if ckpt_segment_len is not None else args.get("ckpt_segment_len", 16)
    treedepth = ckpt_tree_depth  if ckpt_tree_depth  is not None else args.get("ckpt_tree_depth", 2)
    baselen   = ckpt_base_len    if ckpt_base_len    is not None else args.get("ckpt_base_len", 32)

    prop_scan = make_propagate_scan_sharded_checkpointed(
        shard_t, replicate,
        event_fn=event_fn,
        stop_on_event=stop_on_event,
        event_check_every=event_check_every,
        strategy=strategy,
        segment_len=int(seglen),
        tree_depth=int(treedepth),
        base_len=int(baselen),
    )

    t0 = time.time()
    field_saved, meta = prop_scan(
        A0_kwo,
        payload=(event_payload if event_payload is not None else {}),
        steps_total=prep["steps_total"],
        save_idx=prep["save_idx"], save_n=prep["save_n"],
        dt=prep["dt"], f0=prep["f0"],
        fr=prep["fr"], sw=prep["sw"],
        deltaZ_linear=prep["deltaZ_linear"],
        deltaZ_NL=prep["deltaZ_NL"],
        gamma=prep["gamma"],
        omega_vec=prep["omega_vec"],
        D_half=prep["D_half"],
        PML_half=prep["PML_half"],
        Nprop_half=prep["Nprop_half"],
        hrw=prep["hrw"],
        gain_term=prep["gain_term"],
        saturation_intensity=args["saturation_intensity"],
        use_gain=prep["use_gain"],
        m_nl_substeps=prep["m_nl_substeps"],
        nl_outer_subcycles=prep["nl_outer_subcycles"],
        skip_nl_every=prep["skip_nl_every"],
        strategy=strategy, segment_len=int(seglen), tree_depth=int(treedepth), base_len=int(baselen),
        save_as_fp32=bool(save_as_fp32),
    )
    elapsed = time.time() - t0

    return dict(field=field_saved, dt=prep["dt"], dx=prep["dx"], seconds=elapsed, **meta)


# ======================================================================================
# LEAN PROPAGATOR — field-only carry (building block for windowed gradient)
# ======================================================================================
def _materialize_xyt(field_kwo, replicate):
    """Transform k-space field to spatial (x,y,t) domain."""
    field_kwo = jax.lax.with_sharding_constraint(field_kwo, replicate)
    field_xyw = jnp.fft.ifftn(field_kwo, axes=(0, 1))
    return jnp.fft.ifft(field_xyw, axis=2)


def make_propagate_lean(shard_t, replicate):
    """Return a JIT-compiled function: field_kwo -> field_kwo_final.

    The scan carry contains only (i, field_kwo) — no save_buf — so
    backward-pass memory scales with field size, not snapshot buffer size.
    """

    @partial(
        jax.jit,
        static_argnames=(
            'steps_total', 'fr', 'sw', 'use_gain',
            'm_nl_substeps', 'nl_outer_subcycles', 'skip_nl_every',
        ),
        # No donate_argnums: donation conflicts with jax.value_and_grad
        # which needs the input buffer alive for the backward pass.
    )
    def _propagate_lean(
        A0_kwo: jnp.ndarray,
        *,
        steps_total: int,
        dt: float, f0: float,
        fr: float, sw: int,
        deltaZ_linear: float,
        deltaZ_NL: float,
        gamma: float,
        omega_vec: jnp.ndarray,
        D_half: jnp.ndarray,
        PML_half: jnp.ndarray,
        Nprop_half: jnp.ndarray,
        hrw: jnp.ndarray,
        gain_term: jnp.ndarray,
        saturation_intensity: float,
        use_gain: bool,
        m_nl_substeps: int,
        nl_outer_subcycles: int,
        skip_nl_every: int,
    ):
        field_kwo0 = jax.lax.with_sharding_constraint(A0_kwo, shard_t)
        i0 = jnp.array(0, dtype=jnp.int32)

        def one_step(state, _):
            i, field_kwo = state
            apply_nl = ((i + 1) % jnp.asarray(skip_nl_every, dtype=i.dtype)) == 0

            field_kwo = split_step_sharded(
                field_kwo,
                shard_t=shard_t, replicate=replicate,
                dt=dt, f0=f0, fr=fr, sw=sw,
                deltaZ_linear=deltaZ_linear, deltaZ_NL=deltaZ_NL,
                gamma=gamma, omega_vec=omega_vec,
                D_half=D_half, Nprop_half=Nprop_half, PML_half=PML_half,
                hrw=hrw, gain_term=gain_term,
                saturation_intensity=saturation_intensity,
                use_gain=use_gain,
                m_nl_substeps=m_nl_substeps,
                nl_outer_subcycles=nl_outer_subcycles,
                apply_nl=apply_nl,
            )
            return (i + 1, field_kwo), None

        # Body-level checkpoint: lax.scan stores only carries, not
        # the body's internal intermediates (FFTs, Kerr products).
        ckpt_body = jax.checkpoint(one_step, prevent_cse=False)

        state0 = (i0, field_kwo0)
        state_end, _ = jax.lax.scan(ckpt_body, state0, xs=None, length=int(steps_total))

        _, field_end_kwo = state_end
        return field_end_kwo

    return _propagate_lean


# --------------------------------------------------------------------------------------
# Helper: build the kwargs dict for the lean propagator
# --------------------------------------------------------------------------------------
def _build_lean_kw(prep, args):
    """Build kwargs dict for _propagate_lean from _prepare_propagation output."""
    return dict(
        dt=prep["dt"], f0=prep["f0"],
        fr=prep["fr"], sw=prep["sw"],
        deltaZ_linear=prep["deltaZ_linear"],
        deltaZ_NL=prep["deltaZ_NL"],
        gamma=prep["gamma"],
        omega_vec=prep["omega_vec"],
        D_half=prep["D_half"],
        PML_half=prep["PML_half"],
        Nprop_half=prep["Nprop_half"],
        hrw=prep["hrw"],
        gain_term=prep["gain_term"],
        saturation_intensity=args["saturation_intensity"],
        use_gain=prep["use_gain"],
        m_nl_substeps=prep["m_nl_substeps"],
        nl_outer_subcycles=prep["nl_outer_subcycles"],
        skip_nl_every=prep["skip_nl_every"],
    )


# --------------------------------------------------------------------------------------
# GNLSE3D_propagate_lean — forward with snapshots, lean carry
# --------------------------------------------------------------------------------------
def GNLSE3D_propagate_lean(
    args, A0,
    *,
    precision: str | None = None,
    save_as_fp32: bool = True,
):
    """Forward propagation returning (field_saved, meta) like GNLSE3D_propagate,
    but using the lean propagator internally.

    Snapshots are collected outside the scan by replaying sub-propagations to
    each save index.  For gradient computation, use windowed_grad instead.
    """
    prec = precision or args.get("precision", "fp64")
    RD, CD, _ = _resolve_precision(prec)
    prep = _prepare_propagation(args, A0, precision=prec)
    CD_save = jnp.complex64 if (save_as_fp32 and CD == jnp.complex128) else CD

    A0_kwo = jnp.fft.fftn(jnp.asarray(A0, dtype=CD), axes=(0, 1, 2))
    _, shard_t, replicate = _make_mesh_for_time_axis(A0_kwo.shape[2])

    prop_lean = make_propagate_lean(shard_t, replicate)
    lean_kw = _build_lean_kw(prep, args)

    # Collect snapshots by running sub-propagations to each save index.
    # In the prototype, save_now = (i == save_idx[si]) fires AFTER physics
    # is applied at step i.  So save_idx[k] means "save after completing
    # step k", i.e. the field has been propagated (k+1) times from A0.
    save_idx = np.asarray(prep["save_idx"])
    save_n = int(prep["save_n"])
    Nx, Ny, Nt = A0_kwo.shape

    save_buf = np.zeros((Nx, Ny, Nt, save_n),
                        dtype=np.complex64 if save_as_fp32 else np.complex128)

    t0 = time.time()

    field_kwo = A0_kwo
    steps_done = 0
    for si in range(save_n):
        target_done = int(save_idx[si]) + 1   # +1: prototype saves AFTER step i
        n_steps_here = target_done - steps_done
        if n_steps_here > 0:
            field_kwo = prop_lean(field_kwo, **lean_kw, steps_total=n_steps_here)
        snap = _materialize_xyt(field_kwo, replicate).astype(CD_save)
        save_buf[..., si] = np.asarray(jax.device_get(snap))
        steps_done = target_done

    remaining = prep["steps_total"] - steps_done
    if remaining > 0:
        field_kwo = prop_lean(field_kwo, **lean_kw, steps_total=remaining)

    elapsed = time.time() - t0

    meta = dict(
        steps_executed=jnp.array(prep["steps_total"], dtype=jnp.int32),
        stopped_early=jnp.array(False),
        z_event=jnp.array(jnp.nan),
        n_saved=jnp.array(save_n, dtype=jnp.int32),
    )
    return dict(field=jnp.asarray(save_buf), dt=prep["dt"], dx=prep["dx"],
                seconds=elapsed, **meta)


# ======================================================================================
# WINDOWED PROPAGATION — memory-efficient gradient via per-window VJP
# ======================================================================================
#
# Forward:
#   A0 -[n_steps]-> A1 -[n_steps]-> ... -[n_steps]-> A_W
#   Store each A_w on host (numpy).
#
# Gradient (reverse):
#   Starting from dL/dA_W, for w = W-1 down to 0:
#     Reload A_w to device
#     A_{w+1}, vjp_fn = jax.vjp(propagate, A_w)
#     dL/dA_w = vjp_fn(dL/dA_{w+1})
#
# Peak GPU memory = one window's backward (steps_per_window carries).

def _uniform_window_steps(steps_total, n_windows):
    """Distribute steps across windows.  Last window absorbs remainder."""
    base = steps_total // n_windows
    window_steps = [base] * n_windows
    window_steps[-1] += steps_total - base * n_windows
    return window_steps


def make_windowed_context(
    args,
    Nt: int,
    *,
    precision: str | None = None,
):
    """Build a reusable context dict for windowed_forward / windowed_grad.

    Call this **once** before a GD loop and pass as ``ctx=``.
    This avoids repeated JIT recompilation.
    """
    prec = precision or args.get("precision", "fp64")
    RD, CD, _ = _resolve_precision(prec)

    prep = _prepare_propagation(args, np.zeros((1, 1, Nt)), precision=prec)
    _, shard_t, replicate = _make_mesh_for_time_axis(Nt)

    prop_lean = make_propagate_lean(shard_t, replicate)
    lean_kw = _build_lean_kw(prep, args)

    return dict(
        prop_lean=prop_lean,
        lean_kw=lean_kw,
        prep=prep,
        CD=CD,
        shard_t=shard_t,
        replicate=replicate,
    )


def _get_or_build_ctx(args, Nt, precision, ctx):
    """Return *ctx* if provided, otherwise build one."""
    if ctx is not None:
        return ctx
    return make_windowed_context(args, Nt, precision=precision)


def windowed_forward(
    args, A0,
    *,
    n_windows: int = 10,
    precision: str | None = None,
    save_as_fp32: bool = True,
    ctx: dict | None = None,
):
    """Run the full propagation in *n_windows* independent JIT calls.

    Returns dict with 'field_final' (xyt), 'checkpoints' (host numpy), 'seconds'.
    """
    prec = precision or args.get("precision", "fp64")
    RD, CD, _ = _resolve_precision(prec)
    CD_save = jnp.complex64 if (save_as_fp32 and CD == jnp.complex128) else CD

    A0_kwo = jnp.fft.fftn(jnp.asarray(A0, dtype=CD), axes=(0, 1, 2))
    Nx, Ny, Nt = A0_kwo.shape

    c = _get_or_build_ctx(args, Nt, prec, ctx)
    prop_lean = c["prop_lean"]
    lean_kw   = c["lean_kw"]
    prep      = c["prep"]
    replicate = c["replicate"]

    steps_total = prep["steps_total"]
    window_steps = _uniform_window_steps(steps_total, n_windows)

    t0 = time.time()

    checkpoints_host = [np.asarray(jax.device_get(A0_kwo))]
    field_kwo = A0_kwo

    for w in range(n_windows):
        if window_steps[w] == 0:
            checkpoints_host.append(checkpoints_host[-1].copy())
            continue
        field_kwo = prop_lean(field_kwo, **lean_kw, steps_total=window_steps[w])
        field_kwo.block_until_ready()
        checkpoints_host.append(np.asarray(jax.device_get(field_kwo)))

    elapsed = time.time() - t0

    field_final_xyt = _materialize_xyt(field_kwo, replicate).astype(CD_save)

    return dict(
        field_final=field_final_xyt,
        checkpoints=checkpoints_host,
        n_windows=n_windows,
        steps_per_window=window_steps[0],
        steps_total=steps_total,
        seconds=elapsed,
        dt=prep["dt"], dx=prep["dx"],
        _ctx=c,
    )


def windowed_grad(
    loss_fn,
    args, A0,
    *,
    n_windows: int = 10,
    precision: str | None = None,
    ctx: dict | None = None,
):
    """Compute dL/dA0_kwo by back-propagating through windows in reverse.

    Parameters
    ----------
    loss_fn : callable
        Maps field_kwo_final (device array, k-space) -> scalar loss.
        Must be JAX-differentiable.
    ctx : dict, optional
        Pre-built context from make_windowed_context.  **Always** pass
        this in a GD loop to avoid per-step recompilation.

    Returns
    -------
    dict with keys: loss, grad, fwd_seconds, bwd_seconds
    """
    prec = precision or args.get("precision", "fp64")
    RD, CD, _ = _resolve_precision(prec)
    A0_kwo = jnp.fft.fftn(jnp.asarray(A0, dtype=CD), axes=(0, 1, 2))
    Nt = A0_kwo.shape[2]

    c = _get_or_build_ctx(args, Nt, prec, ctx)
    prop_lean = c["prop_lean"]
    lean_kw   = c["lean_kw"]
    prep      = c["prep"]

    steps_total = prep["steps_total"]
    window_steps = _uniform_window_steps(steps_total, n_windows)

    # Cache closures: at most 2 distinct step counts (base + possibly last window).
    _prop_cache = {}
    def _get_window_prop(n_steps):
        if n_steps not in _prop_cache:
            def _wp(a_start, _ns=n_steps):
                return prop_lean(a_start, **lean_kw, steps_total=_ns)
            _prop_cache[n_steps] = _wp
        return _prop_cache[n_steps]

    # ---- Forward: store checkpoints on host ----
    t_fwd = time.time()
    checkpoints_host = [np.asarray(jax.device_get(A0_kwo))]
    field_kwo = A0_kwo
    for w in range(n_windows):
        if window_steps[w] == 0:
            checkpoints_host.append(checkpoints_host[-1].copy())
            continue
        field_kwo = prop_lean(field_kwo, **lean_kw, steps_total=window_steps[w])
        field_kwo.block_until_ready()
        checkpoints_host.append(np.asarray(jax.device_get(field_kwo)))
    fwd_seconds = time.time() - t_fwd

    # ---- Evaluate loss at final field ----
    field_final = jnp.asarray(checkpoints_host[-1])
    loss_val, grad_final = jax.value_and_grad(loss_fn)(field_final)

    # ---- Backward: vjp through each window in reverse ----
    t_bwd = time.time()
    grad_carry = grad_final
    for w in reversed(range(n_windows)):
        if window_steps[w] == 0:
            continue
        A_start = jnp.asarray(checkpoints_host[w])

        wp = _get_window_prop(window_steps[w])
        _, vjp_fn = jax.vjp(wp, A_start)
        (grad_carry,) = vjp_fn(grad_carry)
        grad_carry.block_until_ready()

    bwd_seconds = time.time() - t_bwd

    return dict(
        loss=loss_val,
        grad=grad_carry,
        fwd_seconds=fwd_seconds,
        bwd_seconds=bwd_seconds,
    )


# ── Convenience wrappers ────────────────────────────────────────────────────


def _make_args_base(
    Nx=1, Ny=1, Nt=1,
    Lx=1e-6, Ly=1e-6, Lt=1e-12, Lz=1e-3,
    n_xyomega=None, n_ref=1.453,
    n2_val=0.0,
    lambda0=1064e-9,
    beta0_val=None, beta1_val=None, beta2_val=0.0,
    deltaZ=10e-6, deltaZ_NL=None,
    fr=0.0, sw=0,
    pml_thickness=15, pml_Wmax=1e12,
    gain_coeff=0.0, gain_fwhm=0.0,
    saturation_intensity=1e20,
    t1=12.2e-15, t2=32e-15,
    n_saves=20,
    precision="fp64",
    **overrides,
):
    """Build a complete ``args`` dict for GNLSE propagation.

    Parameters
    ----------
    n_ref : float
        Reference refractive index, used to fill *n_xyomega* when it is
        ``None`` (uniform medium).  Default 1.453 (silica at 1064 nm).
    lambda0 : float
        Central wavelength [m].
    beta0_val, beta1_val : float or None
        Propagation constant and inverse group velocity.  If ``None``,
        derived from *n_ref*: ``beta0 = n_ref * 2π/λ₀``,
        ``beta1 = n_ref / c₀`` (group index ≈ phase index approx).
    t1, t2 : float
        Raman time constants [s].  Defaults are silica values.

    All other parameters mirror the keys of the ``args`` dict consumed by
    the solvers — see ``CLAUDE.md`` for the full list.
    """
    if beta0_val is None:
        beta0_val = n_ref * 2 * np.pi / lambda0
    if beta1_val is None:
        beta1_val = n_ref / C0
    if deltaZ_NL is None:
        deltaZ_NL = deltaZ

    if n_xyomega is None:
        n_xyomega = jnp.full((Nx, Ny, Nt), n_ref)
    elif n_xyomega.ndim == 2:
        n_xyomega = jnp.tile(n_xyomega[:, :, None], (1, 1, Nt))

    save_at = np.linspace(0, Lz, n_saves + 1)[1:]

    args = {
        "Lx": Lx, "Ly": Ly, "Lz": Lz, "Lt": Lt,
        "Nx": Nx, "Ny": Ny, "Nt": Nt,
        "n_xyomega": n_xyomega,
        "n2": n2_val,
        "beta0": beta0_val, "beta1": beta1_val, "beta2": beta2_val,
        "lambda0": lambda0,
        "deltaZ": deltaZ, "deltaZ_NL": deltaZ_NL,
        "save_at": save_at,
        "saturation_intensity": saturation_intensity,
        "gain_coeff": gain_coeff, "gain_fwhm": gain_fwhm,
        "t1": t1, "t2": t2,
        "pml_thickness": pml_thickness, "pml_Wmax": pml_Wmax,
        "fr": fr, "sw": sw,
        "precision": precision,
    }
    args.update(overrides)
    return args


def propagate_windowed(args, field, n_windows=None, ctx=None):
    """Propagate and return snapshots in visualisation-friendly format.

    Wraps :func:`windowed_forward` and materialises the k-space
    checkpoints to ``(Nx, Ny, Nt, Nsave)`` complex-64 on the host.

    Parameters
    ----------
    args : dict
        From :func:`make_args`.
    field : array (Nx, Ny, Nt)
        Initial field.
    n_windows : int or None
        Number of windows.  Defaults to ``len(args['save_at'])``.
    ctx : dict or None
        Pre-built context from :func:`make_windowed_context`.

    Returns
    -------
    dict with keys ``field``, ``save_at``, ``dt``, ``dx``, ``dy``,
    ``seconds``.
    """
    save_at = np.asarray(args["save_at"])
    n_saves = len(save_at)
    if n_windows is None:
        n_windows = max(n_saves, 1)

    Nx, Ny, Nt = args["Nx"], args["Ny"], args["Nt"]
    Lz = args["Lz"]
    dx, dy, dt = args["Lx"] / Nx, args["Ly"] / Ny, args["Lt"] / Nt

    steps_total = int(round(Lz / args["deltaZ"]))
    if n_windows > steps_total:
        n_windows = max(steps_total, 1)

    fwd = windowed_forward(args, field, n_windows=n_windows, ctx=ctx)

    replicate = fwd["_ctx"]["replicate"]
    checkpoints = fwd["checkpoints"]
    n_ckpts = len(checkpoints) - 1

    save_buf = np.zeros((Nx, Ny, Nt, n_ckpts), dtype=np.complex64)
    for i in range(n_ckpts):
        xyt = _materialize_xyt(jnp.asarray(checkpoints[i + 1]), replicate)
        save_buf[..., i] = np.asarray(xyt).astype(np.complex64)

    dZ = float(args["deltaZ"])
    window_z, z_acc = [], 0.0
    steps_per = steps_total // n_windows
    remainder = steps_total - steps_per * n_windows
    for w in range(n_windows):
        n_here = steps_per + (remainder if w == n_windows - 1 else 0)
        z_acc += n_here * dZ
        window_z.append(z_acc)

    return dict(
        field=save_buf,
        save_at=np.array(window_z),
        dt=dt, dx=dx, dy=dy,
        seconds=fwd["seconds"],
    )

# --------------------------------------------------------------------------------------
# Spectral colour filter factory
# --------------------------------------------------------------------------------------

def make_noise_filter(Nt: int, dt: float, *,
                      bandwidth_hz: float,
                      filter_type: str = 'gaussian',
                      dtype=jnp.float64) -> jnp.ndarray:
    """Build a normalised frequency-domain noise colour filter H(omega), shape (Nt,).

    The filter is applied as  noise_step = IFFT(H * FFT(eps_step))  inside the
    propagation scan.  H is normalised so that mean(|H|^2) = 1, which preserves
    the physical meaning of noise_sigma regardless of filter shape — the same
    noise_sigma value gives the same total injected power per unit propagation
    length whether the noise is white or coloured.

    Parameters
    ----------
    Nt           : number of time samples
    dt           : time step [s]
    bandwidth_hz : -3 dB (half-power) bandwidth [Hz]
        For 'gaussian': sigma_f such that H(f_bw) = 1/sqrt(e) ≈ 0.607
        For 'lorentzian': half-width at half-maximum [Hz]
        For 'rect': half-bandwidth, i.e. H = 1 for |f| <= bandwidth_hz
    filter_type  : 'gaussian' | 'lorentzian' | 'rect'
    dtype        : real dtype for the returned array

    Returns
    -------
    H : jnp.ndarray, shape (Nt,), real, normalised so mean(H^2) = 1
    """
    freq = jnp.fft.fftfreq(Nt, dt)           # Hz, same ordering as FFT output
    bw   = float(bandwidth_hz)

    if filter_type == 'gaussian':
        # H(f) = exp(-f^2 / (2 * bw^2))
        H = jnp.exp(-0.5 * (freq / bw) ** 2)
    elif filter_type == 'lorentzian':
        # H(f) = 1 / (1 + (f/bw)^2)
        H = 1.0 / (1.0 + (freq / bw) ** 2)
    elif filter_type == 'rect':
        # H(f) = 1 if |f| <= bw else 0
        H = jnp.where(jnp.abs(freq) <= bw, 1.0, 0.0)
    else:
        raise ValueError(f"filter_type must be 'gaussian', 'lorentzian', or 'rect'; got {filter_type!r}")

    # Normalise: mean(H^2) = 1  →  noise_sigma retains physical units
    H = H / jnp.sqrt(jnp.mean(H ** 2))
    return H.astype(dtype)


# --------------------------------------------------------------------------------------
# Noise sample factory
# --------------------------------------------------------------------------------------

def make_noise_samples(key, steps_total: int, Nx: int, Ny: int, Nt: int,
                       *, dtype=jnp.complex128) -> jnp.ndarray:
    """Draw (steps_total, Nx, Ny, Nt) white complex Gaussian noise, unit variance.

    Each element satisfies E[|eps|^2] = 1 (real and imag parts each N(0, 1/2)).
    Spectral colouring, if desired, is applied inside the propagation scan via
    noise_filter_w — this function always returns white noise in time.

    Parameters
    ----------
    key  : jax.random.PRNGKey
    dtype: complex dtype (match solver precision; default complex128)

    Returns
    -------
    eps : jnp.ndarray, shape (steps_total, Nx, Ny, Nt), dtype=dtype
    """
    real_dtype = jnp.float32 if dtype == jnp.complex64 else jnp.float64
    scale = real_dtype(1.0 / np.sqrt(2.0))
    k1, k2 = jax.random.split(key)
    shape  = (int(steps_total), int(Nx), int(Ny), int(Nt))
    eps_r  = jax.random.normal(k1, shape, dtype=real_dtype) * scale
    eps_i  = jax.random.normal(k2, shape, dtype=real_dtype) * scale
    return jax.lax.complex(eps_r, eps_i).astype(dtype)


# --------------------------------------------------------------------------------------
# make_args with noise parameters
# --------------------------------------------------------------------------------------

def make_args(
    *args_positional,
    noise_sigma: float = 0.0,
    noise_filter_w=None,
    temporal_abs_t=None,
    **kwargs,
):
    """Extends the base make_args with noise-related parameters.

    noise_sigma    : float, stored for reference; pass explicitly to propagators.
    noise_filter_w : (Nt,) array from make_noise_filter, or None (white noise).
                     Stored for reference only; pass explicitly to propagators.
    temporal_abs_t : (Nt,) real array of per-step absorption coefficients [1/m],
                     or None (no temporal absorption).  Applied each split-step as
                     A_xyt *= exp(-temporal_abs_t * deltaZ).  Set large values at
                     the temporal edges to create absorbing boundary layers.
    """
    base = _make_args_base(*args_positional, **kwargs)
    base["noise_sigma"]     = float(noise_sigma)
    base["noise_filter_w"]  = noise_filter_w   # None or jnp array
    base["temporal_abs_t"]  = temporal_abs_t   # None or (Nt,) real array
    return base


# --------------------------------------------------------------------------------------
# Noisy lean propagator
# --------------------------------------------------------------------------------------

def make_propagate_lean_noisy(shard_t, replicate, *, temporal_abs_mask=None,
                              loss_coeff=0.0):
    """Return a JIT-compiled noisy propagator: (field_kwo, eps, sigma, H) -> field_kwo.

    Parameters (at call time)
    -------------------------
    A0_kwo       : (Nx, Ny, Nt) complex, k-space initial field
    eps          : (steps_total, Nx, Ny, Nt) complex, pre-sampled white noise
    noise_sigma  : scalar; differentiable noise amplitude [same units as A]
    noise_filter_w : (Nt,) real, frequency-domain filter, mean(H^2)=1.
                     Ignored (no extra FFTs) when use_noise_filter=False.
    use_noise_filter : bool (static) — whether to apply spectral colouring

    Closure parameters
    ------------------
    temporal_abs_mask : (1, 1, Nt) real array or None.
        Per-step multiplicative decay mask = exp(-alpha_t * deltaZ).
        Applied each step as A_xyt *= mask, after noise injection.
        None → no temporal absorption (fastest path).
    loss_coeff : float, default 0.0
        Uniform linear loss [1/m] applied to ALL temporal modes each step as
            A_xyt *= exp(-loss_coeff * deltaZ)
        in Itô order (damping BEFORE noise injection) so that the step solves
            dA/dz = F(A) - loss_coeff * A + noise_sigma * xi(z,t)
        Setting loss_coeff = noise_sigma**2 / 2 satisfies the
        fluctuation-dissipation relation: the equilibrium variance per temporal
        mode is |A|**2_eq = 1 (in field amplitude units), giving a stationary
        noise temperature independent of propagation length.
        Default 0.0 → pure additive ASE noise (no compensating loss).

    Noise model per step (Itô order)
    ---------------------------------
        1. split_step (dispersion + Kerr + gain)
        2. A_xyt *= exp(-loss_coeff * deltaZ)        [if loss_coeff > 0]
        3. noise_step = H * FFT(eps_step) -> IFFT    [if use_noise_filter]
                      = eps_step                      [otherwise]
           A_xyt += noise_sigma * sqrt(deltaZ) * noise_step
        4. A_xyt *= temporal_abs_mask                [if temporal_abs_mask is not None]
    """
    _use_temporal_abs = temporal_abs_mask is not None
    _use_loss         = float(loss_coeff) > 0.0
    _loss_coeff       = float(loss_coeff)

    @partial(
        jax.jit,
        static_argnames=(
            'steps_total', 'fr', 'sw', 'use_gain',
            'm_nl_substeps', 'nl_outer_subcycles', 'skip_nl_every',
            'use_noise_filter',
        ),
    )
    def _propagate_lean_noisy(
        A0_kwo: jnp.ndarray,
        eps: jnp.ndarray,             # (steps_total, Nx, Ny, Nt)
        noise_sigma,                  # scalar; gradient flows through this
        noise_filter_w: jnp.ndarray,  # (Nt,) real filter in freq domain
        *,
        use_noise_filter: bool,       # static — removes filter FFTs when False
        steps_total: int,
        dt: float, f0: float,
        fr: float, sw: int,
        deltaZ_linear: float,
        deltaZ_NL: float,
        gamma: float,
        omega_vec: jnp.ndarray,
        D_half: jnp.ndarray,
        PML_half: jnp.ndarray,
        Nprop_half: jnp.ndarray,
        hrw: jnp.ndarray,
        gain_term: jnp.ndarray,
        saturation_intensity: float,
        use_gain: bool,
        m_nl_substeps: int,
        nl_outer_subcycles: int,
        skip_nl_every: int,
    ):
        CD = A0_kwo.dtype
        RD = jnp.float32 if CD == jnp.complex64 else jnp.float64

        # Precompute FDT loss factor once (outside scan, constant across steps).
        # exp(-loss_coeff * deltaZ) — exact exponential is stable for any deltaZ.
        if _use_loss:
            _loss_factor = jnp.exp(RD(-_loss_coeff) * RD(deltaZ_linear))

        field_kwo0 = jax.lax.with_sharding_constraint(A0_kwo, shard_t)

        def one_step(field_kwo, eps_step):
            field_kwo = split_step_sharded(
                field_kwo,
                shard_t=shard_t, replicate=replicate,
                dt=dt, f0=f0, fr=fr, sw=sw,
                deltaZ_linear=deltaZ_linear, deltaZ_NL=deltaZ_NL,
                gamma=gamma, omega_vec=omega_vec,
                D_half=D_half, Nprop_half=Nprop_half, PML_half=PML_half,
                hrw=hrw, gain_term=gain_term,
                saturation_intensity=saturation_intensity,
                use_gain=use_gain,
                m_nl_substeps=m_nl_substeps,
                nl_outer_subcycles=nl_outer_subcycles,
                apply_nl=True,
            )

            # ── materialise to xyt domain ──────────────────────────────────
            field_xyw = jnp.fft.ifftn(field_kwo, axes=(0, 1))
            field_xyt = jnp.fft.ifft(field_xyw, axis=2)

            # ── Itô damping: A *= exp(-loss_coeff * dz), BEFORE noise ──────
            # Correct Itô order for dA/dz = F(A) - loss_coeff*A + sigma*xi.
            # Equilibrium variance: |A|²_eq = sigma² / (2*loss_coeff) per mode.
            # With loss_coeff = sigma²/2 → |A|²_eq = 1 (noise temperature fixed).
            if _use_loss:
                field_xyt = field_xyt * _loss_factor

            # ── reparameterized noise injection ────────────────────────────
            # A_xyt += sigma * sqrt(deltaZ) * [H * eps_step]
            # sqrt(deltaZ) → proper Wiener increment, step-size invariant power.
            eps_cd = eps_step.astype(CD)
            if use_noise_filter:
                # Apply spectral colour filter along the time axis.
                # H shape (Nt,) broadcasts to (Nx, Ny, Nt).
                eps_w      = jnp.fft.fft(eps_cd, axis=2)
                eps_cd     = jnp.fft.ifft(
                    noise_filter_w[None, None, :].astype(CD) * eps_w, axis=2
                )

            field_xyt = field_xyt + RD(noise_sigma) * jnp.sqrt(RD(deltaZ_linear)) * eps_cd

            # ── temporal absorbing boundary (optional) ─────────────────────
            # Damps field at temporal edges each step.  temporal_abs_mask is
            # exp(-alpha_t * deltaZ) with alpha_t large at the time-window edges.
            if _use_temporal_abs:
                field_xyt = field_xyt * temporal_abs_mask

            field_xyw = jnp.fft.fft(field_xyt, axis=2)
            field_kwo = jnp.fft.fftn(field_xyw, axes=(0, 1))

            return field_kwo, None

        ckpt_step = jax.checkpoint(one_step, prevent_cse=False)
        field_end_kwo, _ = jax.lax.scan(
            ckpt_step, field_kwo0, xs=eps, length=int(steps_total)
        )
        return field_end_kwo

    return _propagate_lean_noisy


# --------------------------------------------------------------------------------------
# Internal helper: resolve noise_filter_w and use_noise_filter flag
# --------------------------------------------------------------------------------------

def _resolve_filter(noise_filter_w, Nt, CD):
    """Return (filter_array, use_noise_filter_bool) ready to pass to the JIT fn."""
    if noise_filter_w is None:
        # Return a dummy scalar; use_noise_filter=False means it is never read.
        return jnp.ones(Nt, dtype=jnp.float64), False
    else:
        real_dtype = jnp.float32 if CD == jnp.complex64 else jnp.float64
        return jnp.asarray(noise_filter_w, dtype=real_dtype), True


# --------------------------------------------------------------------------------------
# Windowed context for noisy propagation
# --------------------------------------------------------------------------------------

def make_windowed_context_noisy(args, Nt: int, *, precision: str | None = None,
                                loss_coeff: float = 0.0):
    """Like make_windowed_context but returns the noisy lean propagator."""
    prec = precision or args.get("precision", "fp64")
    prep = _prepare_propagation(args, np.zeros((1, 1, Nt)), precision=prec)
    _, shard_t, replicate = _make_mesh_for_time_axis(Nt)

    prop_lean_noisy = make_propagate_lean_noisy(shard_t, replicate,
                                                loss_coeff=loss_coeff)
    lean_kw = _build_lean_kw(prep, args)

    return dict(
        prop_lean=prop_lean_noisy,
        lean_kw=lean_kw,
        prep=prep,
        CD=_resolve_precision(prec)[1],
        shard_t=shard_t,
        replicate=replicate,
    )


# --------------------------------------------------------------------------------------
# GNLSE3D_propagate_noisy — snapshot-collecting forward with noise
# --------------------------------------------------------------------------------------

def GNLSE3D_propagate_noisy(
    args, A0,
    eps,
    noise_sigma: float = 0.0,
    *,
    loss_coeff: float = 0.0,
    noise_filter_w=None,
    precision: str | None = None,
    save_as_fp32: bool = True,
):
    """Forward propagation with per-step reparameterized Gaussian noise.

    Parameters
    ----------
    eps : array (steps_total, Nx, Ny, Nt) or None
        Pre-sampled unit-Gaussian white noise from make_noise_samples().
        If None (or noise_sigma == 0), noiseless propagation is performed.
    noise_sigma : float
        Noise amplitude [sqrt(W/m^2) for physical fields].  Combined with the
        internal sqrt(deltaZ) factor this is the Langevin diffusion coefficient
        [sqrt(W/m^2 / m)], i.e. the RMS noise per sqrt(meter) of propagation.
    noise_filter_w : (Nt,) array or None
        Frequency-domain colour filter from make_noise_filter().
        None → white noise (no extra FFTs per step).

    Returns
    -------
    dict matching GNLSE3D_propagate output:
        field, dt, dx, seconds, steps_executed, stopped_early, z_event, n_saved
    """
    prec = precision or args.get("precision", "fp64")
    RD, CD, _ = _resolve_precision(prec)
    prep = _prepare_propagation(args, A0, precision=prec)
    CD_save = jnp.complex64 if (save_as_fp32 and CD == jnp.complex128) else CD

    A0_kwo = jnp.fft.fftn(jnp.asarray(A0, dtype=CD), axes=(0, 1, 2))
    Nx, Ny, Nt = A0_kwo.shape
    _, shard_t, replicate = _make_mesh_for_time_axis(Nt)

    steps_total = prep["steps_total"]

    if eps is None or float(noise_sigma) == 0.0:
        eps_jnp     = jnp.zeros((steps_total, Nx, Ny, Nt), dtype=CD)
        noise_sigma = 0.0
    else:
        eps_jnp = jnp.asarray(eps, dtype=CD)
        assert eps_jnp.shape == (steps_total, Nx, Ny, Nt), (
            f"eps must have shape ({steps_total}, {Nx}, {Ny}, {Nt}), got {eps_jnp.shape}"
        )

    filter_arr, use_filter = _resolve_filter(noise_filter_w, Nt, CD)

    # Build temporal absorption mask (closed over in the JIT propagator).
    _tab = args.get("temporal_abs_t", None)
    if _tab is not None:
        _dz    = float(prep["deltaZ_linear"])
        _mask  = jnp.exp(-jnp.asarray(_tab, dtype=RD)[None, None, :] * RD(_dz))
    else:
        _mask = None
    prop_noisy = make_propagate_lean_noisy(shard_t, replicate,
                                           temporal_abs_mask=_mask,
                                           loss_coeff=loss_coeff)
    lean_kw    = _build_lean_kw(prep, args)

    save_idx = np.asarray(prep["save_idx"])
    save_n   = int(prep["save_n"])
    save_buf = np.zeros((Nx, Ny, Nt, save_n),
                        dtype=np.complex64 if save_as_fp32 else np.complex128)

    t0        = time.time()
    field_kwo = A0_kwo
    steps_done = 0

    for si in range(save_n):
        target_done   = int(save_idx[si]) + 1
        n_steps_here  = target_done - steps_done
        if n_steps_here > 0:
            eps_slice = eps_jnp[steps_done:target_done]
            field_kwo = prop_noisy(
                field_kwo, eps_slice, noise_sigma, filter_arr,
                use_noise_filter=use_filter, **lean_kw, steps_total=n_steps_here,
            )
        snap = _materialize_xyt(field_kwo, replicate).astype(CD_save)
        save_buf[..., si] = np.asarray(jax.device_get(snap))
        steps_done = target_done

    remaining = steps_total - steps_done
    if remaining > 0:
        eps_slice = eps_jnp[steps_done:]
        field_kwo = prop_noisy(
            field_kwo, eps_slice, noise_sigma, filter_arr,
            use_noise_filter=use_filter, **lean_kw, steps_total=remaining,
        )

    elapsed = time.time() - t0
    meta = dict(
        steps_executed=jnp.array(steps_total, dtype=jnp.int32),
        stopped_early=jnp.array(False),
        z_event=jnp.array(jnp.nan),
        n_saved=jnp.array(save_n, dtype=jnp.int32),
    )
    return dict(field=jnp.asarray(save_buf), dt=prep["dt"], dx=prep["dx"],
                seconds=elapsed, **meta)


# --------------------------------------------------------------------------------------
# Windowed forward / grad with noise
# --------------------------------------------------------------------------------------

def windowed_forward_noisy(
    args, A0,
    eps,
    noise_sigma: float = 0.0,
    *,
    loss_coeff: float = 0.0,
    noise_filter_w=None,
    n_windows: int = 10,
    precision: str | None = None,
    save_as_fp32: bool = True,
    ctx: dict | None = None,
):
    """Windowed forward pass with reparameterized noise and optional colour filter.

    Parameters
    ----------
    eps            : (steps_total, Nx, Ny, Nt) or None
    noise_sigma    : float, differentiable noise amplitude
    noise_filter_w : (Nt,) array or None — colour filter from make_noise_filter()

    Returns dict with 'field_final', 'checkpoints', 'seconds', '_ctx'.
    """
    prec = precision or args.get("precision", "fp64")
    RD, CD, _ = _resolve_precision(prec)
    CD_save = jnp.complex64 if (save_as_fp32 and CD == jnp.complex128) else CD

    A0_kwo = jnp.fft.fftn(jnp.asarray(A0, dtype=CD), axes=(0, 1, 2))
    Nx, Ny, Nt = A0_kwo.shape

    if ctx is None:
        ctx = make_windowed_context_noisy(args, Nt, precision=prec,
                                          loss_coeff=loss_coeff)

    prop_lean = ctx["prop_lean"]
    lean_kw   = ctx["lean_kw"]
    prep      = ctx["prep"]
    replicate = ctx["replicate"]

    steps_total  = prep["steps_total"]
    window_steps = _uniform_window_steps(steps_total, n_windows)

    if eps is None or float(noise_sigma) == 0.0:
        eps_jnp     = jnp.zeros((steps_total, Nx, Ny, Nt), dtype=CD)
        noise_sigma = 0.0
    else:
        eps_jnp = jnp.asarray(eps, dtype=CD)

    filter_arr, use_filter = _resolve_filter(noise_filter_w, Nt, CD)

    t0 = time.time()
    checkpoints_host = [np.asarray(jax.device_get(A0_kwo))]
    field_kwo  = A0_kwo
    step_start = 0

    for w in range(n_windows):
        nw = window_steps[w]
        if nw == 0:
            checkpoints_host.append(checkpoints_host[-1].copy())
            continue
        eps_w = eps_jnp[step_start:step_start + nw]
        field_kwo = prop_lean(
            field_kwo, eps_w, noise_sigma, filter_arr,
            use_noise_filter=use_filter, **lean_kw, steps_total=nw,
        )
        field_kwo.block_until_ready()
        checkpoints_host.append(np.asarray(jax.device_get(field_kwo)))
        step_start += nw

    elapsed = time.time() - t0
    field_final_xyt = _materialize_xyt(field_kwo, replicate).astype(CD_save)

    return dict(
        field_final=field_final_xyt,
        checkpoints=checkpoints_host,
        n_windows=n_windows,
        steps_per_window=window_steps[0],
        steps_total=steps_total,
        seconds=elapsed,
        dt=prep["dt"], dx=prep["dx"],
        _ctx=ctx,
    )


def windowed_grad_noisy(
    loss_fn,
    args, A0,
    eps,
    noise_sigma: float = 0.0,
    *,
    loss_coeff: float = 0.0,
    noise_filter_w=None,
    n_windows: int = 10,
    precision: str | None = None,
    ctx: dict | None = None,
):
    """Memory-efficient dL/dA0 via per-window VJP, with noise and optional colour filter.

    Parameters
    ----------
    loss_fn        : field_kwo_final -> scalar
    eps            : (steps_total, Nx, Ny, Nt) or None
    noise_sigma    : float (treated as constant; not differentiated here)
    noise_filter_w : (Nt,) array or None

    Returns
    -------
    dict: loss, grad (kwo-space, same shape as A0_kwo), fwd_seconds, bwd_seconds
    """
    prec = precision or args.get("precision", "fp64")
    RD, CD, _ = _resolve_precision(prec)
    A0_kwo = jnp.fft.fftn(jnp.asarray(A0, dtype=CD), axes=(0, 1, 2))
    Nx, Ny, Nt = A0_kwo.shape

    if ctx is None:
        ctx = make_windowed_context_noisy(args, Nt, precision=prec,
                                          loss_coeff=loss_coeff)

    prop_lean = ctx["prop_lean"]
    lean_kw   = ctx["lean_kw"]
    prep      = ctx["prep"]

    steps_total  = prep["steps_total"]
    window_steps = _uniform_window_steps(steps_total, n_windows)

    if eps is None or float(noise_sigma) == 0.0:
        eps_jnp     = jnp.zeros((steps_total, Nx, Ny, Nt), dtype=CD)
        noise_sigma = 0.0
    else:
        eps_jnp = jnp.asarray(eps, dtype=CD)

    filter_arr, use_filter = _resolve_filter(noise_filter_w, Nt, CD)

    # Slice eps per window and cache on host
    step_start = 0
    eps_slices_host = []
    for w in range(n_windows):
        nw = window_steps[w]
        eps_slices_host.append(np.asarray(eps_jnp[step_start:step_start + nw]))
        step_start += nw

    # ── Forward pass: store checkpoints on host ───────────────────────────
    t_fwd = time.time()
    checkpoints_host = [np.asarray(jax.device_get(A0_kwo))]
    field_kwo = A0_kwo

    for w in range(n_windows):
        nw = window_steps[w]
        if nw == 0:
            checkpoints_host.append(checkpoints_host[-1].copy())
            continue
        eps_w = jnp.asarray(eps_slices_host[w])
        field_kwo = prop_lean(
            field_kwo, eps_w, noise_sigma, filter_arr,
            use_noise_filter=use_filter, **lean_kw, steps_total=nw,
        )
        field_kwo.block_until_ready()
        checkpoints_host.append(np.asarray(jax.device_get(field_kwo)))

    fwd_seconds = time.time() - t_fwd

    # ── Loss at final field ────────────────────────────────────────────────
    field_final = jnp.asarray(checkpoints_host[-1])
    loss_val, grad_final = jax.value_and_grad(loss_fn)(field_final)

    # ── Backward: per-window VJP in reverse ───────────────────────────────
    t_bwd     = time.time()
    grad_carry = grad_final

    for w in reversed(range(n_windows)):
        nw = window_steps[w]
        if nw == 0:
            continue
        A_start = jnp.asarray(checkpoints_host[w])
        eps_w   = jnp.asarray(eps_slices_host[w])

        def _wp(a_start, _eps=eps_w, _nw=nw):
            return prop_lean(
                a_start, _eps, noise_sigma, filter_arr,
                use_noise_filter=use_filter, **lean_kw, steps_total=_nw,
            )

        _, vjp_fn = jax.vjp(_wp, A_start)
        (grad_carry,) = vjp_fn(grad_carry)
        grad_carry.block_until_ready()

    bwd_seconds = time.time() - t_bwd

    return dict(
        loss=loss_val,
        grad=grad_carry,
        fwd_seconds=fwd_seconds,
        bwd_seconds=bwd_seconds,
    )
