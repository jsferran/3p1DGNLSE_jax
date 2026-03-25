# ======================================================================================
# PROTOTYPE: gnlse_visualizations_prototype.py
# This file contains fixes for dimensional flexibility (Nx=1, Ny=1, or Nt=1 cases).
# Changes from original:
#   - Safe dx/dy inference in power_vs_time_from_results() for singleton arrays
#   - Mode plotting functions for 2D, 1D, and gallery views
# ======================================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# --------------------------------------------------------------------------------------
# Helper for safe grid spacing calculation from coordinate arrays
# --------------------------------------------------------------------------------------
def _safe_grid_spacing(coord_array, fallback=1.0):
    """
    Compute grid spacing from a coordinate array.
    For singleton arrays (len=1), returns the fallback value.
    For arrays with 2+ elements, returns coord[1] - coord[0].
    """
    coord = np.asarray(coord_array)
    if coord.size >= 2:
        return float(coord[1] - coord[0])
    else:
        # Singleton array: use fallback
        return float(fallback)


def make_xy_t_animation(field4d,
                        z_index=-1,
                        x=None, y=None, t=None,
                        quantity='intensity',      # 'intensity' (|E|^2), 'abs', 'real', 'imag', 'phase'
                        norm='global',             # 'global' or 'per_frame'
                        t_window=None,             # (t_start, t_end) in same units as t (ignored if t is None)
                        frame_window=None,         # (i_start, i_end) inclusive/exclusive frame indices
                        stride=1,                  # take every 'stride' frame in the selected window
                        fps=30,
                        filename='xy_t.gif',
                        dpi=120):
    """
    Animate the transverse (x,y) field vs time at a fixed z, optionally over a narrower time range.

    Parameters
    ----------
    field4d : np.ndarray, complex, shape (Nx, Ny, Nt, Nz)
        The complex field.
    z_index : int
        Which z slice to animate (default last).
    x, y : 1D arrays or None
        Spatial axes (used only for labeling/extent). If None, pixel indices are used.
    t : 1D array or None
        Time axis. Required to use t_window. If None, you can use frame_window instead.
    quantity : str
        'intensity', 'abs', 'real', 'imag', or 'phase'.
    norm : str
        'global' -> single color scale over displayed frames;
        'per_frame' -> auto-scale each frame independently.
    t_window : tuple or None
        (t_start, t_end) limits the animation to this physical time window. Requires `t`.
        Both endpoints are inclusive of the nearest sample.
    frame_window : tuple or None
        (i_start, i_end) frame indices. i_end follows Python slicing (exclusive).
        Ignored if t_window is provided.
    stride : int
        Use every `stride`-th frame within the selected window (for decimating long sequences).
    fps : int
        Frames per second for the GIF.
    filename : str
        Output GIF path.
    dpi : int
        Figure DPI for saving.

    Returns
    -------
    filename : str
        Path to the saved GIF.
    """

    assert field4d.ndim == 4, "Expected (Nx, Ny, Nt, Nz)."
    Nx, Ny, Nt, Nz = field4d.shape
    if not (-Nz <= z_index < Nz):
        raise IndexError(f"z_index {z_index} out of range for Nz={Nz}")

    # Extract the (x,y,t) block at fixed z
    F = field4d[..., z_index]  # (Nx, Ny, Nt)

    # Map to requested quantity
    if quantity == 'intensity':
        data_t = np.abs(F)**2
        cbar_label = r'|E|$^2$'
    elif quantity == 'abs':
        data_t = np.abs(F)
        cbar_label = r'|E|'
    elif quantity == 'real':
        data_t = np.real(F)
        cbar_label = 'Re{E}'
    elif quantity == 'imag':
        data_t = np.imag(F)
        cbar_label = 'Im{E}'
    elif quantity == 'phase':
        data_t = np.angle(F)
        cbar_label = 'arg(E)'
    else:
        raise ValueError("quantity must be one of: 'intensity', 'abs', 'real', 'imag', 'phase'")

    # Determine the temporal/frame subset
    if t_window is not None:
        if t is None:
            raise ValueError("t_window was provided but `t` axis is None.")
        t = np.asarray(t)
        if len(t) != Nt:
            raise ValueError("Length of `t` must match Nt dimension of field.")
        t_start, t_end = t_window
        if t_start > t_end:
            t_start, t_end = t_end, t_start
        # nearest-sample indices spanning [t_start, t_end]
        i0 = int(np.clip(np.searchsorted(t, t_start, side='left'), 0, Nt-1))
        i1 = int(np.clip(np.searchsorted(t, t_end,   side='right'), 0, Nt))  # exclusive
    elif frame_window is not None:
        i0, i1 = frame_window
        i0 = int(np.clip(i0, 0, Nt))
        i1 = int(np.clip(i1, i0+1, Nt))  # ensure at least one frame if possible
    else:
        i0, i1 = 0, Nt

    # Apply stride
    frame_indices = np.arange(i0, i1, stride, dtype=int)
    if frame_indices.size == 0:
        raise ValueError("Selected time/frame window is empty after applying stride.")

    data_sel = data_t[..., frame_indices]  # (Nx, Ny, Nf)
    t_sel = t[frame_indices] if (t is not None and len(t) == Nt) else None
    Nf = data_sel.shape[-1]

    # Axes extents for imshow
    if x is not None and y is not None:
        x = np.asarray(x); y = np.asarray(y)
        extent = [x.min(), x.max(), y.min(), y.max()]
        xlabel, ylabel = 'x', 'y'
    else:
        extent = None
        xlabel, ylabel = 'pixel x', 'pixel y'

    # Normalization over the **displayed** frames
    if norm == 'global':
        vmin = np.nanmin(data_sel)
        vmax = np.nanmax(data_sel)
        if vmax == vmin:
            vmax = vmin + (1e-12 if np.isfinite(vmin) else 1.0)
    elif norm == 'per_frame':
        vmin = vmax = None
    else:
        raise ValueError("norm must be 'global' or 'per_frame'")

    # Prepare figure
    fig, ax = plt.subplots()
    im = ax.imshow(data_sel[..., 0].T, origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if t_sel is not None:
        title = ax.set_title(f"z index = {z_index}, t = {t_sel[0]}")
    else:
        title = ax.set_title(f"z index = {z_index}, frame = {frame_indices[0]}")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    # Frame update
    def update(k):
        frame_data = data_sel[..., k]
        if norm == 'per_frame':
            im.set_clim(np.nanmin(frame_data), np.nanmax(frame_data))
        im.set_data(frame_data.T)
        if t_sel is not None:
            title.set_text(f"z index = {z_index}, t = {t_sel[k]}")
        else:
            title.set_text(f"z index = {z_index}, frame = {frame_indices[k]}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=Nf, interval=1000.0/fps, blit=False)

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)
    return filename

import numpy as np

def power_vs_time_from_results(results, z_index=-1, dx=None, dy=None):
    """
    Compute temporal power P(t) at the fiber output (or any z_index).

    Parameters
    ----------
    results : dict
        Must contain 'fields' or 'field' with shape (Nx, Ny, Nt, Nz).
        If dx, dy are not provided, results should also contain 1D 'x' and 'y'.
    z_index : int
        Which z slice to use; default -1 (output).
    dx, dy : float or None
        Spatial sample spacings. If None, inferred from results['x'], results['y'].

    Returns
    -------
    P_t : np.ndarray, shape (Nt,)
        Temporal power at the chosen z, i.e. sum_{x,y} |E(x,y,t,z)|^2 * dx * dy.
    """
    # get field
    field4d = results.get("fields", results.get("field", None))
    if field4d is None:
        raise KeyError("results must contain 'fields' or 'field' with shape (Nx, Ny, Nt, Nz).")
    if field4d.ndim != 4:
        raise ValueError(f"Expected field shape (Nx, Ny, Nt, Nz); got {field4d.shape}.")

    Nx, Ny, Nt, Nz = field4d.shape
    if not (-Nz <= z_index < Nz):
        raise IndexError(f"z_index {z_index} out of range for Nz={Nz}.")

    # spacings
    if dx is None or dy is None:
        x = results.get("x", None)
        y = results.get("y", None)
        if x is None or y is None:
            # Try to get dx/dy directly from results if available
            if "dx" in results and "dy" in results:
                dx = float(results["dx"]) if dx is None else dx
                dy = float(results["dy"]) if dy is None else dy
            else:
                raise ValueError("Provide dx, dy, or include 1D arrays 'x' and 'y' (or 'dx'/'dy') in results.")
        else:
            x = np.asarray(x); y = np.asarray(y)
            if x.ndim != 1 or y.ndim != 1:
                raise ValueError("results['x'] and results['y'] must be 1D arrays.")
            # Safe dx/dy calculation for singleton arrays
            dx = _safe_grid_spacing(x, fallback=1.0) if dx is None else dx
            dy = _safe_grid_spacing(y, fallback=1.0) if dy is None else dy

    # slice at z and integrate over x,y
    F = field4d[..., z_index]                # (Nx, Ny, Nt)
    P_t = np.sum(np.abs(F)**2, axis=(0, 1))  # integrate over x,y (no spacings yet)
    P_t = P_t * dx * dy                      # apply area element

    # ensure real (tiny imag can appear from numerics)
    return np.real_if_close(P_t, tol=1000)



def make_xy_z_animation(field4d,
                        t_index=-1,
                        x=None, y=None, z=None,
                        quantity='intensity',      # 'intensity' (|E|^2), 'abs', 'real', 'imag', 'phase'
                        norm='global',             # 'global' or 'per_frame'
                        z_window=None,             # (z_start, z_end) in same units as z (ignored if z is None)
                        frame_window=None,         # (i_start, i_end) inclusive/exclusive frame indices
                        stride=1,                  # take every 'stride' frame in the selected window
                        fps=30,
                        filename='xy_z.gif',
                        dpi=120):
    """
    Animate the transverse (x,y) field vs propagation distance z at a fixed time index t.

    Parameters
    ----------
    field4d : np.ndarray, complex, shape (Nx, Ny, Nt, Nz)
        The complex field.
    t_index : int
        Which time slice to animate (default last).
    x, y : 1D arrays or None
        Spatial axes (used only for labeling/extent). If None, pixel indices are used.
    z : 1D array or None
        z axis. Required to use z_window. If None, you can use frame_window instead.
    quantity : str
        'intensity', 'abs', 'real', 'imag', or 'phase'.
    norm : str
        'global' -> single color scale over displayed frames;
        'per_frame' -> auto-scale each frame independently.
    z_window : tuple or None
        (z_start, z_end) limits the animation to this physical z window. Requires `z`.
        Both endpoints are inclusive of the nearest sample.
    frame_window : tuple or None
        (i_start, i_end) frame indices along z. i_end follows Python slicing (exclusive).
        Ignored if z_window is provided.
    stride : int
        Use every `stride`-th frame within the selected window (for decimating long sequences).
    fps : int
        Frames per second for the GIF.
    filename : str
        Output GIF path.
    dpi : int
        Figure DPI for saving.

    Returns
    -------
    filename : str
        Path to the saved GIF.
    """

    assert field4d.ndim == 4, "Expected (Nx, Ny, Nt, Nz)."
    Nx, Ny, Nt, Nz = field4d.shape
    if not (-Nt <= t_index < Nt):
        raise IndexError(f"t_index {t_index} out of range for Nt={Nt}")

    # Extract the (x,y,z) block at fixed t
    F = field4d[:, :, t_index, :]  # (Nx, Ny, Nz)

    # Map to requested quantity
    if quantity == 'intensity':
        data_z = np.abs(F)**2
        cbar_label = r'|E|$^2$'
    elif quantity == 'abs':
        data_z = np.abs(F)
        cbar_label = r'|E|'
    elif quantity == 'real':
        data_z = np.real(F)
        cbar_label = 'Re{E}'
    elif quantity == 'imag':
        data_z = np.imag(F)
        cbar_label = 'Im{E}'
    elif quantity == 'phase':
        data_z = np.angle(F)
        cbar_label = 'arg(E)'
    else:
        raise ValueError("quantity must be one of: 'intensity', 'abs', 'real', 'imag', 'phase'")

    # Determine the z/frame subset
    if z_window is not None:
        if z is None:
            raise ValueError("z_window was provided but `z` axis is None.")
        z = np.asarray(z)
        if len(z) != Nz:
            raise ValueError("Length of `z` must match Nz dimension of field.")
        z_start, z_end = z_window
        if z_start > z_end:
            z_start, z_end = z_end, z_start
        i0 = int(np.clip(np.searchsorted(z, z_start, side='left'), 0, Nz-1))
        i1 = int(np.clip(np.searchsorted(z, z_end,   side='right'), 0, Nz))  # exclusive
    elif frame_window is not None:
        i0, i1 = frame_window
        i0 = int(np.clip(i0, 0, Nz))
        i1 = int(np.clip(i1, i0+1, Nz))  # ensure at least one frame if possible
    else:
        i0, i1 = 0, Nz

    # Apply stride
    frame_indices = np.arange(i0, i1, stride, dtype=int)
    if frame_indices.size == 0:
        raise ValueError("Selected z/frame window is empty after applying stride.")

    data_sel = data_z[..., frame_indices]              # (Nx, Ny, Nf)
    z_sel = z[frame_indices] if (z is not None and len(z) == Nz) else None
    Nf = data_sel.shape[-1]

    # Axes extents for imshow
    if x is not None and y is not None:
        x = np.asarray(x); y = np.asarray(y)
        extent = [x.min(), x.max(), y.min(), y.max()]
        xlabel, ylabel = 'x', 'y'
    else:
        extent = None
        xlabel, ylabel = 'pixel x', 'pixel y'

    # Normalization over the displayed frames
    if norm == 'global':
        vmin = np.nanmin(data_sel)
        vmax = np.nanmax(data_sel)
        if vmax == vmin:
            vmax = vmin + (1e-12 if np.isfinite(vmin) else 1.0)
    elif norm == 'per_frame':
        vmin = vmax = None
    else:
        raise ValueError("norm must be 'global' or 'per_frame'")

    # Prepare figure
    fig, ax = plt.subplots()
    im = ax.imshow(data_sel[..., 0].T, origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if z_sel is not None:
        title = ax.set_title(f"t index = {t_index}, z = {z_sel[0]}")
    else:
        title = ax.set_title(f"t index = {t_index}, frame = {frame_indices[0]}")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    # Frame update
    def update(k):
        frame_data = data_sel[..., k]
        if norm == 'per_frame':
            im.set_clim(np.nanmin(frame_data), np.nanmax(frame_data))
        im.set_data(frame_data.T)
        if z_sel is not None:
            title.set_text(f"t index = {t_index}, z = {z_sel[k]}")
        else:
            title.set_text(f"t index = {t_index}, frame = {frame_indices[k]}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=Nf, interval=1000.0/fps, blit=False)

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)
    return filename


def make_transverse_vs_z_vs_I_plot(
    out: dict,
    args: dict | None = None,
    *,
    axis: str = "x",                 # "x" → vertical is x; "y" → vertical is y
    mode: str = "time_integrated",   # "time_integrated" | "single_t"
    t_select: float | int | None = None,  # for single_t: time [s] or index; None → center frame
    reduce: str = "centerline",      # "centerline" | "aperture" | "sum" | "mean"
    aperture_px: int = 3,            # half-width (pixels) for reduce="aperture"
    log10: bool = False,             # log color scale (log10(I + eps))
    normalize: bool = False,         # normalize map by global max before optional log
    cmap: str | plt.Colormap = "jet",# default colormap
    figsize: tuple = (8, 4),
    clim: tuple | None = None,       # (vmin, vmax) after transform
    title: str | None = None,
    # ---- NEW windowing options (meters) ----
    z_window: tuple[float, float] | None = None,     # (z_min, z_max) in meters
    axis_window: tuple[float, float] | None = None,  # (x_min, x_max) if axis='x'; (y_min, y_max) if axis='y'
):
    """
    Bird's-eye view with z on the horizontal axis and x/y on the vertical.
    Color shows intensity after the requested reductions.

    Windowing:
      - z_window=(zmin,zmax) in meters limits the horizontal range (defaults to full saved range).
      - axis_window=(amin,amax) in meters limits the vertical range along the chosen axis.
    """
    # --- fetch and normalize field shape ---
    F = np.asarray(out["field"])
    if F.ndim == 3:  # (Nx, Ny, Nt) → add single-z dimension
        F = F[..., None]
    Nx, Ny, Nt, Nsave = F.shape

    dx = float(out["dx"])
    dy = float(out.get("dy", out["dx"]))
    dt = float(out.get("dt", 1.0))

    # --- z-axis (meters) ---
    if "save_at" in out:
        z = np.asarray(out["save_at"], float)
    elif args is not None and ("save_at" in args):
        z = np.asarray(args["save_at"], float)
        if z.ndim == 0:
            z = np.array([float(z)])
    else:
        z = np.arange(Nsave, dtype=float)
    if z.size != Nsave:
        z = np.linspace(0.0, float(Nsave - 1), Nsave)

    # --- time selection / integration ---
    if mode not in ("time_integrated", "single_t"):
        raise ValueError("mode must be 'time_integrated' or 'single_t'.")

    if mode == "single_t":
        if isinstance(t_select, int):
            ti = int(np.clip(t_select, 0, Nt-1))
        elif isinstance(t_select, float):
            if args is not None and ("Lt" in args):
                Lt = float(args["Lt"])
                tgrid = np.linspace(-Lt/2, Lt/2, Nt)
            else:
                tgrid = dt * (np.arange(Nt) - (Nt-1)/2.0)
            ti = int(np.argmin(np.abs(tgrid - float(t_select))))
        else:
            ti = Nt // 2
    else:
        ti = None  # integrate

    # --- coordinates for vertical axis (x or y), meters ---
    if args is not None and ("Lx" in args and "Ly" in args):
        Lx = float(args["Lx"]); Ly = float(args["Ly"])
        x_axis = np.linspace(-Lx/2, Lx/2, Nx)
        y_axis = np.linspace(-Ly/2, Ly/2, Ny)
    else:
        x_axis = np.arange(Nx) * dx
        y_axis = np.arange(Ny) * dy

    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'.")
    use_x = (axis == "x")
    v_axis = x_axis if use_x else y_axis  # vertical coordinate array (meters)

    # --- build index windows (z and vertical axis) ---
    # z (horizontal)
    if z_window is not None:
        zmin, zmax = float(min(*z_window)), float(max(*z_window))
        z_idx = np.where((z >= zmin) & (z <= zmax))[0]
        if z_idx.size == 0:
            # fall back to nearest-in-bounds indices
            z_idx = np.array([np.argmin(np.abs(z - zmin)), np.argmin(np.abs(z - zmax))])
            z_idx.sort()
    else:
        z_idx = np.arange(Nsave, dtype=int)
    Z_sel = z[z_idx]

    # vertical axis (x or y)
    if axis_window is not None:
        amin, amax = float(min(*axis_window)), float(max(*axis_window))
        v_mask = (v_axis >= amin) & (v_axis <= amax)
        if not np.any(v_mask):
            # nearest pair
            jlo = int(np.argmin(np.abs(v_axis - amin)))
            jhi = int(np.argmin(np.abs(v_axis - amax)))
            if jlo > jhi: jlo, jhi = jhi, jlo
            jhi = min(jhi+1, v_axis.size)
        else:
            j_inds = np.where(v_mask)[0]
            jlo, jhi = int(j_inds.min()), int(j_inds.max()+1)
    else:
        jlo, jhi = 0, v_axis.size
    V_sel = v_axis[jlo:jhi]

    # --- center indices for reductions over the OTHER transverse axis ---
    cx, cy = Nx // 2, Ny // 2

    # --- map to fill: rows = V_sel, cols = Z_sel ---
    I_map = np.zeros((V_sel.size, Z_sel.size), dtype=np.float64)

    # --- main loop over selected z frames only ---
    for col, iz in enumerate(z_idx):
        Fz = F[..., iz]  # (Nx,Ny,Nt)

        # time intensity
        if mode == "time_integrated":
            Ixy = np.sum(np.abs(Fz) ** 2, axis=2) * dt
        else:
            Ixy = np.abs(Fz[:, :, ti]) ** 2

        # reduce over the orthogonal transverse axis, then slice to window
        if use_x:
            if reduce == "centerline":
                line = Ixy[:, cy]
            elif reduce == "aperture":
                lo = max(0, cy - aperture_px); hi = min(Ny, cy + aperture_px + 1)
                line = Ixy[:, lo:hi].mean(axis=1)
            elif reduce == "sum":
                line = Ixy.sum(axis=1) * dy
            elif reduce == "mean":
                line = Ixy.mean(axis=1)
            else:
                raise ValueError("reduce must be one of: 'centerline','aperture','sum','mean'.")
            I_map[:, col] = line[jlo:jhi]
        else:
            if reduce == "centerline":
                line = Ixy[cx, :]
            elif reduce == "aperture":
                lo = max(0, cx - aperture_px); hi = min(Nx, cx + aperture_px + 1)
                line = Ixy[lo:hi, :].mean(axis=0)
            elif reduce == "sum":
                line = Ixy.sum(axis=0) * dx
            elif reduce == "mean":
                line = Ixy.mean(axis=0)
            else:
                raise ValueError("reduce must be one of: 'centerline','aperture','sum','mean'.")
            I_map[:, col] = line[jlo:jhi]

    # --- optional normalization & log scaling ---
    eps = 1e-18 * (np.nanmax(I_map) if I_map.size else 1.0)
    if normalize and I_map.max() > 0:
        I_map = I_map / (I_map.max() + 1e-30)

    data_to_show = np.log10(I_map + eps) if log10 else I_map
    vmin, vmax = (None, None) if clim is None else clim

    # --- colormap ---
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    # --- plot with z horizontal, x/y vertical, limited to windows ---
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    extent = [Z_sel.min(), Z_sel.max(), V_sel.min(), V_sel.max()]
    im = ax.imshow(
        data_to_show,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if log10:
        cbar.set_label(r"$\log_{10}(\text{intensity})$ (arb.)")
    else:
        if mode == "time_integrated":
            cbar.set_label(r"$\int |E(x,y,t)|^2\,dt$ (arb.)")
        else:
            cbar.set_label(r"$|E(x,y,t)|^2$ (arb.)")

    ax.set_xlabel("z (m)")
    ax.set_ylabel(f"{axis} (m)")

    if title is None:
        red = f"{reduce}" if reduce != "aperture" else f"aperture ±{aperture_px}px"
        tim = "∫ dt" if mode == "time_integrated" else (f"t index {ti}" if isinstance(t_select, int) else "single t")
        lg  = " (log)" if log10 else ""
        win_z = "" if z_window is None else f" | z∈[{Z_sel.min():.3g},{Z_sel.max():.3g}]"
        win_a = "" if axis_window is None else f" | {axis}∈[{V_sel.min():.3g},{V_sel.max():.3g}]"
        ax.set_title(f"z vs {axis.upper()} — {red}, {tim}{lg}{win_z}{win_a}")
    else:
        ax.set_title(title)

    return fig, ax, V_sel, Z_sel, I_map


# ======================================================================================
# MODE LOADING AND PLOTTING FUNCTIONS
# ======================================================================================

from pathlib import Path
import glob


def list_modes_in_folder(folder, heading="mode", file_format="npz"):
    """
    List all mode files in a folder and return their indices.

    Parameters
    ----------
    folder : str or Path
        Path to the folder containing mode files.
    heading : str
        Prefix of mode files (e.g., "mode" for "mode_0000.npz").
    file_format : str
        File extension (default "npz").

    Returns
    -------
    indices : list of int
        Sorted list of mode indices found in the folder.
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    pattern = f"{heading}_*.{file_format}"
    files = list(folder.glob(pattern))

    indices = []
    for f in files:
        # Extract index from filename like "mode_0003.npz"
        stem = f.stem  # "mode_0003"
        try:
            idx_str = stem.replace(f"{heading}_", "")
            indices.append(int(idx_str))
        except ValueError:
            continue

    return sorted(indices)


def load_mode(folder, mode_index, heading="mode", file_format="npz"):
    """
    Load a single mode from file.

    Parameters
    ----------
    folder : str or Path
        Path to the folder containing mode files.
    mode_index : int
        Index of the mode to load.
    heading : str
        Prefix of mode files (e.g., "mode" for "mode_0000.npz").
    file_format : str
        File extension (default "npz").

    Returns
    -------
    mode_data : dict
        Dictionary containing:
        - 'field': complex ndarray, shape (Nx, Ny) or (Nx,) or (Ny,)
        - 'beta': float, propagation constant
        - 'x': 1D array, x coordinates (if available)
        - 'y': 1D array, y coordinates (if available)
        - 'index': int, mode index
    """
    folder = Path(folder)
    filepath = folder / f"{heading}_{mode_index:04d}.{file_format}"

    if not filepath.exists():
        raise FileNotFoundError(f"Mode file not found: {filepath}")

    data = np.load(filepath, allow_pickle=True)

    result = {
        'field': np.asarray(data['field']),
        'index': mode_index,
    }

    # Optional fields
    if 'beta' in data:
        result['beta'] = float(data['beta'])
    if 'x' in data:
        result['x'] = np.asarray(data['x'])
    if 'y' in data:
        result['y'] = np.asarray(data['y'])

    return result


def _get_mode_dimensionality(field):
    """
    Determine the dimensionality type of a mode field.

    Returns
    -------
    dim_type : str
        One of: '2d', '1d_x', '1d_y', '0d'
    """
    shape = field.shape
    if field.ndim == 1:
        return '1d_x'  # Assume 1D is along x
    elif field.ndim == 2:
        Nx, Ny = shape
        if Nx == 1 and Ny == 1:
            return '0d'
        elif Nx == 1:
            return '1d_y'
        elif Ny == 1:
            return '1d_x'
        else:
            return '2d'
    else:
        raise ValueError(f"Unexpected field shape: {shape}")


def plot_mode_2d(mode_data,
                 quantity='intensity',
                 cmap='hot',
                 figsize=(6, 5),
                 title=None,
                 show_colorbar=True,
                 ax=None):
    """
    Plot a 2D mode profile.

    Parameters
    ----------
    mode_data : dict
        Output from load_mode(), or just a dict with 'field' key.
    quantity : str
        'intensity', 'abs', 'real', 'imag', or 'phase'.
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size if creating new figure.
    title : str or None
        Custom title. If None, auto-generates.
    show_colorbar : bool
        Whether to show colorbar.
    ax : matplotlib Axes or None
        If provided, plot on this axes.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    field = np.asarray(mode_data['field'])
    mode_idx = mode_data.get('index', '?')
    beta = mode_data.get('beta', None)

    # Handle different shapes
    if field.ndim == 1:
        # 1D mode - reshape for imshow
        field = field.reshape(-1, 1) if field.size > 1 else field.reshape(1, 1)

    Nx, Ny = field.shape

    # Get coordinates
    x = mode_data.get('x', np.arange(Nx))
    y = mode_data.get('y', np.arange(Ny))
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Compute quantity to plot
    if quantity == 'intensity':
        data = np.abs(field)**2
        cbar_label = r'$|E|^2$'
    elif quantity == 'abs':
        data = np.abs(field)
        cbar_label = r'$|E|$'
    elif quantity == 'real':
        data = np.real(field)
        cbar_label = r'Re$\{E\}$'
    elif quantity == 'imag':
        data = np.imag(field)
        cbar_label = r'Im$\{E\}$'
    elif quantity == 'phase':
        data = np.angle(field)
        cbar_label = r'arg$(E)$'
    else:
        raise ValueError("quantity must be 'intensity', 'abs', 'real', 'imag', or 'phase'")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine extent
    if x.size >= 2 and y.size >= 2:
        dx = x[1] - x[0] if x.size >= 2 else 1.0
        dy = y[1] - y[0] if y.size >= 2 else 1.0
        extent = [x.min() - dx/2, x.max() + dx/2, y.min() - dy/2, y.max() + dy/2]
    else:
        extent = None

    im = ax.imshow(data.T, origin='lower', cmap=cmap, extent=extent, aspect='auto')

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    # Labels
    ax.set_xlabel('x (m)' if x.size > 1 else 'x')
    ax.set_ylabel('y (m)' if y.size > 1 else 'y')

    # Title
    if title is None:
        beta_str = f", β={beta:.2f} rad/m" if beta is not None else ""
        title = f"Mode {mode_idx}{beta_str}"
    ax.set_title(title)

    return fig, ax


def plot_mode_1d(mode_data,
                 quantity='intensity',
                 figsize=(8, 4),
                 title=None,
                 ax=None,
                 label=None,
                 **plot_kwargs):
    """
    Plot a 1D mode profile (for Nx=1 or Ny=1 cases).

    Parameters
    ----------
    mode_data : dict
        Output from load_mode(), or just a dict with 'field' key.
    quantity : str
        'intensity', 'abs', 'real', 'imag', or 'phase'.
    figsize : tuple
        Figure size if creating new figure.
    title : str or None
        Custom title.
    ax : matplotlib Axes or None
        If provided, plot on this axes.
    label : str or None
        Legend label.
    **plot_kwargs : dict
        Additional arguments passed to ax.plot().

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    field = np.asarray(mode_data['field'])
    mode_idx = mode_data.get('index', '?')
    beta = mode_data.get('beta', None)

    # Flatten to 1D
    field_1d = field.ravel()
    N = field_1d.size

    # Determine which axis is the "real" one
    x = mode_data.get('x', None)
    y = mode_data.get('y', None)

    if x is not None and np.asarray(x).size == N:
        coord = np.asarray(x).ravel()
        coord_label = 'x (m)'
    elif y is not None and np.asarray(y).size == N:
        coord = np.asarray(y).ravel()
        coord_label = 'y (m)'
    else:
        coord = np.arange(N)
        coord_label = 'index'

    # Compute quantity to plot
    if quantity == 'intensity':
        data = np.abs(field_1d)**2
        ylabel = r'$|E|^2$'
    elif quantity == 'abs':
        data = np.abs(field_1d)
        ylabel = r'$|E|$'
    elif quantity == 'real':
        data = np.real(field_1d)
        ylabel = r'Re$\{E\}$'
    elif quantity == 'imag':
        data = np.imag(field_1d)
        ylabel = r'Im$\{E\}$'
    elif quantity == 'phase':
        data = np.angle(field_1d)
        ylabel = r'arg$(E)$'
    else:
        raise ValueError("quantity must be 'intensity', 'abs', 'real', 'imag', or 'phase'")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot
    if label is None:
        label = f"Mode {mode_idx}"
    ax.plot(coord, data, label=label, **plot_kwargs)

    ax.set_xlabel(coord_label)
    ax.set_ylabel(ylabel)

    # Title
    if title is None:
        beta_str = f", β={beta:.2f} rad/m" if beta is not None else ""
        title = f"Mode {mode_idx}{beta_str}"
    ax.set_title(title)

    return fig, ax


def plot_mode(mode_data, quantity='intensity', **kwargs):
    """
    Automatically plot a mode using the appropriate function based on dimensionality.

    Parameters
    ----------
    mode_data : dict
        Output from load_mode().
    quantity : str
        'intensity', 'abs', 'real', 'imag', or 'phase'.
    **kwargs : dict
        Additional arguments passed to plot_mode_2d or plot_mode_1d.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    field = np.asarray(mode_data['field'])
    dim_type = _get_mode_dimensionality(field)

    if dim_type == '2d':
        return plot_mode_2d(mode_data, quantity=quantity, **kwargs)
    elif dim_type in ('1d_x', '1d_y'):
        return plot_mode_1d(mode_data, quantity=quantity, **kwargs)
    elif dim_type == '0d':
        # Single point - just print the value
        print(f"Mode {mode_data.get('index', '?')}: field = {field.ravel()[0]}")
        return None, None
    else:
        raise ValueError(f"Unknown dimensionality: {dim_type}")


def plot_modes_gallery(folder,
                       mode_indices=None,
                       heading="mode",
                       file_format="npz",
                       quantity='intensity',
                       cmap='hot',
                       max_modes=16,
                       ncols=4,
                       figsize_per_mode=(3, 2.5),
                       suptitle=None):
    """
    Plot a gallery of modes from a folder.

    Parameters
    ----------
    folder : str or Path
        Path to the folder containing mode files.
    mode_indices : list of int or None
        Specific mode indices to plot. If None, plots first max_modes.
    heading : str
        Prefix of mode files.
    file_format : str
        File extension.
    quantity : str
        'intensity', 'abs', 'real', 'imag', or 'phase'.
    cmap : str
        Colormap for 2D plots.
    max_modes : int
        Maximum number of modes to show if mode_indices is None.
    ncols : int
        Number of columns in the gallery.
    figsize_per_mode : tuple
        Figure size per subplot.
    suptitle : str or None
        Super title for the figure.

    Returns
    -------
    fig : matplotlib figure
    """
    folder = Path(folder)

    # Get mode indices
    if mode_indices is None:
        available = list_modes_in_folder(folder, heading, file_format)
        mode_indices = available[:max_modes]

    if len(mode_indices) == 0:
        print(f"No modes found in {folder}")
        return None

    n_modes = len(mode_indices)
    nrows = int(np.ceil(n_modes / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_mode[0] * ncols,
                                      figsize_per_mode[1] * nrows),
                             squeeze=False)

    for i, mode_idx in enumerate(mode_indices):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        try:
            mode_data = load_mode(folder, mode_idx, heading, file_format)
            field = np.asarray(mode_data['field'])
            dim_type = _get_mode_dimensionality(field)

            beta = mode_data.get('beta', None)
            beta_str = f"\nβ={beta:.2f}" if beta is not None else ""

            if dim_type == '2d':
                plot_mode_2d(mode_data, quantity=quantity, cmap=cmap, ax=ax,
                             title=f"Mode {mode_idx}{beta_str}", show_colorbar=False)
            elif dim_type in ('1d_x', '1d_y'):
                plot_mode_1d(mode_data, quantity=quantity, ax=ax,
                             title=f"Mode {mode_idx}{beta_str}")
            else:
                ax.text(0.5, 0.5, f"Mode {mode_idx}\n(0D)",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\nmode {mode_idx}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide empty subplots
    for i in range(n_modes, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].axis('off')

    if suptitle is None:
        suptitle = f"Modes from {folder.name}"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()

    return fig


def plot_modes_overlay_1d(folder,
                          mode_indices=None,
                          heading="mode",
                          file_format="npz",
                          quantity='intensity',
                          max_modes=10,
                          figsize=(10, 6),
                          title=None):
    """
    Overlay multiple 1D modes on the same plot.

    Parameters
    ----------
    folder : str or Path
        Path to the folder containing mode files.
    mode_indices : list of int or None
        Specific mode indices to plot. If None, plots first max_modes.
    heading : str
        Prefix of mode files.
    file_format : str
        File extension.
    quantity : str
        'intensity', 'abs', 'real', 'imag', or 'phase'.
    max_modes : int
        Maximum number of modes to show if mode_indices is None.
    figsize : tuple
        Figure size.
    title : str or None
        Plot title.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    folder = Path(folder)

    # Get mode indices
    if mode_indices is None:
        available = list_modes_in_folder(folder, heading, file_format)
        mode_indices = available[:max_modes]

    if len(mode_indices) == 0:
        print(f"No modes found in {folder}")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    for mode_idx in mode_indices:
        try:
            mode_data = load_mode(folder, mode_idx, heading, file_format)
            beta = mode_data.get('beta', None)
            label = f"Mode {mode_idx}" + (f" (β={beta:.2f})" if beta else "")
            plot_mode_1d(mode_data, quantity=quantity, ax=ax, label=label)
        except Exception as e:
            print(f"Error loading mode {mode_idx}: {e}")

    ax.legend(loc='best', fontsize=8)

    if title is None:
        title = f"1D Modes from {folder.name}"
    ax.set_title(title)

    return fig, ax


# ======================================================================================
# 1D ANIMATION: I(x) or I(t) vs z
# ======================================================================================

def make_1d_z_animation(field4d,
                        coord_axis='x',
                        reduce_method='centerline',
                        coord=None,
                        z=None,
                        quantity='intensity',
                        norm='global',
                        z_window=None,
                        frame_window=None,
                        stride=1,
                        fps=30,
                        filename='1d_z.gif',
                        figsize=(8, 4),
                        ylim=None,
                        dpi=120):
    """
    Animate a 1D intensity profile (along x, y, or t) stepping through z frames.

    Parameters
    ----------
    field4d : np.ndarray, complex, shape (Nx, Ny, Nt, Nz)
        The complex field.
    coord_axis : str
        Which axis to display: 'x', 'y', or 't'.
    reduce_method : str
        How to reduce over other axes: 'centerline' (center slice),
        'sum', or 'mean'.
    coord : 1D array or None
        Coordinate values for the displayed axis (for labeling).
    z : 1D array or None
        z axis values. Required for z_window.
    quantity : str
        'intensity', 'abs', 'real', 'imag', or 'phase'.
    norm : str
        'global' or 'per_frame'.
    z_window : tuple or None
        (z_start, z_end) to limit z range.
    frame_window : tuple or None
        (i_start, i_end) frame indices along z.
    stride : int
        Take every stride-th frame.
    fps : int
        Frames per second.
    filename : str
        Output GIF path.
    figsize : tuple
        Figure size.
    ylim : tuple or None
        Fixed y-axis limits (ymin, ymax).
    dpi : int
        Figure DPI.

    Returns
    -------
    filename : str
        Path to the saved GIF.
    """
    assert field4d.ndim == 4, "Expected (Nx, Ny, Nt, Nz)."
    Nx, Ny, Nt, Nz = field4d.shape

    # --- Extract 1D profile at each z ---
    def _extract_1d(F3d):
        """F3d: (Nx, Ny, Nt) -> 1D profile along coord_axis."""
        if coord_axis == 'x':
            if reduce_method == 'centerline':
                return F3d[:, Ny // 2, Nt // 2]
            elif reduce_method == 'sum':
                return F3d.sum(axis=(1, 2))
            elif reduce_method == 'mean':
                return F3d.mean(axis=(1, 2))
        elif coord_axis == 'y':
            if reduce_method == 'centerline':
                return F3d[Nx // 2, :, Nt // 2]
            elif reduce_method == 'sum':
                return F3d.sum(axis=(0, 2))
            elif reduce_method == 'mean':
                return F3d.mean(axis=(0, 2))
        elif coord_axis == 't':
            if reduce_method == 'centerline':
                return F3d[Nx // 2, Ny // 2, :]
            elif reduce_method == 'sum':
                return F3d.sum(axis=(0, 1))
            elif reduce_method == 'mean':
                return F3d.mean(axis=(0, 1))
        raise ValueError(f"Invalid coord_axis='{coord_axis}' or reduce_method='{reduce_method}'")

    # Map to quantity
    if quantity == 'intensity':
        compute = lambda F: np.abs(F) ** 2
        ylabel = r'$|E|^2$'
    elif quantity == 'abs':
        compute = lambda F: np.abs(F)
        ylabel = r'$|E|$'
    elif quantity == 'real':
        compute = lambda F: np.real(F)
        ylabel = r'Re$\{E\}$'
    elif quantity == 'imag':
        compute = lambda F: np.imag(F)
        ylabel = r'Im$\{E\}$'
    elif quantity == 'phase':
        compute = lambda F: np.angle(F)
        ylabel = r'arg$(E)$'
    else:
        raise ValueError("quantity must be 'intensity', 'abs', 'real', 'imag', or 'phase'")

    # --- z frame selection ---
    if z_window is not None:
        if z is None:
            raise ValueError("z_window requires z array.")
        z = np.asarray(z)
        z_start, z_end = sorted(z_window)
        i0 = int(np.clip(np.searchsorted(z, z_start, side='left'), 0, Nz - 1))
        i1 = int(np.clip(np.searchsorted(z, z_end, side='right'), 0, Nz))
    elif frame_window is not None:
        i0, i1 = frame_window
        i0 = int(np.clip(i0, 0, Nz))
        i1 = int(np.clip(i1, i0 + 1, Nz))
    else:
        i0, i1 = 0, Nz

    frame_indices = np.arange(i0, i1, stride, dtype=int)
    if frame_indices.size == 0:
        raise ValueError("Selected z/frame window is empty after applying stride.")

    # Build data matrix: (N_coord, N_frames)
    N_coord = {'x': Nx, 'y': Ny, 't': Nt}[coord_axis]
    data_all = np.zeros((N_coord, frame_indices.size))
    for k, iz in enumerate(frame_indices):
        data_all[:, k] = _extract_1d(compute(field4d[..., iz]))

    z_sel = z[frame_indices] if z is not None else None
    Nf = frame_indices.size

    # Coordinate axis
    if coord is not None:
        xvals = np.asarray(coord)
    else:
        xvals = np.arange(N_coord)
    xlabel = coord_axis

    # Normalization
    if norm == 'global':
        ymin = np.nanmin(data_all)
        ymax = np.nanmax(data_all)
        margin = 0.05 * (ymax - ymin) if ymax != ymin else 1.0
        auto_ylim = (ymin - margin, ymax + margin)
    else:
        auto_ylim = None

    use_ylim = ylim if ylim is not None else auto_ylim

    # --- Animate ---
    fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot(xvals, data_all[:, 0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if use_ylim is not None:
        ax.set_ylim(use_ylim)

    if z_sel is not None:
        title = ax.set_title(f"z = {z_sel[0]:.4g}")
    else:
        title = ax.set_title(f"frame = {frame_indices[0]}")

    def update(k):
        line.set_ydata(data_all[:, k])
        if norm == 'per_frame':
            ymin_f = np.nanmin(data_all[:, k])
            ymax_f = np.nanmax(data_all[:, k])
            margin_f = 0.05 * (ymax_f - ymin_f) if ymax_f != ymin_f else 1.0
            ax.set_ylim(ymin_f - margin_f, ymax_f + margin_f)
        if z_sel is not None:
            title.set_text(f"z = {z_sel[k]:.4g}")
        else:
            title.set_text(f"frame = {frame_indices[k]}")
        return (line,)

    anim = FuncAnimation(fig, update, frames=Nf, interval=1000.0 / fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)
    return filename


# ======================================================================================
# TEMPORAL vs Z HEATMAP
# ======================================================================================

def make_temporal_vs_z_plot(
    out: dict,
    args: dict | None = None,
    *,
    reduce: str = "centerline",
    log10: bool = False,
    normalize: bool = False,
    cmap: str | plt.Colormap = "jet",
    figsize: tuple = (8, 4),
    clim: tuple | None = None,
    title: str | None = None,
    z_window: tuple[float, float] | None = None,
    t_window: tuple[float, float] | None = None,
):
    """
    Static 2D heatmap: time (vertical) vs z (horizontal), showing I(t, z).

    Analogous to make_transverse_vs_z_vs_I_plot but for the temporal axis.
    Reduces over x, y via centerline/sum/mean.

    Parameters
    ----------
    out : dict
        Propagation output with 'field' (Nx, Ny, Nt, Nz), 'dt', 'dx'.
    args : dict or None
        Propagation args (for Lt, save_at, etc.).
    reduce : str
        'centerline', 'sum', or 'mean' — how to reduce x, y.
    log10 : bool
        If True, show log10(I + eps).
    normalize : bool
        Normalize by global max before optional log.
    cmap : str or Colormap
        Colormap.
    figsize : tuple
        Figure size.
    clim : tuple or None
        (vmin, vmax) for color scale.
    title : str or None
        Plot title.
    z_window : tuple or None
        (z_min, z_max) in meters.
    t_window : tuple or None
        (t_min, t_max) in seconds.

    Returns
    -------
    fig, ax, t_sel, z_sel, I_map
    """
    F = np.asarray(out["field"])
    if F.ndim == 3:
        F = F[..., None]
    Nx, Ny, Nt, Nsave = F.shape

    dt = float(out.get("dt", 1.0))

    # --- z axis ---
    if "save_at" in out:
        z = np.asarray(out["save_at"], float)
    elif args is not None and "save_at" in args:
        z = np.asarray(args["save_at"], float)
        if z.ndim == 0:
            z = np.array([float(z)])
    else:
        z = np.arange(Nsave, dtype=float)
    if z.size != Nsave:
        z = np.linspace(0.0, float(Nsave - 1), Nsave)

    # --- t axis ---
    if args is not None and "Lt" in args:
        Lt = float(args["Lt"])
        t_axis = np.linspace(-Lt / 2, Lt / 2, Nt)
    else:
        t_axis = dt * (np.arange(Nt) - (Nt - 1) / 2.0)

    # --- z window ---
    if z_window is not None:
        zmin, zmax = float(min(*z_window)), float(max(*z_window))
        z_idx = np.where((z >= zmin) & (z <= zmax))[0]
        if z_idx.size == 0:
            z_idx = np.array([np.argmin(np.abs(z - zmin)), np.argmin(np.abs(z - zmax))])
            z_idx.sort()
    else:
        z_idx = np.arange(Nsave, dtype=int)
    Z_sel = z[z_idx]

    # --- t window ---
    if t_window is not None:
        tmin, tmax = float(min(*t_window)), float(max(*t_window))
        t_mask = (t_axis >= tmin) & (t_axis <= tmax)
        if not np.any(t_mask):
            jlo = int(np.argmin(np.abs(t_axis - tmin)))
            jhi = int(np.argmin(np.abs(t_axis - tmax)))
            if jlo > jhi:
                jlo, jhi = jhi, jlo
            jhi = min(jhi + 1, t_axis.size)
        else:
            j_inds = np.where(t_mask)[0]
            jlo, jhi = int(j_inds.min()), int(j_inds.max() + 1)
    else:
        jlo, jhi = 0, Nt
    T_sel = t_axis[jlo:jhi]

    # --- center indices ---
    cx, cy = Nx // 2, Ny // 2

    # --- build I(t, z) map ---
    I_map = np.zeros((T_sel.size, Z_sel.size), dtype=np.float64)

    for col, iz in enumerate(z_idx):
        Fz = F[..., iz]  # (Nx, Ny, Nt)
        I_xyt = np.abs(Fz) ** 2

        if reduce == "centerline":
            line = I_xyt[cx, cy, :]
        elif reduce == "sum":
            dx = float(out.get("dx", 1.0))
            dy = float(out.get("dy", out.get("dx", 1.0)))
            line = I_xyt.sum(axis=(0, 1)) * dx * dy
        elif reduce == "mean":
            line = I_xyt.mean(axis=(0, 1))
        else:
            raise ValueError("reduce must be 'centerline', 'sum', or 'mean'.")

        I_map[:, col] = line[jlo:jhi]

    # --- normalize & log ---
    eps = 1e-18 * (np.nanmax(I_map) if I_map.size else 1.0)
    if normalize and I_map.max() > 0:
        I_map = I_map / (I_map.max() + 1e-30)

    data_to_show = np.log10(I_map + eps) if log10 else I_map
    vmin, vmax = (None, None) if clim is None else clim

    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    # --- plot ---
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    extent = [Z_sel.min(), Z_sel.max(), T_sel.min(), T_sel.max()]
    im = ax.imshow(
        data_to_show,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if log10:
        cbar.set_label(r"$\log_{10}(\text{intensity})$ (arb.)")
    else:
        cbar.set_label(r"$|E|^2$ (arb.)")

    ax.set_xlabel("z (m)")
    ax.set_ylabel("t (s)")

    if title is None:
        lg = " (log)" if log10 else ""
        ax.set_title(f"I(t, z) — {reduce}{lg}")
    else:
        ax.set_title(title)

    return fig, ax, T_sel, Z_sel, I_map

