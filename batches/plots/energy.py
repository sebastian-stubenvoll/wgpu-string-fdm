import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter1d


def _dynamic_above_baseline(x):
    return x - np.min(x)


def _upper_envelope(y):
    """Upper envelope of an oscillating signal, interpolated through its peaks.

    Falls back to the raw signal when there are too few peaks to interpolate
    (e.g. a smooth, monotonically decaying total).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size < 3:
        return y
    peaks, _ = find_peaks(y)
    if peaks.size < 2:
        return y
    idx = np.concatenate(([0], peaks, [y.size - 1]))
    return np.interp(np.arange(y.size), idx, y[idx])


def plot_energies(time, trans_ke, rot_ke, bend_pe, shear_pe, print_totals=False,
                  normalized=False, mode_transfer=False, title_prefix="", filename=""):

    trans_ke = np.asarray(trans_ke, dtype=np.float64)
    rot_ke = np.asarray(rot_ke, dtype=np.float64)
    bend_pe = np.asarray(bend_pe, dtype=np.float64)
    shear_pe = np.asarray(shear_pe, dtype=np.float64)

    fig = plt.figure(figsize=(12, 6))

    if normalized or mode_transfer:
        bend_dyn = _dynamic_above_baseline(bend_pe)
        shear_dyn = _dynamic_above_baseline(shear_pe)
        kin = trans_ke + rot_ke
        pot = bend_dyn + shear_dyn
        total = kin + pot

        peak = np.max(total) if total.size else 0.0
        active = total > (1e-3 * peak) if peak > 0 else np.zeros_like(total, dtype=bool)

        def fraction(x, fill):
            out = np.full_like(x, fill)
            np.divide(x, total, out=out, where=active)
            return out

        active_idx = np.nonzero(active)[0]
        if active_idx.size:
            t0, t1 = time[active_idx[0]], time[active_idx[-1]]
            span = max(t1 - t0, 1e-9)
            view = (max(float(time[0]), t0 - 0.02 * span),
                    min(float(time[-1]), t1 + 0.02 * span))
        else:
            view = (float(time[0]), float(time[-1]))

        if normalized:
            trans = fraction(trans_ke, 0.0)
            rot = fraction(rot_ke, 0.0)
            bend = fraction(bend_dyn, 0.0)
            shear = fraction(shear_dyn, 0.0)

            plt.stackplot(
                time, trans, rot, bend, shear,
                labels=["Translational KE", "Rotational KE",
                        "Bend/Twist PE", "Shear/Stretch PE"],
                colors=["tab:blue", "tab:orange", "tab:green", "tab:red"],
            )
            plt.ylim(0, 1)
            plt.xlim(*view)
            plt.ylabel("Fraction of Dynamic Energy")
            title = f"[{title_prefix}] Energy Distribution (dynamic; ringing window)"
        else:  
            trans = fraction(trans_ke, np.nan)
            rot = fraction(rot_ke, np.nan)
            bend = fraction(bend_dyn, np.nan)
            shear = fraction(shear_dyn, np.nan)

            plt.plot(time, trans, label="Translational KE", alpha=0.5, linestyle=':')
            plt.plot(time, rot, label="Rotational KE", alpha=0.5, linestyle=':')
            plt.plot(time, bend, label="Bend/Twist PE", alpha=0.5, linestyle=':')
            plt.plot(time, shear, label="Shear/Stretch PE", alpha=0.5, linestyle=':')
            if print_totals:
                plt.plot(time, fraction(kin, np.nan), label="Total kinetic",
                         linewidth=2, color='blue')
                plt.plot(time, fraction(pot, np.nan), label="Total potential",
                         linewidth=2, color='orange')
            plt.ylim(-0.05, 1.05)
            plt.xlim(*view)
            plt.ylabel("Fraction of Dynamic Energy")
            title = f"[{title_prefix}] Energy Mode Transfer (dynamic fraction)"
    else:
        # Kinetic energy is zero at rest (no static offset) so it is shown as-is.
        # The potentials carry a large static tension baseline; we remove it by
        # subtracting the quiescent value (a constant), which keeps every curve
        # >= 0 and makes the total system energy a clean, decaying mechanical energy
        # equal to the sum of the plotted components.
        bend_dyn = _dynamic_above_baseline(bend_pe)
        shear_dyn = _dynamic_above_baseline(shear_pe)

        total_kin = trans_ke + rot_ke
        total_pot = bend_dyn + shear_dyn
        total_energy = total_kin + total_pot

        title = f"[{title_prefix}] Cosserat Rod Energy Breakdown (Offset Removed)"
        plt.plot(time, trans_ke, label="Translational KE", alpha=0.4, linestyle=':')
        plt.plot(time, rot_ke, label="Rotational KE", alpha=0.4, linestyle=':')
        plt.plot(time, bend_dyn, label="Bend/Twist PE", alpha=0.4, linestyle=':')
        plt.plot(time, shear_dyn, label="Shear/Stretch PE", alpha=0.4, linestyle=':')

        if print_totals:
            plt.plot(time, total_kin, label="Total kinetic", linewidth=2, color='blue')
            plt.plot(time, total_pot, label="Total potential", linewidth=2, color='orange')
            plt.plot(time, total_energy, label="Total system energy",
                     color='gray', linestyle='--', linewidth=1.5)
        plt.ylabel("Energy (Joules)")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_energy_envelope(time, trans_ke, rot_ke, bend_pe, shear_pe,
                         title_prefix="", filename=""):
    """Decay overview for the long phases: the total energies as thin lines,
    with peak-interpolated envelopes for the oscillating kinetic/potential."""
    trans_ke = np.asarray(trans_ke, dtype=np.float64)
    rot_ke = np.asarray(rot_ke, dtype=np.float64)
    bend_pe = np.asarray(bend_pe, dtype=np.float64)
    shear_pe = np.asarray(shear_pe, dtype=np.float64)

    bend_dyn = _dynamic_above_baseline(bend_pe)
    shear_dyn = _dynamic_above_baseline(shear_pe)

    total_kin = trans_ke + rot_ke
    total_pot = bend_dyn + shear_dyn
    total_energy = total_kin + total_pot

    fig = plt.figure(figsize=(12, 6))

    # Faint raw totals for context, envelopes on top as the readable lines.
    plt.plot(time, total_kin, color='blue', alpha=0.15, linewidth=0.5)
    plt.plot(time, total_pot, color='orange', alpha=0.15, linewidth=0.5)

    plt.plot(time, _upper_envelope(total_kin), color='blue', linewidth=1.0,
             label="Kinetic (envelope)")
    plt.plot(time, _upper_envelope(total_pot), color='orange', linewidth=1.0,
             label="Potential (envelope)")
    plt.plot(time, total_energy, color='gray', linestyle='--', linewidth=1.0,
             label="Total system energy")

    plt.title(f"[{title_prefix}] Energy Decay Envelopes (Offset Removed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (Joules)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


# The eight energy components: (label, family colour, axial?). Colour by family,
# transverse solid / axial dashed. The potential parts are split by *strain
# component*, not by motion: with shear stiffness far above the tension, the
# frame follows the tangent and a transverse slope shows up as axial strain, so
# the transverse wave's potential energy lives in "axial strain".
COMPONENTS = (
    ("Transversal kinetic", 'tab:blue', False),
    ("Longitudinal kinetic", 'tab:blue', True),
    ("Tilting rotation", 'tab:orange', False),
    ("Torsional rotation", 'tab:orange', True),
    ("Bending", 'tab:green', False),
    ("Twisting", 'tab:green', True),
    ("Shear strain", 'tab:red', False),
    ("Axial strain (incl. transverse-wave tension)", 'tab:red', True),
)


def energy_components(trans_ke, rot_ke, bend_pe, shear_pe,
                      trans_ke_long, rot_ke_tor, bend_pe_twist, shear_pe_ext):
    """Split the four energy families into transverse and axial parts, in
    COMPONENTS order. The passed *_long/_tor/_twist/_ext arrays are the axial
    parts; the transverse part is the remainder (all quadratic forms are
    diagonal, so transverse + axial == total exactly). Kinetic parts are zero at
    rest and kept as-is; potential parts carry a static tension baseline
    (dominantly in extension), removed per component."""
    a = [np.asarray(x, dtype=np.float64) for x in
         (trans_ke, rot_ke, bend_pe, shear_pe, trans_ke_long, rot_ke_tor, bend_pe_twist, shear_pe_ext)]
    trans_ke, rot_ke, bend_pe, shear_pe, trans_ke_long, rot_ke_tor, bend_pe_twist, shear_pe_ext = a
    return [
        trans_ke - trans_ke_long,
        trans_ke_long,
        rot_ke - rot_ke_tor,
        rot_ke_tor,
        _dynamic_above_baseline(bend_pe - bend_pe_twist),
        _dynamic_above_baseline(bend_pe_twist),
        _dynamic_above_baseline(shear_pe - shear_pe_ext),
        _dynamic_above_baseline(shear_pe_ext),
    ]


def plot_energy_components(time, trans_ke, rot_ke, bend_pe, shear_pe,
                           trans_ke_long, rot_ke_tor, bend_pe_twist, shear_pe_ext,
                           title_prefix="", filename=""):
    """Fine-grained breakdown: each of the four energy families split into its
    transverse and axial parts (eight lines total). Only the kinetic split
    (v_x vs v_y, v_z) cleanly separates transverse from longitudinal motion;
    the potential split is by strain component (see COMPONENTS)."""
    comps = energy_components(trans_ke, rot_ke, bend_pe, shear_pe,
                              trans_ke_long, rot_ke_tor, bend_pe_twist, shear_pe_ext)

    fig = plt.figure(figsize=(12, 6))
    for y, (label, color, axial) in zip(comps, COMPONENTS):
        plt.plot(time, y, color=color, linewidth=1.0, linestyle='--' if axial else '-', label=label)

    plt.title(f"[{title_prefix}] Energy Component Breakdown (Offset Removed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (Joules)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_energy_totals(time, trans_ke, rot_ke, bend_pe, shear_pe,
                       title_prefix="", filename=""):
    """Totals only: kinetic, potential and system energy on top, and the system
    energy alone on its own y-scale below so decay/drift is readable."""
    trans_ke = np.asarray(trans_ke, dtype=np.float64)
    rot_ke = np.asarray(rot_ke, dtype=np.float64)
    total_kin = trans_ke + rot_ke
    total_pot = (_dynamic_above_baseline(np.asarray(bend_pe, dtype=np.float64))
                 + _dynamic_above_baseline(np.asarray(shear_pe, dtype=np.float64)))
    total_energy = total_kin + total_pot

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 2]})

    ax_top.plot(time, total_kin, color='tab:blue', linewidth=0.6, alpha=0.6,
                label="Total kinetic")
    ax_top.plot(time, total_pot, color='tab:orange', linewidth=0.6, alpha=0.6,
                label="Total potential")
    ax_top.plot(time, total_energy, color='black', linewidth=2.0, zorder=5,
                label="Total system energy")
    ax_top.set_ylabel("Energy (Joules)")
    ax_top.set_title(f"[{title_prefix}] Total Energies (Offset Removed)")
    ax_top.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(time, total_energy, color='black', linewidth=1.2)
    ax_bot.set_ylabel("System Energy (Joules)")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_title("Total system energy (own scale)")
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def _moving_average(y, window):
    """Centred O(n) moving average; the window shrinks at the edges."""
    y = np.asarray(y, dtype=np.float64)
    window = int(max(1, min(window, y.size)))
    c = np.concatenate(([0.0], np.cumsum(y)))
    half = window // 2
    lo = np.clip(np.arange(y.size) - half, 0, y.size)
    hi = np.clip(np.arange(y.size) - half + window, 0, y.size)
    return (c[hi] - c[lo]) / (hi - lo)


def _smooth(y, window, passes=3):
    """Repeated boxcar (~Gaussian): suppresses the 2f KE ripple far better than
    a single boxcar, which matters once the result is differentiated."""
    for _ in range(passes):
        y = _moving_average(y, window)
    return y


def _blank_edges(y, window):
    """NaN the ends where the shrinking smoothing window biases the result
    (three boxcar passes reach 1.5 windows in from each end)."""
    edge = int(1.5 * window)
    y = np.array(y, dtype=np.float64)
    if y.size > 4 * edge:
        y[:edge] = np.nan
        y[-edge:] = np.nan
    return y


def _window_frames(time, window_s):
    dt = float(time[1] - time[0]) if len(time) > 1 else 1.0
    return max(1, int(round(window_s / dt)))


def plot_kinetic_trans_long(time, trans_ke, trans_ke_long, window_s=0.02,
                            title_prefix="", filename=""):
    """Transversal vs longitudinal translational KE on a log scale, plus the
    ratio of their envelopes (running max over window_s)."""
    trans_ke = np.asarray(trans_ke, dtype=np.float64)
    long_ke = np.asarray(trans_ke_long, dtype=np.float64)
    transv_ke = np.maximum(trans_ke - long_ke, 0.0)

    window = _window_frames(time, window_s)
    # Running max, then a boxcar of the same width to remove the staircase.
    env_t = _moving_average(maximum_filter1d(transv_ke, window, mode='nearest'), window)
    env_l = _moving_average(maximum_filter1d(long_ke, window, mode='nearest'), window)
    tiny = 1e-30

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 2]})

    ax_top.plot(time, np.maximum(transv_ke, tiny), color='tab:blue', alpha=0.2, linewidth=0.5)
    ax_top.plot(time, np.maximum(long_ke, tiny), color='tab:purple', alpha=0.2, linewidth=0.5)
    ax_top.plot(time, np.maximum(env_t, tiny), color='tab:blue', linewidth=1.2,
                label="Transversal KE (envelope)")
    ax_top.plot(time, np.maximum(env_l, tiny), color='tab:purple', linewidth=1.2,
                label="Longitudinal KE (envelope)")
    ax_top.set_yscale('log')
    positive = np.concatenate([env_t[env_t > 0], env_l[env_l > 0]])
    if positive.size:
        ax_top.set_ylim(max(positive.min(), positive.max() * 1e-12), positive.max() * 2)
    ax_top.set_ylabel("Energy (Joules)")
    ax_top.set_title(f"[{title_prefix}] Transversal vs Longitudinal Kinetic Energy")
    ax_top.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_top.grid(True, which='both', alpha=0.3)

    ratio = np.full_like(env_t, np.nan)
    np.divide(env_l, env_t, out=ratio, where=env_t > 0)
    ratio[ratio <= 0] = np.nan
    ax_bot.plot(time, ratio, color='tab:gray', linewidth=1.0)
    ax_bot.set_yscale('log')
    ax_bot.set_ylabel("E_long / E_trans (envelopes)")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_kinetic_transfer(time, trans_ke, trans_ke_long, window_s=0.02,
                          title_prefix="", filename=""):
    """Energy exchange between transversal and longitudinal motion.

    Instantaneous KE swings at twice each mode frequency as energy trades with
    PE inside the mode; smoothing over a couple of fundamental periods
    (window_s) removes that, leaving <KE> ~ half of each family's energy. The
    time derivative of the averages is the power flowing into (positive) or
    out of (negative) each family.
    """
    time = np.asarray(time, dtype=np.float64)
    trans_ke = np.asarray(trans_ke, dtype=np.float64)
    long_ke = np.asarray(trans_ke_long, dtype=np.float64)
    transv_ke = trans_ke - long_ke

    window = _window_frames(time, window_s)
    avg_t = _smooth(transv_ke, window)
    avg_l = _smooth(long_ke, window)

    total = avg_t + avg_l
    share = np.full_like(total, np.nan)
    np.divide(avg_l, total, out=share, where=total > 1e-3 * np.max(total))

    p_t = _blank_edges(np.gradient(avg_t, time) if time.size > 1 else np.zeros_like(avg_t), window)
    p_l = _blank_edges(np.gradient(avg_l, time) if time.size > 1 else np.zeros_like(avg_l), window)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(time, avg_t, color='tab:blue', label="\u27e8Transversal KE\u27e9")
    ax.set_ylabel("Transversal (J)", color='tab:blue')
    ax2 = ax.twinx()
    ax2.plot(time, avg_l, color='tab:purple', label="\u27e8Longitudinal KE\u27e9")
    ax2.set_ylabel("Longitudinal (J)", color='tab:purple')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='upper right')
    ax.set_title(f"[{title_prefix}] Transversal \u2194 Longitudinal Kinetic Energy Transfer "
                 f"(smoothed over {window_s * 1e3:.1f} ms)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(time, share, color='tab:gray')
    ax.set_ylabel("Longitudinal share")
    ax.grid(True, alpha=0.3)

    # Powers differ by orders of magnitude, so each gets its own linear axis
    # (both centred on zero so the sign comparison reads directly).
    ax = axes[2]
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.plot(time, p_t, color='tab:blue', linewidth=1.0, label="d\u27e8E_trans\u27e9/dt")
    ax.set_ylabel("Transversal power (W)", color='tab:blue')
    ax2 = ax.twinx()
    ax2.plot(time, p_l, color='tab:purple', linewidth=1.0, label="d\u27e8E_long\u27e9/dt")
    # Longitudinal gains while transversal loses -> transfer trans -> long.
    ax2.fill_between(time, 0, p_l, where=(p_l > 0) & (p_t < 0), color='tab:purple',
                     alpha=0.25, label="trans \u2192 long")
    ax2.fill_between(time, 0, p_l, where=(p_l < 0) & (p_t > 0), color='tab:blue',
                     alpha=0.25, label="long \u2192 trans")
    ax2.set_ylabel("Longitudinal power (W)", color='tab:purple')
    for a, p in ((ax, p_t), (ax2, p_l)):
        lim = 1.1 * np.nanmax(np.abs(p)) if np.any(np.isfinite(p)) and np.nanmax(np.abs(p)) > 0 else 1.0
        a.set_ylim(-lim, lim)
    handles = ax.get_lines()[1:] + ax2.get_lines() + ax2.collections
    ax.legend(handles, [h.get_label() for h in handles],
              loc='center left', bbox_to_anchor=(1.08, 0.5))
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


# Frames averaged per row of the per-element energy map (0.32 ms at 50 kHz).
ENERGY_DENSITY_BLOCK = 16


def block_mean(a, block):
    """Mean over consecutive blocks of `block` rows (last block may be short)."""
    starts = np.arange(0, a.shape[0], block)
    counts = np.diff(np.append(starts, a.shape[0]))
    return (np.add.reduceat(a, starts, axis=0) / counts[:, None]).astype(np.float32)


def block_centers(offset, n_frames, block, frame_dt):
    """Times (s) of the block_mean rows for a chunk starting at frame `offset`."""
    starts = np.arange(0, n_frames, block)
    counts = np.diff(np.append(starts, n_frames))
    return (offset + starts + 0.5 * (counts - 1)) * frame_dt


def plot_energy_density(time, density, baseline, length, title_prefix="", filename=""):
    """Total energy of every node/element over time (heatmap).

    density: (rows, N) block-averaged per-element energy (node terms plus half
    of each adjacent edge), baseline: (N,) per-element static energy (its
    minimum over the whole run) which is subtracted.
    """
    e = np.asarray(density, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)[None, :]
    peak = float(e.max()) if e.size else 0.0
    if peak <= 0:
        peak = 1.0
    e = np.maximum(e, peak * 1e-6)
    n = e.shape[1]
    x = np.linspace(0.0, length, n)
    t = np.asarray(time, dtype=np.float64)
    dt = (t[1] - t[0]) if t.size > 1 else 1e-3
    dx = x[1] - x[0]

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(e, aspect='auto', origin='lower', interpolation='nearest', cmap='inferno',
                   norm=LogNorm(vmin=peak * 1e-6, vmax=peak),
                   extent=(x[0] - dx / 2, x[-1] + dx / 2, t[0] - dt / 2, t[-1] + dt / 2))
    fig.colorbar(im, ax=ax, label="Energy per element (J, static part removed)")
    ax.set_xlabel("Position along string (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"[{title_prefix}] Energy per Node/Element")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_energy_balance(time, trans_ke, rot_ke, bend_pe, shear_pe, p_damp, release_time=None,
                        title_prefix="", filename=""):
    """Energy bookkeeping: stored energy E, cumulative damping loss D and E + D.

    With all losses accounted for, E + D only changes while the hammer does
    work; after release it should stay flat. Any remaining drift is energy the
    bookkeeping can't explain (integrator error or energy-formula mismatch).
    """
    time = np.asarray(time, dtype=np.float64)
    stored = (np.asarray(trans_ke, np.float64) + np.asarray(rot_ke, np.float64)
              + _dynamic_above_baseline(np.asarray(bend_pe, np.float64))
              + _dynamic_above_baseline(np.asarray(shear_pe, np.float64)))
    dissipated = cumulative_trapezoid(np.asarray(p_damp, np.float64), time, initial=0.0)
    budget = stored + dissipated

    has_release = release_time is not None and time[0] < release_time < time[-1]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})

    ax = axes[0]
    ax.plot(time, stored, color='tab:blue', linewidth=0.8, label="Stored E (offset removed)")
    ax.plot(time, dissipated, color='tab:red', linewidth=1.2, label="Dissipated D = \u222bP_damp dt")
    ax.plot(time, budget, color='black', linewidth=1.6, label="E + D")
    title = f"[{title_prefix}] Energy Balance"
    if has_release:
        i_rel = int(np.searchsorted(time, release_time))
        hammer_in = budget[i_rel] - budget[0]
        ax.axvline(release_time, color='tab:gray', linestyle=':', label="End of excitation window")
        title += f" (hammer input \u2248 {hammer_in:.3g} J)"
    ax.set_ylabel("Energy (J)")
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    # Drift of E + D relative to a reference point: the end of the excitation
    # window if it's in range, else the phase start.
    ax = axes[1]
    i_ref = int(np.searchsorted(time, release_time)) if has_release else 0
    scale = np.max(np.abs(budget)) if budget.size and np.max(np.abs(budget)) > 0 else 1.0
    drift = 100.0 * (budget - budget[i_ref]) / scale
    ax.plot(time[i_ref:], drift[i_ref:], color='black', linewidth=1.0)
    ax.axhline(0.0, color='tab:gray', linewidth=0.8)
    ax.set_ylabel("E + D drift (% of peak)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Unaccounted energy after " + ("the excitation window" if has_release else "phase start"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_motion_power(time, comps, p_damp, window_s=0.02, title_prefix="", filename=""):
    """Power flowing into (+) / out of (-) each kind of motion.

    Subsystem energies come from kinetic energy only (the one split that
    cleanly separates motion types), using E ~ 2<KE> (virial theorem, averaged
    over whole periods via window_s):
      transverse   = 2<transversal KE + tilting rotation KE>
      longitudinal = 2<longitudinal KE>
      torsional    = 2<torsional rotation KE>
    Each gets its own panel and scale; a shared panel of relative rates
    (dE/dt)/E makes subsystems of very different sizes comparable.
    comps: the eight energy_components() arrays (COMPONENTS order).
    """
    time = np.asarray(time, dtype=np.float64)
    window = _window_frames(time, window_s)
    systems = (
        ("Transverse", 'tab:blue', 2.0 * (np.asarray(comps[0]) + np.asarray(comps[2]))),
        ("Longitudinal", 'tab:purple', 2.0 * np.asarray(comps[1])),
        ("Torsional", 'tab:orange', 2.0 * np.asarray(comps[3])),
    )

    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    rel_vals = []
    for ax, (name, color, e_raw) in zip(axes[:3], systems):
        e = _smooth(e_raw, window)
        p = _blank_edges(np.gradient(e, time) if time.size > 1 else np.zeros_like(e), window)
        ax.axhline(0.0, color='black', linewidth=0.6)
        ax.plot(time, p, color=color, linewidth=1.0)
        if np.any(np.isfinite(p)) and np.nanmax(np.abs(p)) > 0:
            lim = 1.1 * np.nanmax(np.abs(p))
            ax.set_ylim(-lim, lim)
        ax.set_ylabel("dE/dt (W)")
        ax.set_title(f"{name} motion", fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)

        # Relative rate, masked where the subsystem holds almost no energy.
        rel = np.full_like(p, np.nan)
        ok = np.isfinite(p) & (e > 1e-3 * np.max(e)) if e.size and np.max(e) > 0 else np.zeros_like(p, bool)
        rel[ok] = p[ok] / e[ok]
        axes[3].plot(time, rel, color=color, linewidth=1.0, label=name)
        rel_vals.append(rel[np.isfinite(rel)])

    ax = axes[3]
    ax.axhline(0.0, color='black', linewidth=0.6)
    vals = np.concatenate(rel_vals) if rel_vals else np.array([])
    if vals.size:
        lo, hi = np.percentile(vals, [1, 99])
        pad = 0.1 * max(hi - lo, 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("(dE/dt) / E (1/s)")
    ax.set_title("Relative rate (comparable across subsystems)", fontsize=10, loc='left')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    ax.plot(time, -_smooth(np.asarray(p_damp, np.float64), window), color='black', linewidth=1.0,
            label="\u2212 damping power")
    ax.axhline(0.0, color='tab:gray', linewidth=0.6)
    ax.set_ylabel("Loss (W)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Dissipation", fontsize=10, loc='left')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"[{title_prefix}] Power per Motion Type (E \u2248 2\u27e8KE\u27e9, "
                 f"smoothed over {window_s * 1e3:.1f} ms)")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_energy_share(time, comps, window_s=0.02, title_prefix="", filename=""):
    """Smoothed share of each of the eight components in the total dynamic
    energy, as lines on a log axis so tiny shares stay readable."""
    time = np.asarray(time, dtype=np.float64)
    window = _window_frames(time, window_s)
    sm = np.array([np.maximum(_smooth(c, window), 0.0) for c in comps])
    total = sm.sum(axis=0)
    frac = np.full_like(sm, np.nan)
    active = total > 1e-3 * total.max() if total.size else total > 0
    np.divide(sm, total, out=frac, where=active[None, :] & (sm > 0))

    fig, ax = plt.subplots(figsize=(12, 6))
    for y, (label, color, axial) in zip(frac, COMPONENTS):
        ax.plot(time, y, color=color, linewidth=1.0, linestyle='--' if axial else '-', label=label)
    ax.set_yscale('log')
    positive = frac[np.isfinite(frac) & (frac > 0)]
    lo = max(positive.min(), 1e-10) if positive.size else 1e-10
    ax.set_ylim(lo * 0.5, 1.5)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Share of dynamic energy")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"[{title_prefix}] Energy Share per Component (smoothed over {window_s * 1e3:.1f} ms)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_modal_energy(modal_ke, modal_time, title_prefix="", filename=""):
    """Kinetic energy per spatial mode over time (chunk-averaged, ~half of each
    mode's energy), transverse and longitudinal. Energy moving between modes
    shows up as brightness shifting between rows."""
    modal_ke = np.asarray(modal_ke, dtype=np.float64)        # (chunks, n_modes, 2)
    t = np.asarray(modal_time, dtype=np.float64)
    n = np.arange(1, modal_ke.shape[1] + 1)
    dt = (t[1] - t[0]) if t.size > 1 else 1e-2

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout='constrained')
    for ax, (idx, label) in zip(axes, ((0, "Transverse (Y+Z)"), (1, "Longitudinal (X)"))):
        e = modal_ke[:, :, idx]
        peak = e.max() if e.size and e.max() > 0 else 1.0
        db = 10.0 * np.log10(np.maximum(e / peak, 1e-8))
        # Only show modes that ever carry energy within 80 dB of the peak.
        active = np.nonzero(db.max(axis=0) > -80.0)[0]
        n_max = n[active[-1]] if active.size else n[-1]
        im = ax.imshow(db.T[:n_max], aspect='auto', origin='lower', interpolation='nearest',
                       cmap='viridis', vmin=-80, vmax=0,
                       extent=(t[0] - dt / 2, t[-1] + dt / 2, 0.5, n_max + 0.5))
        fig.colorbar(im, ax=ax, label="Mode KE (dB re. peak)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mode number n")
        ax.set_title(label)
    fig.suptitle(f"[{title_prefix}] Energy per Mode")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]
