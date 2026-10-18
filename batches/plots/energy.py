import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import maximum_filter1d


def _dynamic_above_baseline(x):
    return x - np.min(x)


# Categorical palette (validated colour-blind safe on a white surface, adjacent
# pairs). Every chart assigns slots in this fixed order, starting at slot 1.
# Aqua, yellow and magenta sit below 3:1 contrast on white, so charts using them
# carry a legend and direct labels.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948")
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID_KW = dict(color="#e1e0d9", linewidth=0.6)

# Energy colour code, used by every energy plot: one hue per family (palette
# slots 1-4 in order); inside a family the transverse part is solid and the
# axial part dashed, so a family line has the colour of its two parts. Sums
# across families are neutral ink.
FAMILIES = ("Translational KE", "Rotational KE", "Bend/Twist PE", "Shear/Stretch PE")
FAMILY_COLORS = dict(zip(FAMILIES, PALETTE[:4]))
SOLID, DASHED = "-", (0, (5, 2.5))
ENERGY_STYLE = {
    "Translational KE": dict(color=PALETTE[0], linestyle=SOLID),
    "Transverse KE": dict(color=PALETTE[0], linestyle=SOLID),
    "Longitudinal KE": dict(color=PALETTE[0], linestyle=DASHED),
    "Rotational KE": dict(color=PALETTE[1], linestyle=SOLID),
    "Tilt KE": dict(color=PALETTE[1], linestyle=SOLID),
    "Torsion KE": dict(color=PALETTE[1], linestyle=DASHED),
    "Bend/Twist PE": dict(color=PALETTE[2], linestyle=SOLID),
    "Bending PE": dict(color=PALETTE[2], linestyle=SOLID),
    "Torsion PE": dict(color=PALETTE[2], linestyle=DASHED),
    "Shear/Stretch PE": dict(color=PALETTE[3], linestyle=SOLID),
    "Shear PE": dict(color=PALETTE[3], linestyle=SOLID),
    "Axial PE": dict(color=PALETTE[3], linestyle=DASHED),
    "Total": dict(color=INK, linestyle=SOLID),
    "Kinetic": dict(color=INK_MUTED, linestyle=SOLID),
    "Potential": dict(color=INK_MUTED, linestyle=":"),
    "Longitudinal (KE + PE)": dict(color=INK, linestyle=DASHED),
}

# Polarity (power into / out of longitudinal motion): a diverging pair that
# uses no family hue, around the neutral grey midpoint.
INTO_LONG, OUT_OF_LONG, MIDPOINT = PALETTE[7], PALETTE[6], "#f0efec"
DIVERGING_CMAP = LinearSegmentedColormap.from_list("out_of__into_long",
                                                   [OUT_OF_LONG, MIDPOINT, INTO_LONG])


def _darken(hex_color, f=0.3):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return (r * f, g * f, b * f)


def family_cmap(color):
    """Single-hue sequential map for energy maps: white -> family hue -> dark."""
    return LinearSegmentedColormap.from_list(f"seq_{color}", ["#ffffff", color, _darken(color)])


TRANSLATIONAL_CMAP = family_cmap(PALETTE[0])
NEUTRAL_CMAP = family_cmap(INK_MUTED)


def _style_axis(ax):
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_energies(time, trans_ke, rot_ke, bend_pe, shear_pe,
                  normalized=False, mode_transfer=False, window_s=None,
                  title_prefix="", filename=""):
    """The four energy families: absolute (default), as a stacked share of the
    dynamic energy (normalized) or as share lines (mode_transfer).

    With window_s the families are smoothed over whole periods first, which
    removes the KE<->PE swing inside every mode and leaves the slow trends.
    """
    time = np.asarray(time, dtype=np.float64)
    fam = [np.asarray(trans_ke, np.float64), np.asarray(rot_ke, np.float64),
           _dynamic_above_baseline(np.asarray(bend_pe, np.float64)),
           _dynamic_above_baseline(np.asarray(shear_pe, np.float64))]
    if window_s:
        w = _window_frames(time, window_s)
        fam = [_smooth(f, w) for f in fam]
    smooth_note = f", smoothed over {window_s * 1e3:.1f} ms" if window_s else ""

    fig, ax = plt.subplots(figsize=(12, 6))
    if normalized or mode_transfer:
        total = np.sum(fam, axis=0)
        peak = np.max(total) if total.size else 0.0
        active = total > 1e-3 * peak if peak > 0 else np.zeros_like(total, dtype=bool)
        fracs = []
        for f in fam:
            out = np.full_like(f, 0.0 if normalized else np.nan)
            np.divide(f, total, out=out, where=active)
            fracs.append(out)
        if normalized:
            ax.stackplot(time, *fracs, labels=FAMILIES, colors=[FAMILY_COLORS[f] for f in FAMILIES],
                         edgecolor="white", linewidth=0.3)
            ax.set_ylim(0, 1)
            title = f"[{title_prefix}] Energy Distribution (stacked share{smooth_note})"
        else:
            for f, label in zip(fracs, FAMILIES):
                ax.plot(time, f, linewidth=1.2, label=label, **ENERGY_STYLE[label])
            ax.set_ylim(-0.02, 1.02)
            title = f"[{title_prefix}] Energy Share per Family{smooth_note}"
        ax.set_ylabel("Fraction of dynamic energy")
        ax.set_xlim(time[0], time[-1])
    else:
        for f, label in zip(fam, FAMILIES):
            ax.plot(time, f, linewidth=1.0, label=label, **ENERGY_STYLE[label])
        ax.set_ylabel("Energy (J)")
        title = f"[{title_prefix}] Energy per Family (offset removed{smooth_note})"

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    _style_axis(ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


# The eight energy components: (label, style, axial?), styled by ENERGY_STYLE
# (family hue; transverse part solid, axial part dashed). The potential parts
# are split by *strain component*, not by motion: with shear stiffness far
# above the tension, the frame follows the tangent and a transverse slope shows
# up as axial strain, so the transverse wave's potential energy lives in
# "Axial PE". "Torsion KE" (spin about the axis) and "Torsion PE" (twist
# curvature) are the kinetic and potential energy of the same deformation.
COMPONENTS = tuple((label, ENERGY_STYLE[label], axial) for label, axial in (
    ("Transverse KE", False),
    ("Longitudinal KE", True),
    ("Tilt KE", False),
    ("Torsion KE", True),
    ("Bending PE", False),
    ("Torsion PE", True),
    ("Shear PE", False),
    ("Axial PE", True),
))


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
                           window_s=None, title_prefix="", filename=""):
    """Fine-grained breakdown: each of the four energy families split into its
    transverse and axial parts (eight lines total). Only the kinetic split
    (v_x vs v_y, v_z) cleanly separates transverse from longitudinal motion;
    the potential split is by strain component (see COMPONENTS)."""
    time = np.asarray(time, dtype=np.float64)
    comps = energy_components(trans_ke, rot_ke, bend_pe, shear_pe,
                              trans_ke_long, rot_ke_tor, bend_pe_twist, shear_pe_ext)
    if window_s:
        w = _window_frames(time, window_s)
        comps = [_smooth(c, w) for c in comps]
    smooth_note = f", smoothed over {window_s * 1e3:.1f} ms" if window_s else ""

    fig, ax = plt.subplots(figsize=(12, 6))
    for y, (label, style, _) in zip(comps, COMPONENTS):
        ax.plot(time, y, linewidth=1.0, label=label, **style)

    ax.set_title(f"[{title_prefix}] Energy Component Breakdown (offset removed{smooth_note})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    _style_axis(ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_energy_totals(time, trans_ke, rot_ke, bend_pe, shear_pe,
                       title_prefix="", filename=""):
    """Totals only: kinetic, potential and total system energy."""
    trans_ke = np.asarray(trans_ke, dtype=np.float64)
    rot_ke = np.asarray(rot_ke, dtype=np.float64)
    total_kin = trans_ke + rot_ke
    total_pot = (_dynamic_above_baseline(np.asarray(bend_pe, dtype=np.float64))
                 + _dynamic_above_baseline(np.asarray(shear_pe, dtype=np.float64)))
    total_energy = total_kin + total_pot

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, total_kin, linewidth=0.8, label="Total kinetic", **ENERGY_STYLE["Kinetic"])
    ax.plot(time, total_pot, linewidth=0.8, label="Total potential", **ENERGY_STYLE["Potential"])
    ax.plot(time, total_energy, linewidth=2.0, zorder=5, label="Total system energy",
            **ENERGY_STYLE["Total"])
    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"[{title_prefix}] Total Energies (offset removed)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    _style_axis(ax)
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

    ax_top.plot(time, np.maximum(env_t, tiny), linewidth=1.2,
                label="Transversal KE (envelope)", **ENERGY_STYLE["Transverse KE"])
    ax_top.plot(time, np.maximum(env_l, tiny), linewidth=1.2,
                label="Longitudinal KE (envelope)", **ENERGY_STYLE["Longitudinal KE"])
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
    ax_bot.plot(time, ratio, color=INK_MUTED, linewidth=1.0)
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
    ax.plot(time, avg_t, label="\u27e8Transversal KE\u27e9", **ENERGY_STYLE["Transverse KE"])
    ax.set_ylabel("Transversal (J)")
    ax2 = ax.twinx()
    ax2.plot(time, avg_l, label="\u27e8Longitudinal KE\u27e9", **ENERGY_STYLE["Longitudinal KE"])
    ax2.set_ylabel("Longitudinal (J)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='upper right')
    ax.set_title(f"[{title_prefix}] Transversal \u2194 Longitudinal Kinetic Energy Transfer "
                 f"(smoothed over {window_s * 1e3:.1f} ms)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(time, share, color=INK_MUTED)
    ax.set_ylabel("Longitudinal share")
    ax.grid(True, alpha=0.3)

    # Powers differ by orders of magnitude, so each gets its own linear axis
    # (both centred on zero so the sign comparison reads directly).
    ax = axes[2]
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.plot(time, p_t, linewidth=1.0, label="d\u27e8E_trans\u27e9/dt", **ENERGY_STYLE["Transverse KE"])
    ax.set_ylabel("Transversal power (W)")
    ax2 = ax.twinx()
    ax2.plot(time, p_l, linewidth=1.0, label="d\u27e8E_long\u27e9/dt", **ENERGY_STYLE["Longitudinal KE"])
    # Longitudinal gains while transversal loses -> transfer trans -> long.
    ax2.fill_between(time, 0, p_l, where=(p_l > 0) & (p_t < 0), color=INTO_LONG,
                     alpha=0.25, label="trans \u2192 long")
    ax2.fill_between(time, 0, p_l, where=(p_l < 0) & (p_t > 0), color=OUT_OF_LONG,
                     alpha=0.25, label="long \u2192 trans")
    ax2.set_ylabel("Longitudinal power (W)")
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
    counts = counts.reshape((-1,) + (1,) * (a.ndim - 1))
    return (np.add.reduceat(a, starts, axis=0) / counts).astype(np.float32)


def block_centers(offset, n_frames, block, frame_dt):
    """Times (s) of the block_mean rows for a chunk starting at frame `offset`."""
    starts = np.arange(0, n_frames, block)
    counts = np.diff(np.append(starts, n_frames))
    return (offset + starts + 0.5 * (counts - 1)) * frame_dt


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
    for y, (label, style, _) in zip(frac, COMPONENTS):
        ax.plot(time, y, linewidth=1.2, label=label, **style)
    ax.set_yscale('log')
    positive = frac[np.isfinite(frac) & (frac > 0)]
    lo = max(positive.min(), 1e-10) if positive.size else 1e-10
    ax.set_ylim(lo * 0.5, 1.5)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Share of dynamic energy")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"[{title_prefix}] Energy Share per Component (smoothed over {window_s * 1e3:.1f} ms)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    _style_axis(ax)
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
                       cmap=TRANSLATIONAL_CMAP, vmin=-80, vmax=0,
                       extent=(t[0] - dt / 2, t[-1] + dt / 2, 0.5, n_max + 0.5))
        fig.colorbar(im, ax=ax, label="Mode KE (dB re. peak)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mode number n")
        ax.set_title(label)
    fig.suptitle(f"[{title_prefix}] Energy per Mode")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]
