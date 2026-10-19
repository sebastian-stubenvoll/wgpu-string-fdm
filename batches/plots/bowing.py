"""Bowed-string figures and the per-period regime classification.

Inputs come straight from the bow output (simulation_run b_*.pkl.gz): per frame the normal force
and bow velocity of the profile, and per contact whether it is stuck, the relative surface velocity
v_rel = v_surface - v_b, the friction force on the string and the contact temperature, all at the
frame's last solver step.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots.energy import PALETTE, INK, INK_MUTED, GRID_KW

BOW_COLUMNS = ("sticking", "slip_velocity", "force", "impulse", "bow_work", "friction_work",
               "normal_work", "temperature", "heat_power")


def bow_arrays(frames):
    """Bow output frames (list of (normal force, velocity, step, contacts)) as arrays:
    normal_force, velocity, step (F,) and every BOW_COLUMNS entry as (F, contacts)."""
    out = {"normal_force": np.array([f[0] for f in frames], np.float64),
           "velocity": np.array([f[1] for f in frames], np.float64),
           "step": np.array([f[2] for f in frames], np.int64)}
    for i, name in enumerate(BOW_COLUMNS):
        if i == 0:
            out[name] = np.array([[c[0] for c in f[3]] for f in frames], np.float64)
        else:
            out[name] = np.array([[c[1][i - 1] for c in f[3]] for f in frames], np.float64)
    return out


def slip_onsets(stuck):
    """Frame indices where the contact goes from stuck to slipping (stuck: (F,) bool)."""
    return np.nonzero(stuck[:-1] & ~stuck[1:])[0] + 1


def flybacks(surface_velocity, time, bow_velocity):
    """(start times, durations) of the flybacks: intervals in which the string surface at the bow moves
    backward (surface velocity < 0) and, at some instant, faster than the bow moves forward (minimum
    below -bow velocity). Marginal zero crossings of a creeping or partly slipping contact do not
    count. An interval still open at the end is left out."""
    back = surface_velocity < 0.0
    on = np.nonzero(~back[:-1] & back[1:])[0] + 1
    off = np.nonzero(back[:-1] & ~back[1:])[0] + 1
    t_on, dur = [], []
    for o in on:
        later = off[off > o]
        if len(later) and np.min(surface_velocity[o:later[0]] + bow_velocity[o:later[0]]) < 0.0:
            t_on.append(time[o])
            dur.append(time[later[0]] - time[o])
    return np.asarray(t_on), np.asarray(dur)


def regime(time, surface_velocity, bow_velocity, stuck, period, last_periods=20):
    """Classify the motion over `last_periods` whole periods ending one period before the end of the run.

    A slip in this classification is a flyback (see flybacks): the string surface at the bow moving
    backward, faster than the bow moves forward. Helmholtz motion has one per period. Brief re-slips
    while the contact is captured, and creep at the friction limit (thermal model), do not count. surface_velocity: (F,) the contact's (mean) surface velocity; stuck: (F,) bool, all contact
    nodes stuck (reported as the stuck fraction only). Regimes:
      "Helmholtz"            one flyback in every period
      "multiple slip (n)"    n > 1 flybacks in every period
      "continuous slip"      never stuck and no flyback
      "irregular"            anything else
    """
    t_end = time[-1] - period
    edges = t_end - period * np.arange(last_periods, -1, -1)
    edges = edges[edges >= time[0]]
    t_on, dur = flybacks(surface_velocity, time, bow_velocity)
    counts = np.histogram(t_on, bins=edges)[0] if len(edges) > 1 else np.array([], int)
    window = (time >= edges[0]) & (time < edges[-1])
    stuck_fraction = float(np.mean(stuck[window]))
    if stuck_fraction == 0.0 and (len(counts) == 0 or counts.sum() == 0):
        name = "continuous slip"
    elif len(counts) and np.all(counts == 1):
        name = "Helmholtz"
    elif len(counts) and np.all(counts == counts[0]) and counts[0] > 1:
        name = f"multiple slip ({counts[0]})"
    else:
        name = "irregular"
    inside = (t_on >= edges[0]) & (t_on < edges[-1])
    return {"regime": name, "flybacks_per_period": counts.tolist(), "stuck_fraction": stuck_fraction,
            "periods": len(counts),
            "flyback_duration": float(np.median(dur[inside])) if np.any(inside) else None}


def _axes(figsize=(12, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, **GRID_KW)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def _save(fig, ax, title, xlabel, ylabel, filename):
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_time_series(time, value, title, ylabel, filename, reference=None, reference_label=None):
    """One quantity against time (ms); optional horizontal reference line."""
    fig, ax = _axes()
    ax.plot(time * 1e3, value, color=PALETTE[0], lw=1.0)
    if reference is not None:
        ax.axhline(reference, color=INK_MUTED, lw=0.8, ls=":")
        if reference_label:
            ax.text(time[-1] * 1e3, reference, f" {reference_label}", va="center", ha="left", color=INK_MUTED)
    ax.set_xlim(time[0] * 1e3, time[-1] * 1e3)
    return _save(fig, ax, title, "Time (ms)", ylabel, filename)


def plot_friction_plane(slip_velocity, mu_eff, title, filename, curve=None, curve_label=None):
    """Friction coefficient |F| / N against the relative velocity (points, one per frame), with the
    model's sliding relation (curve: (v, mu))."""
    fig, ax = _axes(figsize=(7, 5.5))
    ax.plot(slip_velocity, mu_eff, ".", ms=1.5, color=PALETTE[0], alpha=0.5)
    if curve is not None:
        ax.plot(curve[0], curve[1], color=INK_MUTED, lw=1.0, ls=(0, (5, 2.5)))
        if curve_label:
            ax.annotate(curve_label, (curve[0][-1], curve[1][-1]), xytext=(0, 8), textcoords="offset points",
                        va="bottom", ha="right", color=INK_MUTED)
    ax.set_ylim(bottom=0)
    return _save(fig, ax, title, "Relative velocity (m/s)", "Friction coefficient", filename)


def plot_regime(time, surface_velocity, bow_velocity, stuck, period, title, filename):
    """Flybacks (backward motion of the surface at the bow) per period over the whole run (bars) and
    the stuck fraction (line)."""
    edges = np.arange(time[0], time[-1] + 1e-12, period)
    on = flybacks(surface_velocity, time, bow_velocity)[0]
    counts = np.histogram(on, bins=edges)[0]
    frac = np.array([np.mean(stuck[(time >= a) & (time < b)]) for a, b in zip(edges[:-1], edges[1:])])
    centers = 0.5 * (edges[:-1] + edges[1:]) * 1e3
    fig, ax = _axes()
    ax.bar(centers, counts, width=period * 1e3 * 0.8, color=PALETTE[0], label="Flybacks per period")
    ax2 = ax.twinx()
    ax2.plot(centers, frac, color=PALETTE[1], lw=1.2, label="Stuck fraction")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Stuck fraction", color=INK)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False)
    ax.set_title(title, loc="left", color=INK, pad=28)
    ax.set_xlabel("Time (ms)", color=INK)
    ax.set_ylabel("Flybacks per period", color=INK)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_string_shapes(x, shapes, phases, title, filename):
    """Transverse displacement along the string (mm) at instants through one period: shapes (K, N) in m,
    phases (K,) in [0, 1); darker = later. The Helmholtz corner travels along the string."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("phase", ["#a9c6ef", PALETTE[0], "#0f2f63"])
    fig, ax = _axes(figsize=(12, 4.5))
    for s, p in zip(shapes, phases):
        ax.plot(x * 1e3, s * 1e3, color=cmap(p), lw=1.2, label=f"{p:.2f} T")
    ax.set_xlim(x[0] * 1e3, x[-1] * 1e3)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(phases))
    ax.set_title(title, loc="left", color=INK, pad=28)
    ax.set_xlabel("Position from the bridge end (mm)", color=INK)
    ax.set_ylabel("Displacement (mm)", color=INK)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return filename
