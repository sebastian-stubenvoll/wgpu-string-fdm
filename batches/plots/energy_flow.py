"""Energy flow between transverse and longitudinal motion.

The simulation's axial energy per edge is 0.5 K_z eta(s)^2 s^2 dl with strain
s = |r|/dl - 1 and eta = 1 - 0.3 s. Expanding it around the pre-stretched rest
state (spacing h = L/(N-1), rest strain s0 = h/dl - 1) gives, per edge with
displacement differences du = u_{e+1} - u_e and s_e = |du_perp|^2 / (2h):

    E ~ E0 + T0 du_x + T0 s_e + 0.5 (K_ax/dl) du_x^2 + (K_ax/dl) (du_x s_e + 0.5 s_e^2)
              ^ sums to 0     ^ transverse  ^ longitudinal        ^ coupling

    T0   = K_z eta0 s0 (eta0 - 0.3 s0)              (static tension)
    K_ax = K_z (eta0^2 - 1.2 eta0 s0 + 0.09 s0^2)   (axial stiffness)

The first-order term T0 du_x moves energy between neighbouring elements but sums
to zero over the string (fixed ends), so it is dropped; left in, it dominates
every element's energy and hides everything else. All remaining terms are
quadratic and zero at rest, so no baseline is needed, and the transverse and
longitudinal energies each contain their own kinetic *and* potential parts, so
they don't swing at twice the mode frequencies.

The coupling force on node j along x is (K_ax/dl)(s_j - s_{j-1}); the power it
delivers, p_j = v_x,j * that force, is the local rate at which transverse
motion pumps energy into longitudinal motion.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

from plots.energy import PALETTE, INK, INK_MUTED, _style_axis, _smooth, _window_frames

# Colours follow the subsystem; a transfer is coloured by where the energy goes.
C_TRANS, C_LONG, C_TOR = PALETTE[0], PALETTE[1], PALETTE[2]
TRANSFER_CMAP = LinearSegmentedColormap.from_list(
    "long_to_trans__trans_to_long", [C_TRANS, "#f0efec", C_LONG])


def string_constants(spacing, dl, K_z):
    """Static tension T0 and axial stiffness K_ax of the pre-stretched string."""
    s0 = spacing / dl - 1.0
    eta0 = 1.0 - 0.3 * s0
    T0 = K_z * eta0 * s0 * (eta0 - 0.3 * s0)
    K_ax = K_z * (eta0 ** 2 - 1.2 * eta0 * s0 + 0.09 * s0 ** 2)
    return T0, K_ax


def edges_to_nodes(edge_vals):
    """Split per-edge values (F, N-1) half/half onto their two nodes (F, N)."""
    F, E = edge_vals.shape
    out = np.zeros((F, E + 1))
    out[:, :-1] += 0.5 * edge_vals
    out[:, 1:] += 0.5 * edge_vals
    return out


def motion_decomposition(pos, vel, spacing, dl, m_node, K_z):
    """Transverse / longitudinal / coupling energies and transfer power.

    pos, vel: (frames, nodes, 3) displacement and velocity, x longitudinal.
    Returns per-node kinetic parts (ke_t, ke_l; ends weighted 0.5), per-edge
    potential parts (pe_t, pe_l, coupling) and the per-node transfer power
    p (W, transverse -> longitudinal positive).
    """
    T0, K_ax = string_constants(spacing, dl, K_z)
    k = K_ax / dl
    v = np.asarray(vel, dtype=np.float64)
    du = np.diff(np.asarray(pos, dtype=np.float64), axis=1)          # (F, N-1, 3)
    dux = du[..., 0]
    s = (du[..., 1] ** 2 + du[..., 2] ** 2) / (2.0 * spacing)         # (F, N-1)

    w = np.ones(v.shape[1])
    w[[0, -1]] = 0.5
    ke_t = 0.5 * m_node * (v[..., 1] ** 2 + v[..., 2] ** 2) * w
    ke_l = 0.5 * m_node * v[..., 0] ** 2 * w

    pe_t = T0 * s
    pe_l = 0.5 * k * dux ** 2
    coupling = k * (dux * s + 0.5 * s ** 2)

    force_x = np.zeros_like(ke_l)
    force_x[:, 1:-1] = k * (s[:, 1:] - s[:, :-1])
    p = v[..., 0] * force_x
    return {"ke_t": ke_t, "ke_l": ke_l, "pe_t": pe_t, "pe_l": pe_l,
            "coupling": coupling, "p": p}


def _direct_label(ax, t, y, text, color):
    """Label a line at its right end: coloured dot + text in ink."""
    finite = np.nonzero(np.isfinite(y) & (y > 0))[0]
    if not finite.size:
        return
    i = finite[-1]
    ax.plot([t[i]], [y[i]], marker='o', markersize=5, color=color, zorder=6)
    ax.annotate(text, (t[i], y[i]), xytext=(6, 0), textcoords='offset points',
                va='center', fontsize=9, color=INK)


def plot_energy_flow(time, flow, window_s=None, onset=None, period=None,
                     title_prefix="", filename=""):
    """Energy in transverse, longitudinal and torsional motion, the transfer
    power between transverse and longitudinal, and a consistency check.

    flow: dict of per-frame arrays E_T, E_L, E_tor, P_TL, D_L (phase slice).
    window_s smooths the transfer power (long phases); onset/period give the
    initial transfer rate over the first period after the strike.
    """
    t = np.asarray(time, dtype=np.float64)
    e_t = np.asarray(flow["E_T"], np.float64)
    e_l = np.asarray(flow["E_L"], np.float64)
    e_tor = np.asarray(flow["E_tor"], np.float64)
    p_raw = np.asarray(flow["P_TL"], np.float64)
    d_l = np.asarray(flow["D_L"], np.float64)
    p = _smooth(p_raw, _window_frames(t, window_s)) if window_s else p_raw

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2]})

    # 1. Energy per kind of motion.
    ax = axes[0]
    series = (("Transverse", C_TRANS, e_t), ("Longitudinal", C_LONG, e_l), ("Torsional", C_TOR, e_tor))
    for label, color, y in series:
        ax.plot(t, np.where(y > 0, y, np.nan), color=color, linewidth=1.4, label=label)
        _direct_label(ax, t, y, label, color)
    ax.set_yscale('log')
    peak = max(np.nanmax(e_t), np.nanmax(e_l), 1e-30)
    ax.set_ylim(peak * 1e-6, peak * 3)
    ax.set_ylabel("Energy (J)")
    ax.set_title("Energy per kind of motion", fontsize=10, loc='left')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=9)
    _style_axis(ax)
    if onset is not None:
        for a in axes:
            a.axvline(onset, color=INK_MUTED, linestyle=':', linewidth=1.0)

    # 2. Transfer power, filled by where the energy goes.
    ax = axes[1]
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.8)
    ax.fill_between(t, 0, p, where=p > 0, color=C_LONG, alpha=0.55, linewidth=0,
                    label="transverse → longitudinal")
    ax.fill_between(t, 0, p, where=p < 0, color=C_TRANS, alpha=0.55, linewidth=0,
                    label="longitudinal → transverse")
    ax.plot(t, p, color=INK, linewidth=0.6)
    ax.set_ylabel("Transfer power (W)")
    sub = "Transfer power transverse → longitudinal"
    if window_s:
        sub += f" (smoothed over {window_s * 1e3:.1f} ms)"
    if onset is not None and period:
        sel = (t >= onset) & (t <= onset + period)
        if sel.sum() > 1:
            rate = np.trapezoid(p_raw[sel], t[sel]) / (t[sel][-1] - t[sel][0])
            sub += f";  mean over first period after strike: {rate * 1e3:.3g} mW"
    ax.set_title(sub, fontsize=10, loc='left')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=9)
    _style_axis(ax)

    # 3. Consistency: does the transfer explain the longitudinal energy?
    ax = axes[2]
    dt = np.diff(t, prepend=t[0])
    transferred = np.cumsum((p_raw - d_l) * dt)
    ax.plot(t, e_l - e_l[0], color=INK, linewidth=1.4, label="change of longitudinal energy")
    ax.plot(t, transferred, color=C_LONG, linewidth=1.2, linestyle='--',
            label="∫(transfer power − longitudinal damping) dt")
    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Check: transfer accounts for the longitudinal energy", fontsize=10, loc='left')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=9)
    _style_axis(ax)

    fig.suptitle(f"[{title_prefix}] Energy Flow between Kinds of Motion")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


MAP_KINDS = {
    "trans": ("Transverse energy per element (J)", "Blues", "Transverse Energy along the String"),
    "long": ("Longitudinal energy per element (J)", "Oranges", "Longitudinal Energy along the String"),
    "transfer": ("Transfer power per element (W)\n← into transverse | into longitudinal →",
                 TRANSFER_CMAP, "Local Transfer Transverse → Longitudinal"),
}


def plot_element_map(time, values, length, kind, onset=None, title_prefix="", filename=""):
    """Position x time map of one per-element quantity (block-averaged rows).

    kind: 'trans' / 'long' (energy, log colour scale) or 'transfer' (power,
    diverging, symmetric limits at the 99th percentile of |p|).
    """
    label, cmap, title = MAP_KINDS[kind]
    v = np.asarray(values, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    n = v.shape[1]
    x = np.linspace(0.0, length, n)
    dx = x[1] - x[0]
    dt = (t[1] - t[0]) if t.size > 1 else 1e-3
    extent = (x[0] - dx / 2, x[-1] + dx / 2, t[0] - dt / 2, t[-1] + dt / 2)

    fig, ax = plt.subplots(figsize=(12, 7))
    if kind == "transfer":
        lim = np.percentile(np.abs(v), 99) if v.size else 1.0
        lim = lim if lim > 0 else 1.0
        im = ax.imshow(v, aspect='auto', origin='lower', interpolation='nearest',
                       cmap=cmap, vmin=-lim, vmax=lim, extent=extent)
    else:
        peak = float(np.max(v)) if v.size and np.max(v) > 0 else 1.0
        im = ax.imshow(np.maximum(v, peak * 1e-4), aspect='auto', origin='lower',
                       interpolation='nearest', cmap=cmap,
                       norm=LogNorm(vmin=peak * 1e-4, vmax=peak), extent=extent)
    fig.colorbar(im, ax=ax, label=label)
    if onset is not None and t[0] <= onset <= t[-1]:
        ax.axhline(onset, color=INK, linestyle=':', linewidth=1.0)
        ax.annotate("strike onset", (x[-1], onset), xytext=(-4, 3), textcoords='offset points',
                    ha='right', fontsize=8, color=INK)
    ax.set_xlabel("Position along string (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"[{title_prefix}] {title}")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]
