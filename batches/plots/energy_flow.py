"""Energy flow between transverse and longitudinal motion, from the shader's
own force law (src/shaders/cosserat.wgsl, compute_internals / compute_forces).

Edge force (lab frame), reference strain 0:

    f_e = R(q_e) K_se eta(e)^2 sigma_e,   sigma_e = R(q_e)^T r_e / dl - e3,
    r_e = ref_e + u_{e+1} - u_e,          e = |r_e| / dl,  eta = 1 - 0.3 (e - 1)

and node force F_j = f_j - f_{j-1} - c_lin v_j (+ hammer, z only). Everything
below is evaluated from the saved state with exactly this law; no expansion.

Definition (the only choice made): the longitudinal energy is the energy the
string would have if transverse and rotational motion were switched off,

    E_L = sum 1/2 m v_x^2 + U_L(u_x),

where U_L is the exact potential of the shader's force in that state (edges
along x at their rest orientation): f = K_z eta(e)^2 (e - 1) x, so
U_L = K_z dl g(e - 1) per edge with g(s) = s^2/2 - 0.2 s^3 + 0.0225 s^4.
Then, exactly by the force law,

    dE_L/dt = P_c - D_L,   P_c = sum_j v_x,j (F_x,j(actual) - F_x,j(longitudinal only)),

the transfer power from transverse/rotational motion into longitudinal motion.
It is split, again exactly (telescoping), into the effect of the transverse
displacement (actual displacement, rest orientations, minus longitudinal only)
and of the edge rotation/shear (actual minus actual displacement at rest
orientations). Transverse potential energy is never needed: the shader's force
isn't the gradient of any energy once shear and the eta scaling combine, so it
has no exact definition. Any gap between dE_L/dt and P_c - D_L in the saved data
is what the force law doesn't explain: time sampling, float32 rounding on the
GPU, or the integrator.

U_L per edge has a first-order part K_z dl g'(s0) (s - s0) (tension doing work
as an edge moves) that sums to exactly zero over the string when all rest
spacings are equal and the ends are fixed. It is kept in the totals and left
out of the per-element maps, where it would otherwise dominate.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, SymLogNorm
from scipy.fft import dst

from plots.energy import PALETTE, INK, INK_MUTED, _style_axis, _smooth, _window_frames

# Colours follow the subsystem; a transfer is coloured by where the energy goes.
C_TRANS, C_LONG, C_ROT, C_TOR = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]
TRANSFER_CMAP = LinearSegmentedColormap.from_list(
    "long_to_trans__trans_to_long", [C_TRANS, "#f0efec", C_LONG])

N_TRANSFER_MODES = 256
_E3 = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# Shader port (quaternions are (x, y, z, w), as in the WGSL)
# --------------------------------------------------------------------------

def _qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def _qinv(q):
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)


def _as_quat(v):
    return np.concatenate([v, np.zeros(v.shape[:-1] + (1,), dtype=v.dtype)], axis=-1)


def rotate(q, v):
    """Material -> lab frame (WGSL rotate)."""
    return _qmul(_qmul(q, _as_quat(v)), _qinv(q))[..., :3]


def rotate_inv(q, v):
    """Lab -> material frame (WGSL rotate_inv)."""
    return _qmul(_qmul(_qinv(q), _as_quat(v)), q)[..., :3]


def edge_forces(r, quats, dl, K_se, dtype=np.float64):
    """Internal edge forces f_e (lab frame) exactly as compute_internals, for
    edge vectors r (..., 3) and orientations quats (..., 4). dtype=float32
    reproduces the shader's precision (up to operation order and FMA)."""
    r = np.asarray(r, dtype=dtype)
    q = np.asarray(quats, dtype=dtype)
    one = dtype(1.0)
    dl_ = dtype(dl)
    dl_inv = one / dl_
    tangent_MF = rotate_inv(q, r * dl_inv)
    sigma = tangent_MF - _E3.astype(dtype)
    len_inv = one / np.sqrt(np.sum(r * r, axis=-1))
    e = one / (len_inv * dl_)
    eta = one - dtype(0.3) * (e - one)
    K = np.asarray(K_se, dtype=dtype)
    return rotate(q, K * (eta * eta)[..., None] * sigma)


def node_forces(f):
    """F_j = f_j - f_{j-1} on interior nodes; zero at the fixed ends."""
    out = np.zeros(f.shape[:-2] + (f.shape[-2] + 1, 3), dtype=f.dtype)
    out[..., 1:-1, :] = f[..., 1:, :] - f[..., :-1, :]
    return out


def _g_derivs(s0):
    """g', g'', g''', g'''' at s0 for g(s) = s^2/2 - 0.2 s^3 + 0.0225 s^4."""
    return (s0 - 0.6 * s0 ** 2 + 0.09 * s0 ** 3, 1.0 - 1.2 * s0 + 0.27 * s0 ** 2,
            -1.2 + 0.54 * s0, 0.54)


def edges_to_nodes(edge_vals):
    """Split per-edge values (F, N-1) half/half onto their two nodes (F, N)."""
    F, E = edge_vals.shape
    out = np.zeros((F, E + 1))
    out[:, :-1] += 0.5 * edge_vals
    out[:, 1:] += 0.5 * edge_vals
    return out


def dst_parseval(n_interior):
    """sum_j x_j y_j = sum_n X_n Y_n * this, for X = dst(x, type=1)."""
    return 1.0 / (2.0 * (n_interior + 1))


def exact_flow(pos, vel, quats, ref, q_rest, dl, m_node, K_se, c_lin=0.0, block=64):
    """Longitudinal energy and the transfer power into it, per node/edge.

    pos, vel: (F, N, 3) displacement and velocity (x longitudinal); quats:
    (F, N-1, 4) edge orientations; ref: (N-1, 3) edge reference vectors;
    q_rest: (N-1, 4) rest orientations. Returns float64 arrays:
      ke_t, ke_l     (F, N)    kinetic energy, transverse / longitudinal
      u_l_quad       (F, N-1)  longitudinal potential above first order
      u_l_lin        (F, N-1)  its first-order part (sums to ~0)
      p_disp, p_rot  (F, N)    transfer power by channel (W)
      f_c            (F, N)    coupling force along x (N)
      d_l            (F, N)    longitudinal damping power (W)
    """
    F, N = pos.shape[:2]
    ref = np.asarray(ref, dtype=np.float64)
    q_rest = np.asarray(q_rest, dtype=np.float64)
    K_z = float(np.ravel(K_se)[2])
    s0 = ref[:, 0] / dl - 1.0
    g1, g2, g3, g4 = _g_derivs(s0)

    w = np.ones(N)
    w[[0, -1]] = 0.5
    out = {k: np.zeros((F, N)) for k in ("ke_t", "ke_l", "p_disp", "p_rot", "f_c", "d_l")}
    out["u_l_quad"] = np.zeros((F, N - 1))
    out["u_l_lin"] = np.zeros((F, N - 1))

    for a in range(0, F, block):
        b = min(F, a + block)
        u = np.asarray(pos[a:b], dtype=np.float64)
        v = np.asarray(vel[a:b], dtype=np.float64)
        q = np.asarray(quats[a:b], dtype=np.float64)
        du = u[:, 1:] - u[:, :-1]
        r = ref + du

        out["ke_t"][a:b] = 0.5 * m_node * (v[..., 1] ** 2 + v[..., 2] ** 2) * w
        out["ke_l"][a:b] = 0.5 * m_node * v[..., 0] ** 2 * w
        out["d_l"][a:b, 1:-1] = c_lin * v[:, 1:-1, 0] ** 2

        # Longitudinal-only state: closed form of the shader force for an
        # edge along x at its rest orientation, and its potential (Taylor
        # series of the quartic g around s0 is exact; no cancellation).
        d = du[..., 0] / dl                                   # s - s0
        e_c = 1.0 + s0 + d
        f_c_edge = K_z * (1.0 - 0.3 * (e_c - 1.0)) ** 2 * (e_c - 1.0)
        out["u_l_lin"][a:b] = K_z * dl * g1 * d
        out["u_l_quad"][a:b] = K_z * dl * (0.5 * g2 * d ** 2 + g3 * d ** 3 / 6.0 + g4 * d ** 4 / 24.0)

        Fx_C = np.zeros((b - a, N))
        Fx_C[:, 1:-1] = f_c_edge[:, 1:] - f_c_edge[:, :-1]
        Fx_B = node_forces(edge_forces(r, np.broadcast_to(q_rest, q.shape), dl, K_se))[..., 0]
        Fx_A = node_forces(edge_forces(r, q, dl, K_se))[..., 0]

        out["p_disp"][a:b] = v[..., 0] * (Fx_B - Fx_C)
        out["p_rot"][a:b] = v[..., 0] * (Fx_A - Fx_B)
        out["f_c"][a:b] = Fx_A - Fx_C
    return out


def roundoff_forces(pos, quats, ref, dl, K_se, block=64):
    """Node force in float32 (shader precision) minus the same in float64,
    from the same float32 state: (F, N, 3). The edge vector is formed in
    float32 like the shader (ref + (u_{j+1} - u_j))."""
    F, N = pos.shape[:2]
    out = np.zeros((F, N, 3))
    ref32 = np.asarray(ref, dtype=np.float32)
    for a in range(0, F, block):
        b = min(F, a + block)
        u = np.asarray(pos[a:b], dtype=np.float32)
        r32 = ref32 + (u[:, 1:] - u[:, :-1])
        q = np.asarray(quats[a:b], dtype=np.float32)
        F32 = node_forces(edge_forces(r32, q, dl, K_se, dtype=np.float32)).astype(np.float64)
        F64 = node_forces(edge_forces(r32.astype(np.float64), q.astype(np.float64), dl, K_se))
        out[a:b] = F32 - F64
    return out


def modal_transfer(v_x, f_c, n_modes=N_TRANSFER_MODES):
    """Exact split of P_c over the sine basis of the interior nodes (DST-I):
    P_c = sum_n Vdot_n Fc_n * c. Returns (P_n for the first n_modes, the
    remainder so that rows sum to P_c, and the coefficients for the
    cross-spectrum)."""
    V = dst(np.asarray(v_x, np.float64)[:, 1:-1], type=1, axis=1)
    Fc = dst(np.asarray(f_c, np.float64)[:, 1:-1], type=1, axis=1)
    c = dst_parseval(V.shape[1])
    p_n = V[:, :n_modes] * Fc[:, :n_modes] * c
    total = np.sum(np.asarray(v_x, np.float64) * np.asarray(f_c, np.float64), axis=1)
    rest = total - p_n.sum(axis=1)
    coeffs = np.stack([V[:, :n_modes], Fc[:, :n_modes]], axis=-1).astype(np.float32)
    return np.concatenate([p_n, rest[:, None]], axis=1), coeffs


def combination_frequencies(f_res, n_max=12):
    """2 f_n, f_m + f_n and |f_m - f_n| of the first n_max measured resonances."""
    f = np.asarray(f_res, dtype=np.float64)[:n_max]
    if f.size == 0:
        return np.zeros(0)
    s = np.add.outer(f, f)
    d = np.abs(np.subtract.outer(f, f))
    iu = np.triu_indices(f.size)
    comb = np.concatenate([s[iu], d[np.triu_indices(f.size, 1)]])
    return np.unique(comb[comb > 0])


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _direct_label(ax, t, y, text, color):
    """Label a line at its right end: coloured dot + text in ink."""
    finite = np.nonzero(np.isfinite(y) & (y > 0))[0]
    if not finite.size:
        return
    i = finite[-1]
    ax.plot([t[i]], [y[i]], marker='o', markersize=5, color=color, zorder=6)
    ax.annotate(text, (t[i], y[i]), xytext=(6, 0), textcoords='offset points',
                va='center', fontsize=9, color=INK)


def _legend_above(ax, ncol=4):
    """Legend directly above the panel, the (left) title lifted above it."""
    handles, labels = ax.get_legend_handles_labels()
    rows = -(-len(labels) // ncol) if labels else 0
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=ncol, frameon=False, fontsize=9)
    title = ax.get_title(loc='left')
    if title:
        ax.set_title(title, loc='left', fontsize=10, pad=8 + 15 * rows)


def _cumulative(y, t):
    """Trapezoid running integral starting at 0."""
    out = np.zeros_like(y, dtype=np.float64)
    if len(t) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def plot_energy_flow(time, flow, window_s=None, onset=None, period=None,
                     title_prefix="", filename=""):
    """How much energy goes from transverse/rotational into longitudinal motion,
    and when. flow: per-frame KE_T, KE_R, KE_tor, E_L, P_disp, P_rot, D_L.
    window_s smooths kinetic energies and the transfer power (long phases)."""
    t = np.asarray(time, dtype=np.float64)
    get = lambda k: np.asarray(flow[k], np.float64)
    e_l, d_l = get("E_L"), get("D_L")
    p_disp, p_rot = get("P_disp"), get("P_rot")
    p_raw = p_disp + p_rot
    sm = (lambda y: _smooth(y, _window_frames(t, window_s))) if window_s else (lambda y: y)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2]})

    # 1. Energies (exact): longitudinal total, the others kinetic only.
    ax = axes[0]
    series = (("Longitudinal (kinetic + potential)", C_LONG, e_l),
              ("Transverse kinetic", C_TRANS, sm(get("KE_T"))),
              ("Rotational kinetic (tilt)", C_ROT, sm(get("KE_R"))),
              ("Torsional kinetic", C_TOR, sm(get("KE_tor"))))
    for label, color, y in series:
        ax.plot(t, np.where(y > 0, y, np.nan), color=color, linewidth=1.4, label=label)
    _direct_label(ax, t, series[0][2], "Longitudinal", C_LONG)
    _direct_label(ax, t, series[1][2], "Transverse KE", C_TRANS)
    ax.set_yscale('log')
    peak = max(np.nanmax(e_l), np.nanmax(series[1][2]), 1e-30)
    ax.set_ylim(peak * 1e-6, peak * 3)
    ax.set_ylabel("Energy (J)")
    ax.set_title("Energy per kind of motion" + (f" (kinetic smoothed over {window_s * 1e3:.1f} ms)"
                                                 if window_s else ""), fontsize=10, loc='left')
    _legend_above(ax, ncol=2)
    _style_axis(ax)
    if onset is not None:
        for a in axes:
            a.axvline(onset, color=INK_MUTED, linestyle=':', linewidth=1.0)

    # 2. Transfer power, filled by where the energy goes.
    ax = axes[1]
    p = sm(p_raw)
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.8)
    ax.fill_between(t, 0, p, where=p > 0, color=C_LONG, alpha=0.55, linewidth=0,
                    label="into longitudinal")
    ax.fill_between(t, 0, p, where=p < 0, color=C_TRANS, alpha=0.55, linewidth=0,
                    label="back into transverse / rotation")
    ax.plot(t, p, color=INK, linewidth=0.6)
    ax.set_ylabel("Transfer power (W)")
    sub = "Transfer power into longitudinal motion"
    if window_s:
        sub += f" (smoothed over {window_s * 1e3:.1f} ms)"
    if onset is not None and period:
        sel = (t >= onset) & (t <= onset + period)
        if sel.sum() > 1:
            rate = np.trapezoid(p_raw[sel], t[sel]) / (t[sel][-1] - t[sel][0])
            sub += f";  mean over first period after strike: {rate * 1e3:.3g} mW"
    ax.set_title(sub, fontsize=10, loc='left')
    _legend_above(ax, ncol=2)
    _style_axis(ax)

    # 3. Cumulative energy delivered, by channel, against what E_L did.
    ax = axes[2]
    c_disp, c_rot = _cumulative(p_disp, t), _cumulative(p_rot, t)
    ax.plot(t, e_l - e_l[0] + _cumulative(d_l, t), color=INK, linewidth=1.6,
            label="ΔE_L + ∫ damping (what happened)")
    ax.plot(t, c_disp + c_rot, color=C_LONG, linewidth=1.2, linestyle='--',
            label="∫ transfer (what the force law delivers)")
    ax.plot(t, c_disp, color=PALETTE[4], linewidth=1.0, label="via transverse displacement")
    ax.plot(t, c_rot, color=PALETTE[6], linewidth=1.0, label="via rotation / shear")
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.6)
    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Energy delivered to longitudinal motion\n"
                 "(a gap black ↔ dashed is not explained by the force law: sampling, float32, integrator)",
                 fontsize=10, loc='left')
    _legend_above(ax, ncol=2)
    _style_axis(ax)

    fig.suptitle(f"[{title_prefix}] Energy Flow into Longitudinal Motion")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


MAP_KINDS = {
    "trans": ("Transverse kinetic energy per node (J)", "Blues", "Transverse Kinetic Energy along the String"),
    "long": ("Longitudinal energy per node (J)", "Oranges", "Longitudinal Energy along the String"),
    "transfer": ("Transfer power per node (W)\n← into transverse | into longitudinal →",
                 TRANSFER_CMAP, "Local Transfer into Longitudinal Motion"),
}


def _extent(x, t):
    dx = x[1] - x[0] if x.size > 1 else 1.0
    dt = (t[1] - t[0]) if t.size > 1 else 1e-3
    return (x[0] - dx / 2, x[-1] + dx / 2, t[0] - dt / 2, t[-1] + dt / 2)


def plot_element_map(time, values, length, kind, onset=None, title_prefix="", filename=""):
    """Position x time map of one per-node quantity (block-averaged rows).

    kind: 'trans' / 'long' (energy, log colour scale) or 'transfer' (power,
    diverging, symmetric limits at the 99th percentile of |p|, plus the
    time-integrated transfer per position underneath).
    """
    label, cmap, title = MAP_KINDS[kind]
    v = np.asarray(values, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    x = np.linspace(0.0, length, v.shape[1])
    extent = _extent(x, t)

    if kind == "transfer":
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                      gridspec_kw={"height_ratios": [4, 1.3]})
    else:
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
    fig.colorbar(im, ax=ax, label=label, pad=0.02)
    if onset is not None and t[0] <= onset <= t[-1]:
        ax.axhline(onset, color=INK, linestyle=':', linewidth=1.0)
        ax.annotate("strike onset", (x[-1], onset), xytext=(-4, 3), textcoords='offset points',
                    ha='right', fontsize=8, color=INK)
    ax.set_ylabel("Time (s)")
    ax.set_title(f"[{title_prefix}] {title}")

    if kind == "transfer":
        dt = (t[1] - t[0]) if t.size > 1 else 0.0
        e = v.sum(axis=0) * dt                                # block means x block length
        ax2.fill_between(x, 0, e, where=e > 0, color=C_LONG, alpha=0.6, linewidth=0, step='mid')
        ax2.fill_between(x, 0, e, where=e < 0, color=C_TRANS, alpha=0.6, linewidth=0, step='mid')
        ax2.plot(x, e, color=INK, linewidth=0.6, drawstyle='steps-mid')
        ax2.axhline(0.0, color=INK_MUTED, linewidth=0.6)
        ax2.set_ylabel("∫ transfer dt (J)")
        ax2.set_title("Net energy delivered to longitudinal motion at each position over this window",
                      fontsize=9, loc='left')
        _style_axis(ax2)
        # Keep the line panel as wide as the map (the map has a colorbar).
        fig.colorbar(im, ax=ax2, pad=0.02).ax.set_visible(False)
        ax2.set_xlabel("Position along string (m)")
    else:
        ax.set_xlabel("Position along string (m)")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def _symlog_map(ax, fig, z, x, y, label):
    z = np.asarray(z, np.float64)
    lim = np.percentile(np.abs(z), 99.5) if z.size else 1.0
    lim = lim if lim > 0 else 1.0
    im = ax.pcolormesh(x, y, z, shading='nearest', cmap=TRANSFER_CMAP,
                       norm=SymLogNorm(linthresh=lim * 1e-3, vmin=-lim, vmax=lim, base=10))
    fig.colorbar(im, ax=ax, label=label, pad=0.02)
    return im


def plot_modal_transfer(p_modes, times, onset=None, title_prefix="", filename=""):
    """Power into each longitudinal mode over time. p_modes: (blocks, n+1),
    the last column = all higher modes (rows sum to the total transfer)."""
    p = np.asarray(p_modes, np.float64)
    t = np.asarray(times, np.float64)
    n = p.shape[1] - 1
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1.3]})
    im = _symlog_map(ax, fig, p[:, :n].T, t, np.arange(1, n + 1),
                     "Power into mode (W)\n← out of | into longitudinal mode →")
    rows = np.abs(p[:, :n]).max(axis=0)
    if rows.size and rows.max() > 0:
        live = np.nonzero(rows > 1e-4 * rows.max())[0]
        ax.set_ylim(0.5, min(n, int(live[-1]) + 3) + 0.5)
    ax.set_ylabel("Longitudinal mode number n")
    ax.set_title(f"[{title_prefix}] Transfer into each Longitudinal Mode (sine basis, exact split)")
    if onset is not None:
        ax.axvline(onset, color=INK, linestyle=':', linewidth=1.0)

    total = p.sum(axis=1)
    ax2.plot(t, total, color=INK, linewidth=1.0, label="all modes (= total transfer)")
    ax2.plot(t, p[:, n], color=PALETTE[6], linewidth=1.0, label=f"modes above {n}")
    ax2.axhline(0.0, color=INK_MUTED, linewidth=0.6)
    ax2.set_ylabel("Power (W)")
    ax2.set_xlabel("Time (s)")
    _legend_above(ax2, ncol=2)
    _style_axis(ax2)
    fig.colorbar(im, ax=ax2, pad=0.02).ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_transfer_spectrum(spec, f_trans=(), f_long=(), title_prefix="", filename=""):
    """Time-averaged transfer power per frequency, summed over modes (top) and
    per longitudinal mode (bottom). The average is Welch's: Hann^2-weighted
    50%-overlap segments, so its integral equals the plain time mean of the
    transfer only where the transfer is stationary. Marked from measured data only: the
    longitudinal resonances and the combination frequencies of the transverse
    resonances (forced transfer)."""
    freqs = np.asarray(spec["freqs"], np.float64)
    pnf = np.asarray(spec["p_nf"], np.float64)          # (n_modes, F) W/Hz
    p_f = pnf.sum(axis=0)
    # Frequency range holding 99.5% of the absolute transfer.
    c = np.cumsum(np.abs(p_f))
    f_max = freqs[min(np.searchsorted(c, 0.995 * c[-1]) + 1, freqs.size - 1)] if c[-1] > 0 else freqs[-1]
    f_max = max(f_max, 1.2 * max(list(f_long[:3]) + [0.0]), freqs[min(20, freqs.size - 1)])
    sel = freqs <= f_max

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 3]})
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.6)
    y = p_f[sel]
    ax.fill_between(freqs[sel], 0, y, where=y > 0, color=C_LONG, alpha=0.6, linewidth=0,
                    label="into longitudinal")
    ax.fill_between(freqs[sel], 0, y, where=y < 0, color=C_TRANS, alpha=0.6, linewidth=0,
                    label="back into transverse / rotation")
    ax.plot(freqs[sel], y, color=INK, linewidth=0.6)
    for i, f in enumerate([f for f in f_long if f <= f_max]):
        ax.axvline(f, color=C_LONG, linestyle=':', linewidth=1.0,
                   label="longitudinal resonance (measured)" if i == 0 else None)
        ax2.axvline(f, color=INK, linestyle=':', linewidth=0.6)
    comb = [f for f in combination_frequencies(f_trans) if f <= f_max]
    if comb:
        ax.plot(comb, np.full(len(comb), 1.0), transform=ax.get_xaxis_transform(), linestyle='none',
                marker='v', markersize=5, color=C_TRANS, clip_on=False,
                label="f_m ± f_n of transverse resonances (measured)")
    ax.set_ylabel("Transfer power density (W/Hz)")
    mean_p = np.sum(p_f) * (freqs[1] - freqs[0])
    ax.set_title("Time-averaged transfer per frequency\n"
                 f"(integral = mean transfer, Welch-weighted: {mean_p * 1e3:.3g} mW)", fontsize=10, loc='left')
    _legend_above(ax, ncol=2)
    _style_axis(ax)

    n = pnf.shape[0]
    im = _symlog_map(ax2, fig, pnf[:, sel], freqs[sel], np.arange(1, n + 1),
                     "W/Hz  (← out of | into mode →)")
    n_show = n
    rows = np.abs(pnf[:, sel]).max(axis=1)
    if rows.max() > 0:
        live = np.nonzero(rows > 1e-4 * rows.max())[0]
        if live.size:
            n_show = min(n, int(live[-1]) + 3)
    ax2.set_ylim(0.5, n_show + 0.5)
    ax2.set_ylabel("Longitudinal mode number n")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_xlim(0, f_max)
    fig.colorbar(im, ax=ax, pad=0.02).ax.set_visible(False)

    fig.suptitle(f"[{title_prefix}] Transfer Spectrum: at which Frequencies Energy Enters Longitudinal Motion")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_energy_source(time, total_energy, flow, title_prefix="", filename=""):
    """Is energy created? total_energy: the shader's quadratic-form total per
    frame. flow: D_all, E_L, P_disp, P_rot, D_L, dF_x_rms, dF_perp_rms,
    H_x, H_perp (roundoff heating estimates, W)."""
    t = np.asarray(time, np.float64)
    get = lambda k: np.asarray(flow[k], np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    ax = axes[0]
    e = np.asarray(total_energy, np.float64)
    resid = e - e[0] + _cumulative(get("D_all"), t)
    ax.plot(t, resid, color=INK, linewidth=1.2, label="E(t) − E(t₀) + ∫ damping")
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.6)
    ax.set_ylabel("Energy (J)")
    ax.set_title("Total energy balance (shader's quadratic-form energy; flat = conserved, "
                 f"start {e[0]:.3g} J)", fontsize=10, loc='left')
    _legend_above(ax, ncol=1)
    _style_axis(ax)

    ax = axes[1]
    e_l = get("E_L")
    gap = e_l - e_l[0] + _cumulative(get("D_L") - get("P_disp") - get("P_rot"), t)
    ax.plot(t, gap, color=C_LONG, linewidth=1.2,
            label="longitudinal energy not explained by the force law")
    ax.plot(t, _cumulative(get("H_x"), t), color=INK, linewidth=1.0, linestyle='--',
            label="float32 roundoff heating, x (if independent per step)")
    ax.plot(t, _cumulative(get("H_perp"), t), color=C_TRANS, linewidth=1.0, linestyle='--',
            label="float32 roundoff heating, transverse")
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.6)
    ax.set_ylabel("Energy (J)")
    ax.set_title("Unexplained longitudinal energy,\nand what float32 rounding could inject",
                 fontsize=10, loc='left')
    _legend_above(ax, ncol=2)
    _style_axis(ax)

    ax = axes[2]
    ax.plot(t, get("dF_x_rms"), color=C_LONG, linewidth=1.0, label="x (longitudinal)")
    ax.plot(t, get("dF_perp_rms"), color=C_TRANS, linewidth=1.0, label="y, z (transverse)")
    ax.set_yscale('log')
    ax.set_ylabel("RMS node force (N)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Float32 rounding of the node force\n(post-processing port of the shader, same "
                 "inputs; the GPU's own rounding differs in detail)", fontsize=10, loc='left')
    _legend_above(ax, ncol=2)
    _style_axis(ax)

    fig.suptitle(f"[{title_prefix}] Energy Source Diagnostic")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]
