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
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.ticker import EngFormatter
from scipy.fft import dst

from plots.energy import (PALETTE, INK, INK_MUTED, ENERGY_STYLE, INTO_LONG, OUT_OF_LONG,
                          DIVERGING_CMAP, TRANSLATIONAL_CMAP, NEUTRAL_CMAP,
                          _style_axis, _smooth, _window_frames)

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


ETA_SLOPE = -0.3


def _variational_slope(eta, sigma, K, omega, inertia, dl, dtype):
    """eta eta' (sigma^T K sigma - omega^T I omega / dl): the stretch derivative of the stored
    energy 1/2 eta^2 sigma^T K sigma dl and of the rotational kinetic energy 1/2 omega^T I eta^2
    omega, per unit of tangent (variational force law)."""
    slope = np.sum(K * sigma * sigma, axis=-1)
    if omega is not None:
        om = np.asarray(omega, dtype=dtype)
        slope = slope - np.sum(np.asarray(inertia, dtype=dtype) * om * om, axis=-1) / dtype(dl)
    return dtype(ETA_SLOPE) * eta * slope


def edge_forces(r, quats, dl, K_se, dtype=np.float64, force_law="legacy", omega=None, inertia=None):
    """Internal edge forces f_e (lab frame) exactly as compute_internals, for
    edge vectors r (..., 3) and orientations quats (..., 4). dtype=float32
    reproduces the shader's precision (up to operation order and FMA).
    force_law="variational" adds eta eta' (sigma^T K sigma - omega^T I omega / dl) along the
    tangent (pass the edges' angular velocities and the per-edge inertia for the kinetic part).
    This is the edge force the shader stores; the bending-region part is in
    bend_stretch_forces."""
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
    n = K * (eta * eta)[..., None] * sigma
    if force_law == "variational":
        tangent_unit = tangent_MF * (dl_ * len_inv)[..., None]
        n = n + _variational_slope(eta, sigma, K, omega, inertia, dl, dtype)[..., None] * tangent_unit
    return rotate(q, n)


def rotation_deviation_inv(p):
    """R(p)^T e3 - e3 for unit quaternions p, from the vector part only (WGSL
    rotation_deviation_inv)."""
    pv, w = p[..., :3], p[..., 3:]
    e3 = _E3.astype(p.dtype)
    return -2 * w * np.cross(pv, e3) + 2 * np.cross(pv, np.cross(pv, e3))


def edge_forces_offset(du, p, frames, offsets, dl, K_se, dtype=np.float64, force_law="legacy",
                       omega=None, inertia=None):
    """Internal edge forces f_e (lab frame) as the current compute_internals
    evaluates them: orientation q = q0 p stored as the deviation p, rest frame
    R0 (frames, (N-1, 3, 3), columns d1 d2 d3) and offsets R0^T ref - dl e3.
    Algebraically identical to edge_forces(ref + du, q0 p, ...); only the
    rounding of the intermediates differs."""
    du = np.asarray(du, dtype=dtype)
    p = np.asarray(p, dtype=dtype)
    R0 = np.asarray(frames, dtype=dtype)
    dl_ = dtype(dl)
    w = np.asarray(offsets, dtype=dtype) + np.einsum('...nk,nkj->...nj', du, R0)
    sigma = rotate_inv(p, w) / dl_ + rotation_deviation_inv(p)
    length = np.sqrt(w[..., 0] ** 2 + w[..., 1] ** 2 + (dl_ + w[..., 2]) ** 2)
    s = (dtype(2) * dl_ * w[..., 2] + np.sum(w * w, axis=-1)) / (dl_ * (length + dl_))
    eta = dtype(1) - dtype(0.3) * s
    K = np.asarray(K_se, dtype=dtype)
    n = K * (eta * eta)[..., None] * sigma
    if force_law == "variational":
        tangent_unit = (_E3.astype(dtype) + sigma) * (dl_ / length)[..., None]
        n = n + _variational_slope(eta, sigma, K, omega, inertia, dl, dtype)[..., None] * tangent_unit
    return np.einsum('nij,...nj->...ni', R0, rotate(p, n))


def curvatures(quats, dl):
    """Curvature of every bending region (edges i, i+1), (..., N-2, 3), as the shader computes it
    with identity reference orientations: 2/dl times the vector part of q_i^-1 q_(i+1), on the
    pole closest to the identity."""
    q = np.asarray(quats, dtype=np.float64)
    rel = _qmul(_qinv(q[..., :-1, :]), q[..., 1:, :])
    rel = np.where(rel[..., 3:] < 0, -rel, rel)
    return 2.0 / float(dl) * rel[..., :3]


def edge_stretch(r, dl):
    """e - 1 per edge from the edge vectors."""
    return np.linalg.norm(np.asarray(r, dtype=np.float64), axis=-1) / float(dl) - 1.0


def bend_stretch_forces(r, quats, dl, K_bt):
    """Edge forces (lab, (..., N-1, 3)) from the stretch dependence of the bending/twist energy
    1/2 eta_v^4 kappa^T K_bt kappa dl of each region (variational force law): edge e carries
    (c_e + c_(e-1)) along its tangent, c_i = eta' eta_v,i^3 kappa_i^T K_bt kappa_i."""
    r = np.asarray(r, dtype=np.float64)
    s = edge_stretch(r, dl)
    kap = curvatures(quats, dl)
    eta_v = 1.0 + ETA_SLOPE * 0.5 * (s[..., :-1] + s[..., 1:])
    c = ETA_SLOPE * eta_v ** 3 * np.sum(np.asarray(K_bt, np.float64) * kap * kap, axis=-1)
    c_edge = np.zeros(s.shape)
    c_edge[..., :-1] += c
    c_edge[..., 1:] += c
    t_hat = r / np.linalg.norm(r, axis=-1, keepdims=True)
    return c_edge[..., None] * t_hat


def stored_energy(r, quats, dl, K_se, K_bt):
    """Stored energy per frame (..., ): sum over edges of 1/2 eta^2 sigma^T K_se sigma dl plus sum
    over bending regions of 1/2 eta_v^4 kappa^T K_bt kappa dl (reference strain and reference
    curvature zero). The variational force law is its exact negative gradient."""
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(quats, dtype=np.float64)
    sigma = rotate_inv(q, r / float(dl)) - _E3
    s = edge_stretch(r, dl)
    eta = 1.0 + ETA_SLOPE * s
    W_s = 0.5 * eta ** 2 * np.sum(np.asarray(K_se, np.float64) * sigma * sigma, axis=-1) * dl
    kap = curvatures(q, dl)
    eta_v = 1.0 + ETA_SLOPE * 0.5 * (s[..., :-1] + s[..., 1:])
    W_b = 0.5 * eta_v ** 4 * np.sum(np.asarray(K_bt, np.float64) * kap * kap, axis=-1) * dl
    return W_s.sum(axis=-1) + W_b.sum(axis=-1)


def stored_energy_offset(du, p, frames, offsets, rel_rest, dl, K_se, K_bt):
    """stored_energy evaluated as the shader evaluates strain and curvature: from the offset
    w = R0^T ref - dl e3 + R0^T du, the deviations p (q = q0 p) and the rest relative rotations
    rel_rest = q0_i^-1 q0_(i+1) (utils.rest_relative_orientations). Algebraically identical to
    stored_energy; but stored_energy forms R(q)^T r / dl - e3 from numbers near 1, so the float32
    norm error of each saved orientation (~6e-8) enters the axial strain and, times the tension,
    the energy (~2e-8 J per rotated edge on the cello D). Here only small quantities are rounded."""
    du = np.asarray(du, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    R0 = np.asarray(frames, dtype=np.float64)
    dl_ = float(dl)
    w = np.asarray(offsets, dtype=np.float64) + np.einsum('...nk,nkj->...nj', du, R0)
    sigma = rotate_inv(p, w) / dl_ + rotation_deviation_inv(p)
    length = np.sqrt(w[..., 0] ** 2 + w[..., 1] ** 2 + (dl_ + w[..., 2]) ** 2)
    s = (2.0 * dl_ * w[..., 2] + np.sum(w * w, axis=-1)) / (dl_ * (length + dl_))
    eta = 1.0 + ETA_SLOPE * s
    W_s = 0.5 * eta ** 2 * np.sum(np.asarray(K_se, np.float64) * sigma * sigma, axis=-1) * dl_
    r0 = np.asarray(rel_rest, dtype=np.float64)
    rel = _qmul(_qmul(_qinv(p[..., :-1, :]), r0), p[..., 1:, :])
    rel = np.where(np.sum(rel * r0, axis=-1, keepdims=True) < 0, -rel, rel)
    kap = 2.0 / dl_ * _qmul(_qinv(r0), rel)[..., :3]
    eta_v = 1.0 + ETA_SLOPE * 0.5 * (s[..., :-1] + s[..., 1:])
    W_b = 0.5 * eta_v ** 4 * np.sum(np.asarray(K_bt, np.float64) * kap * kap, axis=-1) * dl_
    return W_s.sum(axis=-1) + W_b.sum(axis=-1)


def node_forces(f):
    """F_j = f_j - f_{j-1} on interior nodes; zero at the fixed ends."""
    out = np.zeros(f.shape[:-2] + (f.shape[-2] + 1, 3), dtype=f.dtype)
    out[..., 1:-1, :] = f[..., 1:, :] - f[..., :-1, :]
    return out


def _g_derivs(s0, force_law="legacy"):
    """First to fourth derivative at s0 of the longitudinal potential g per K_z dl of one edge:
    legacy      g(s) = s^2/2 - 0.2 s^3 + 0.0225 s^4  (potential of K_z eta^2 s)
    variational g(s) = 1/2 eta^2 s^2 = s^2/2 - 0.3 s^3 + 0.045 s^4  (the stored energy itself)."""
    if force_law == "variational":
        return (s0 - 0.9 * s0 ** 2 + 0.18 * s0 ** 3, 1.0 - 1.8 * s0 + 0.54 * s0 ** 2,
                -1.8 + 1.08 * s0, 1.08)
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


def exact_flow(pos, vel, quats, ref, q_rest, dl, m_node, K_se, c_lin=0.0, block=64,
               force_law="legacy", omega=None, inertia=None, K_bt=None, offset_form=None):
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
      u_l_bend       (F, N-2)  variational law only: change of each bending region's energy
                               in the longitudinal-only state (non-zero for a pre-twisted
                               string, whose rest twist energy depends on stretch)
    force_law="variational" evaluates the variational shader force (pass omega (F, N-1, 3),
    the per-edge inertia and K_bt); states B and C have no rotation, so no kinetic term.
    offset_form=(p, frames, offsets) for runs that store the deviations p (q = q0 p): the edge
    forces of states A and B are then evaluated as the shader evaluates them (edge_forces_offset,
    rest frames and offsets from utils). The absolute form R(q)^T r / dl - e3 subtracts numbers
    near 1, so the float32 norm error of each saved orientation (~1e-7) would enter the axial
    strain, i.e. about EA * 1e-7 per edge (run 002: 1.3e-2 N rms per node).
    """
    F, N = pos.shape[:2]
    ref = np.asarray(ref, dtype=np.float64)
    q_rest = np.asarray(q_rest, dtype=np.float64)
    K_z = float(np.ravel(K_se)[2])
    s0 = ref[:, 0] / dl - 1.0
    g1, g2, g3, g4 = _g_derivs(s0, force_law)
    variational = force_law == "variational"
    if variational:
        kap0 = curvatures(q_rest, dl)                                       # rest curvature
        B0 = np.sum(np.asarray(K_bt, np.float64) * kap0 * kap0, axis=-1)    # per region

    w = np.ones(N)
    w[[0, -1]] = 0.5
    out = {k: np.zeros((F, N)) for k in ("ke_t", "ke_l", "p_disp", "p_rot", "f_c", "d_l")}
    out["u_l_quad"] = np.zeros((F, N - 1))
    out["u_l_lin"] = np.zeros((F, N - 1))
    out["u_l_bend"] = np.zeros((F, N - 2))

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
        s_c = s0 + d                                          # stretch in the longitudinal-only state
        eta_c = 1.0 - 0.3 * s_c
        f_c_edge = K_z * eta_c ** 2 * s_c
        if variational:
            f_c_edge = f_c_edge + K_z * eta_c * ETA_SLOPE * s_c ** 2
            # rest twist energy of each region at the state-C stretches, and its stretch derivative
            eta_v = 1.0 + ETA_SLOPE * 0.5 * (s_c[:, :-1] + s_c[:, 1:])
            eta_v0 = 1.0 + ETA_SLOPE * 0.5 * (s0[:-1] + s0[1:])
            out["u_l_bend"][a:b] = 0.5 * (eta_v ** 4 - eta_v0 ** 4) * B0 * dl
            c = ETA_SLOPE * eta_v ** 3 * B0
            f_c_edge[:, :-1] += c
            f_c_edge[:, 1:] += c
        out["u_l_lin"][a:b] = K_z * dl * g1 * d
        out["u_l_quad"][a:b] = K_z * dl * (0.5 * g2 * d ** 2 + g3 * d ** 3 / 6.0 + g4 * d ** 4 / 24.0)

        Fx_C = np.zeros((b - a, N))
        Fx_C[:, 1:-1] = f_c_edge[:, 1:] - f_c_edge[:, :-1]
        q_r = np.broadcast_to(q_rest, q.shape)
        if offset_form is not None:
            p_dev, frames, offsets = offset_form
            pa = np.asarray(p_dev[a:b], dtype=np.float64)
            pb = np.zeros_like(pa)
            pb[..., 3] = 1.0
            om = np.asarray(omega[a:b], dtype=np.float64) if variational else None
            f_B = edge_forces_offset(du, pb, frames, offsets, dl, K_se, force_law=force_law)
            f_A = edge_forces_offset(du, pa, frames, offsets, dl, K_se, force_law=force_law,
                                     omega=om, inertia=inertia)
            if variational:
                f_B = f_B + bend_stretch_forces(r, q_r, dl, K_bt)
                f_A = f_A + bend_stretch_forces(r, q, dl, K_bt)
        elif variational:
            om = np.asarray(omega[a:b], dtype=np.float64)
            f_B = edge_forces(r, q_r, dl, K_se, force_law=force_law) + bend_stretch_forces(r, q_r, dl, K_bt)
            f_A = (edge_forces(r, q, dl, K_se, force_law=force_law, omega=om, inertia=inertia)
                   + bend_stretch_forces(r, q, dl, K_bt))
        else:
            f_B = edge_forces(r, q_r, dl, K_se)
            f_A = edge_forces(r, q, dl, K_se)
        Fx_B = node_forces(f_B)[..., 0]
        Fx_A = node_forces(f_A)[..., 0]

        out["p_disp"][a:b] = v[..., 0] * (Fx_B - Fx_C)
        out["p_rot"][a:b] = v[..., 0] * (Fx_A - Fx_B)
        out["f_c"][a:b] = Fx_A - Fx_C
    return out


def roundoff_forces(pos, quats, ref, q_rest, frames, offsets, dl, K_se, block=64, pos_lo=None):
    """Float32 rounding of the node force, for both shader formulations:
    (old, new), each (F, N, 3) = float32 evaluation minus the float64
    evaluation of the same stored float32 state.

    old: absolute orientations q and r = ref + (u_{j+1} - u_j), strain as
         R(q)^T r / dl - e3 (the shader before the offset formulation);
    new: deviations p = q0^-1 q and the offset w = R0^T r - dl e3
         (edge_forces_offset).
    The stored state is float32 in both cases (q resp. p, u, ref, R0, offsets),
    so the float64 reference is the exact force of what the GPU holds.
    pos_lo: low parts of the two-part displacement (pos is then the high part); the
    differences are then formed as the shader's node_difference does,
    (hi_{j+1} - hi_j) + (lo_{j+1} - lo_j) in float32.
    """
    F, N = pos.shape[:2]
    old, new = np.zeros((F, N, 3)), np.zeros((F, N, 3))
    ref32 = np.asarray(ref, dtype=np.float32)
    frames32 = np.asarray(frames, dtype=np.float32)
    off32 = np.asarray(offsets, dtype=np.float32)
    q0_inv = _qinv(np.asarray(q_rest, dtype=np.float64))
    for a in range(0, F, block):
        b = min(F, a + block)
        u = np.asarray(pos[a:b], dtype=np.float32)
        du32 = u[:, 1:] - u[:, :-1]
        du64 = np.diff(u.astype(np.float64), axis=1)
        if pos_lo is not None:
            lo = np.asarray(pos_lo[a:b], dtype=np.float32)
            du32 = du32 + (lo[:, 1:] - lo[:, :-1])
            du64 = du64 + np.diff(lo.astype(np.float64), axis=1)
        q64 = np.asarray(quats[a:b], dtype=np.float64)
        q32 = q64.astype(np.float32)
        F32 = node_forces(edge_forces(ref32 + du32, q32, dl, K_se, dtype=np.float32))
        F64 = node_forces(edge_forces(ref32.astype(np.float64) + du64,
                                      q32.astype(np.float64), dl, K_se))
        old[a:b] = F32.astype(np.float64) - F64

        p = _qmul(q0_inv, q64)
        p32 = (p / np.linalg.norm(p, axis=-1, keepdims=True)).astype(np.float32)
        F32 = node_forces(edge_forces_offset(du32, p32, frames32, off32, dl, K_se, dtype=np.float32))
        F64 = node_forces(edge_forces_offset(du64, p32, frames32, off32, dl, K_se))
        new[a:b] = F32.astype(np.float64) - F64
    return old, new


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
        ax.set_title(title, loc='left', fontsize=ax.title.get_fontsize(), pad=8 + 15 * rows)


def _cumulative(y, t):
    """Trapezoid running integral starting at 0."""
    out = np.zeros_like(y, dtype=np.float64)
    if len(t) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def _single(figsize=(12, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _finish(fig, ax, title, filename, onset=None, legend_cols=None, unit=None):
    """Title, then the legend above the axes (which lifts the title), shared
    styling and saving. unit: engineering tick labels (e.g. 20 µW) on a linear
    y axis, instead of a separate exponent that would collide with the legend."""
    if onset is not None:
        ax.axvline(onset, color=INK_MUTED, linestyle=':', linewidth=1.0)
    if unit:
        ax.yaxis.set_major_formatter(EngFormatter(unit=unit))
    ax.set_title(title, loc='left', fontsize=11)
    if legend_cols:
        _legend_above(ax, ncol=legend_cols)
    _style_axis(ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def _polarity_fill(ax, x, y, step=None):
    """Fill by sign: into longitudinal motion (positive) / out of it."""
    kw = dict(alpha=0.6, linewidth=0, step=step)
    ax.fill_between(x, 0, y, where=y > 0, color=INTO_LONG, label="into longitudinal", **kw)
    ax.fill_between(x, 0, y, where=y < 0, color=OUT_OF_LONG, label="out of longitudinal", **kw)
    ax.plot(x, y, color=INK, linewidth=0.6, drawstyle='steps-mid' if step else 'default')
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.6)


def plot_energy_flow(time, flow, window_s=None, onset=None, title_prefix="", filename=""):
    """Energy per kind of motion: E_L (exact longitudinal energy, KE + PE) and
    the kinetic energies of the other motions. window_s smooths the kinetic
    energies (long phases)."""
    t = np.asarray(time, dtype=np.float64)
    get = lambda k: np.asarray(flow[k], np.float64)
    sm = (lambda y: _smooth(y, _window_frames(t, window_s))) if window_s else (lambda y: y)
    series = (("Longitudinal (KE + PE)", get("E_L")),
              ("Transverse KE", sm(get("KE_T"))),
              ("Tilt KE", sm(get("KE_R"))),
              ("Torsion KE", sm(get("KE_tor"))))

    fig, ax = _single((12, 6))
    for label, y in series:
        ax.plot(t, np.where(y > 0, y, np.nan), linewidth=1.4, label=label, **ENERGY_STYLE[label])
    for label, y in series[:2]:
        _direct_label(ax, t, y, label, ENERGY_STYLE[label]["color"])
    ax.set_yscale('log')
    peak = max(np.nanmax(series[0][1]), np.nanmax(series[1][1]), 1e-30)
    ax.set_ylim(peak * 1e-6, peak * 3)
    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("Time (s)")
    title = f"[{title_prefix}] Energy per Kind of Motion"
    if window_s:
        title += f" (KE smoothed over {window_s * 1e3:.1f} ms)"
    return _finish(fig, ax, title, filename, onset, legend_cols=4)


def plot_transfer_power(time, flow, window_s=None, onset=None, period=None,
                        title_prefix="", filename=""):
    """Transfer power P_c = P_disp + P_rot into longitudinal motion."""
    t = np.asarray(time, dtype=np.float64)
    p_raw = np.asarray(flow["P_disp"], np.float64) + np.asarray(flow["P_rot"], np.float64)
    p = _smooth(p_raw, _window_frames(t, window_s)) if window_s else p_raw

    fig, ax = _single()
    _polarity_fill(ax, t, p)
    ax.set_ylabel("Transfer power")
    ax.set_xlabel("Time (s)")
    title = f"[{title_prefix}] Transfer Power into Longitudinal Motion"
    if window_s:
        title += f" (smoothed over {window_s * 1e3:.1f} ms)"
    if onset is not None and period:
        sel = (t >= onset) & (t <= onset + period)
        if sel.sum() > 1:
            rate = np.trapezoid(p_raw[sel], t[sel]) / (t[sel][-1] - t[sel][0])
            title += f"; first-period mean {rate * 1e3:.3g} mW"
    return _finish(fig, ax, title, filename, onset, legend_cols=2, unit='W')


def plot_transfer_cumulative(time, flow, onset=None, title_prefix="", filename=""):
    """Cumulative transfer by channel against the change of E_L plus damping."""
    t = np.asarray(time, dtype=np.float64)
    get = lambda k: np.asarray(flow[k], np.float64)
    e_l = get("E_L")
    c_disp, c_rot = _cumulative(get("P_disp"), t), _cumulative(get("P_rot"), t)

    fig, ax = _single()
    ax.plot(t, e_l - e_l[0] + _cumulative(get("D_L"), t), color=INK, linewidth=1.6,
            label="ΔE_L + ∫D_L dt")
    ax.plot(t, c_disp + c_rot, color=INTO_LONG, linewidth=1.2, linestyle='--', label="∫P_c dt")
    ax.plot(t, c_disp, color=PALETTE[4], linewidth=1.0, label="∫P_disp dt")
    ax.plot(t, c_rot, color=PALETTE[5], linewidth=1.0, label="∫P_rot dt")
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.6)
    ax.set_ylabel("Energy")
    ax.set_xlabel("Time (s)")
    return _finish(fig, ax, f"[{title_prefix}] Energy Delivered to Longitudinal Motion",
                   filename, onset, legend_cols=4, unit='J')


MAP_KINDS = {
    "trans": ("Transverse KE per node (J)", TRANSLATIONAL_CMAP, "Transverse Kinetic Energy along the String"),
    "long_ke": ("Longitudinal KE per node (J)", TRANSLATIONAL_CMAP,
                "Longitudinal Kinetic Energy along the String"),
    "long": ("Longitudinal energy (KE + PE) per node (J)", NEUTRAL_CMAP,
             "Longitudinal Energy along the String"),
    "transfer": ("Transfer power per node (W)", DIVERGING_CMAP,
                 "Local Transfer into Longitudinal Motion"),
}


def _extent(x, t):
    dx = x[1] - x[0] if x.size > 1 else 1.0
    dt = (t[1] - t[0]) if t.size > 1 else 1e-3
    return (x[0] - dx / 2, x[-1] + dx / 2, t[0] - dt / 2, t[-1] + dt / 2)


def plot_element_map(time, values, length, kind, onset=None, title_prefix="", filename=""):
    """Position x time map of one per-node quantity (block-averaged rows).

    kind: 'trans' / 'long_ke' / 'long' (energy, log colour scale) or 'transfer'
    (power, diverging, symmetric limits at the 99th percentile of |p|).
    """
    label, cmap, title = MAP_KINDS[kind]
    v = np.asarray(values, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    x = np.linspace(0.0, length, v.shape[1])
    extent = _extent(x, t)

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
    ax.set_xlabel("Position along string (m)")
    ax.set_title(f"[{title_prefix}] {title}")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_element_transfer_integral(time, values, length, title_prefix="", filename=""):
    """Time-integrated transfer power per position over the window."""
    v = np.asarray(values, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    x = np.linspace(0.0, length, v.shape[1])
    dt = (t[1] - t[0]) if t.size > 1 else 0.0
    e = v.sum(axis=0) * dt                                # block means x block length

    fig, ax = _single((12, 4.5))
    _polarity_fill(ax, x, e, step='mid')
    ax.set_ylabel("∫p_c dt")
    ax.set_xlabel("Position along string (m)")
    return _finish(fig, ax, f"[{title_prefix}] Energy Delivered to Longitudinal Motion per Position",
                   filename, legend_cols=2, unit='J')


def _symlog_map(ax, fig, z, x, y, label):
    z = np.asarray(z, np.float64)
    lim = np.percentile(np.abs(z), 99.5) if z.size else 1.0
    lim = lim if lim > 0 else 1.0
    im = ax.pcolormesh(x, y, z, shading='nearest', cmap=DIVERGING_CMAP,
                       norm=SymLogNorm(linthresh=lim * 1e-3, vmin=-lim, vmax=lim, base=10))
    fig.colorbar(im, ax=ax, label=label, pad=0.02)
    return im


def plot_modal_transfer(p_modes, times, onset=None, title_prefix="", filename=""):
    """Power into each longitudinal mode over time. p_modes: (blocks, n+1),
    the last column = all higher modes (not shown here, see the sum plot)."""
    p = np.asarray(p_modes, np.float64)
    t = np.asarray(times, np.float64)
    n = p.shape[1] - 1
    fig, ax = plt.subplots(figsize=(12, 7))
    _symlog_map(ax, fig, p[:, :n].T, t, np.arange(1, n + 1), "Power into mode (W)")
    rows = np.abs(p[:, :n]).max(axis=0)
    if rows.size and rows.max() > 0:
        live = np.nonzero(rows > 1e-4 * rows.max())[0]
        ax.set_ylim(0.5, min(n, int(live[-1]) + 3) + 0.5)
    ax.set_ylabel("Longitudinal mode number n")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"[{title_prefix}] Transfer into each Longitudinal Mode")
    if onset is not None:
        ax.axvline(onset, color=INK, linestyle=':', linewidth=1.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_modal_transfer_sum(p_modes, times, onset=None, title_prefix="", filename=""):
    """Sum over all modes (= P_c) and the part above the mapped modes."""
    p = np.asarray(p_modes, np.float64)
    t = np.asarray(times, np.float64)
    n = p.shape[1] - 1
    fig, ax = _single((12, 4.5))
    ax.plot(t, p.sum(axis=1), color=INK, linewidth=1.0, label="all modes")
    ax.plot(t, p[:, n], color=INK_MUTED, linewidth=1.0, linestyle=':', label=f"n > {n}")
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.6)
    ax.set_ylabel("Power")
    ax.set_xlabel("Time (s)")
    return _finish(fig, ax, f"[{title_prefix}] Transfer into Longitudinal Modes, Sum", filename, onset, legend_cols=2, unit='W')


def plot_transfer_spectrum(spec, f_trans=(), f_long=(), title_prefix="", filename=""):
    """Time-averaged transfer power per frequency, summed over modes (top) and
    per longitudinal mode (bottom). The average is Welch's: Hann^2-weighted
    50%-overlap segments, so its integral equals the plain time mean of the
    transfer only where the transfer is stationary. Marked from measured data
    only: the longitudinal resonances and the combination frequencies of the
    transverse resonances."""
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
    _polarity_fill(ax, freqs[sel], p_f[sel])
    for i, f in enumerate([f for f in f_long if f <= f_max]):
        ax.axvline(f, color=INK, linestyle=':', linewidth=1.0,
                   label="longitudinal resonance" if i == 0 else None)
        ax2.axvline(f, color=INK, linestyle=':', linewidth=0.6)
    comb = [f for f in combination_frequencies(f_trans) if f <= f_max]
    if comb:
        ax.plot(comb, np.full(len(comb), 1.0), transform=ax.get_xaxis_transform(), linestyle='none',
                marker='v', markersize=5, color=INK_MUTED, clip_on=False,
                label="f_m ± f_n, 2f_n (transverse resonances)")
    ax.set_ylabel("Transfer power density (W/Hz)")
    mean_p = np.sum(p_f) * (freqs[1] - freqs[0])
    ax.set_title(f"Integral {mean_p * 1e3:.3g} mW", fontsize=10, loc='left')
    _legend_above(ax, ncol=2)
    _style_axis(ax)

    n = pnf.shape[0]
    im = _symlog_map(ax2, fig, pnf[:, sel], freqs[sel], np.arange(1, n + 1), "W/Hz")
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

    fig.suptitle(f"[{title_prefix}] Transfer Spectrum")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_energy_balance(time, total_energy, flow, title_prefix="", filename=""):
    """Quadratic-form total energy change plus dissipated energy."""
    t = np.asarray(time, np.float64)
    e = np.asarray(total_energy, np.float64)
    resid = e - e[0] + _cumulative(np.asarray(flow["D_all"], np.float64), t)
    fig, ax = _single()
    ax.plot(t, resid, color=INK, linewidth=1.2)
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.6)
    ax.set_ylabel("E(t) − E(t₀) + ∫D dt")
    ax.set_xlabel("Time (s)")
    return _finish(fig, ax, f"[{title_prefix}] Total Energy Balance", filename, unit='J')


def plot_energy_unexplained(time, flow, title_prefix="", filename=""):
    """Change of E_L not accounted for by transfer and damping, with the
    accumulated per-step float32 heating estimate of the run's formulation."""
    t = np.asarray(time, np.float64)
    get = lambda k: np.asarray(flow[k], np.float64)
    e_l = get("E_L")
    gap = e_l - e_l[0] + _cumulative(get("D_L") - get("P_disp") - get("P_rot"), t)
    fig, ax = _single()
    ax.plot(t, gap, color=INK, linewidth=1.2, label="ΔE_L + ∫(D_L − P_c) dt")
    ax.plot(t, _cumulative(get("H_x"), t), color=INK_MUTED, linewidth=1.0, linestyle='--',
            label="float32 heating, x")
    ax.plot(t, _cumulative(get("H_perp"), t), color=INK_MUTED, linewidth=1.0, linestyle=':',
            label="float32 heating, y, z")
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.6)
    ax.set_ylabel("Energy")
    ax.set_xlabel("Time (s)")
    return _finish(fig, ax, f"[{title_prefix}] Longitudinal Energy Residual", filename, legend_cols=3, unit='J')


def plot_roundoff_force(time, flow, relative=True, title_prefix="", filename=""):
    """RMS float32 rounding of the node force for both shader formulations."""
    t = np.asarray(time, np.float64)
    get = lambda k: np.asarray(flow[k], np.float64)
    run, other = ("offset", "absolute") if relative else ("absolute", "offset")
    fig, ax = _single()
    for suffix, name, ls in (("", run, '-'), ("_other", other, ':')):
        if f"dF_x_rms{suffix}" not in flow:
            continue
        ax.plot(t, get(f"dF_x_rms{suffix}"), color=INK, linewidth=1.0, linestyle=ls,
                label=f"x, {name}")
        ax.plot(t, get(f"dF_perp_rms{suffix}"), color=INK_MUTED, linewidth=1.0, linestyle=ls,
                label=f"y, z, {name}")
    ax.set_yscale('log')
    ax.set_ylabel("RMS node force error (N)")
    ax.set_xlabel("Time (s)")
    return _finish(fig, ax, f"[{title_prefix}] Float32 Rounding of the Node Force", filename, legend_cols=4)
