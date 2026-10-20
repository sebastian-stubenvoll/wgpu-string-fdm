import numpy as np


def reference_vectors(node_count, tuned_length, legacy=False):
    """Edge reference vectors (N-1, 3) float32, as handed to the simulation.

    Every edge gets the bit-identical spacing. legacy=True reproduces the old
    construction (differences of float32 absolute positions), whose spacings
    scatter by ~1 ulp of L (~1e-7 m); that leaves net tension forces of
    several N at rest, i.e. an initial state that isn't in equilibrium. Post-
    processing uses it to evaluate runs made before the fix exactly.
    """
    if legacy:
        x = np.linspace(0.0, tuned_length, node_count).astype(np.float32)
        ref = np.zeros((node_count - 1, 3), dtype=np.float32)
        ref[:, 0] = x[1:] - x[:-1]
        return ref
    ref = np.zeros((node_count - 1, 3), dtype=np.float32)
    ref[:, 0] = np.float32(tuned_length / (node_count - 1))
    return ref


def split_float32(x):
    """x (float64) as the unevaluated sum hi + lo of two float32, the shader's two-part
    displacement: hi = fl32(x), lo = fl32(x - hi) (x - hi is exact in float64)."""
    x = np.asarray(x, dtype=np.float64)
    hi = x.astype(np.float32)
    return hi, (x - hi).astype(np.float32)


def node_displacement(nodes):
    """Displacement (..., 3) float64 from saved node records (..., K, 3), i.e. Simulation.save()
    rows [position, velocity, internal moment, curvature, position_lo]: position + position_lo.
    Runs saved before the low part was exported have K = 4 and give the high part only."""
    nodes = np.asarray(nodes)
    u = nodes[..., 0, :].astype(np.float64)
    if nodes.shape[-2] > 4:
        u += nodes[..., 4, :]
    return u


def quat_mul(q1, q2):
    """Hamilton product of quaternions (x, y, z, w), broadcasting over leading axes."""
    x1, y1, z1, w1 = np.moveaxis(np.asarray(q1, dtype=np.float64), -1, 0)
    x2, y2, z2, w2 = np.moveaxis(np.asarray(q2, dtype=np.float64), -1, 0)
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], axis=-1)


def quat_conj(q):
    q = np.array(q, dtype=np.float64)
    q[..., :3] *= -1.0
    return q


def quat_matrix(q):
    """Rotation matrix of unit quaternions (x, y, z, w); columns are R e1, R e2, R e3."""
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y + z*w), 2*(x*z - y*w)], axis=-1),
        np.stack([2*(x*y - z*w), 1 - 2*(x*x + z*z), 2*(y*z + x*w)], axis=-1),
        np.stack([2*(x*z + y*w), 2*(y*z - x*w), 1 - 2*(x*x + y*y)], axis=-1),
    ], axis=-1)


def _twist_angles(node_count, twists):
    total_edges = node_count - 1
    return 2.0 * np.pi * twists * (np.arange(total_edges) / total_edges)


def rest_orientations(node_count, twists):
    """Rest orientation quaternion (x, y, z, w) of every edge, (N-1, 4) float64:
    director d3 along the string (x), optionally twisted about it."""
    def twist_quaternion(angle):
        half = 0.5 * angle
        return np.array([np.sin(half), 0.0, 0.0, np.cos(half)])

    angle = np.pi / 2
    q_base = np.array([0.0, np.sin(0.5 * angle), 0.0, np.cos(0.5 * angle)])

    orientations = [quat_mul(twist_quaternion(phi), q_base)
                    for phi in _twist_angles(node_count, twists)]
    return np.array([q / np.linalg.norm(q) for q in orientations])


def rest_frames(node_count, twists):
    """Rest directors R0 = R(q0) of every edge, (N-1, 3, 3) float64, columns d1, d2, d3.

    Written out in closed form (base rotation of 90 deg about y, then the twist
    phi about x): d1 = (0, sin phi, -cos phi), d2 = (0, cos phi, sin phi),
    d3 = (1, 0, 0). The zeros and the ones are then exact, also after rounding
    to float32, which the quaternion route only gets to ~1e-16.
    """
    phi = _twist_angles(node_count, twists)
    c, s = np.cos(phi), np.sin(phi)
    zero, one = np.zeros_like(phi), np.ones_like(phi)
    d1 = np.stack([zero, s, -c], axis=-1)
    d2 = np.stack([zero, c, s], axis=-1)
    d3 = np.stack([one, zero, zero], axis=-1)
    frames = np.stack([d1, d2, d3], axis=-1)
    assert np.allclose(frames, quat_matrix(rest_orientations(node_count, twists)), atol=1e-12)
    return frames


def rest_relative_orientations(node_count, twists):
    """q0_i^-1 * q0_{i+1} for i = 0..N-3, (N-2, 4) float64.

    Consecutive rest frames differ by the twist step about the material axis d3:
    (0, 0, sin(dphi/2), cos(dphi/2)), written out so its zero components are exact.
    The twist is uniform, so dphi is computed once and every node gets the
    bit-identical quaternion (differences of the absolute angles would scatter
    by an ulp), and the rest curvature and moment are the same on every node.
    """
    half = np.full(node_count - 2, 0.5 * (2.0 * np.pi * twists / (node_count - 1)))
    rel = np.stack([np.zeros_like(half), np.zeros_like(half), np.sin(half), np.cos(half)], axis=-1)
    q0 = rest_orientations(node_count, twists)
    assert np.allclose(rel, quat_mul(quat_conj(q0[:-1]), q0[1:]), atol=1e-12)
    return rel


def reference_offsets(ref_vecs, frames, dl):
    """R0^T ref - dl e3 per edge (N-1, 3), float64, with dl as the simulation's float32 value.

    The shader adds R0^T (u_{i+1} - u_i) to this to get the small vector
    R0^T r - dl e3, from which it evaluates strain and stretch without
    subtracting anything close to 1 from 1.
    """
    a = np.einsum('nkj,nk->nj', frames, np.asarray(ref_vecs, dtype=np.float64))   # R0^T ref
    a[:, 2] -= float(np.float32(dl))
    return a


# Stretch dependence of the cross-section: eta(e) = 1 + ETA_SLOPE (e - 1) (Poisson ratio 0.3);
# stiffness scales with eta^2 (area), bending/twist with eta^4, inertia with eta^2.
ETA_SLOPE = -0.3
FORCE_LAWS = ("legacy", "variational")


def axial_force_per_EA(s, force_law):
    """Axial force / EA of a straight edge at stretch s = e - 1.
    legacy:      eta^2 s                 (stiffness eta^2 K, without its derivative)
    variational: eta^2 s + eta eta' s^2  (= d/ds of 1/2 eta^2 s^2, the stored energy)"""
    eta = 1.0 + ETA_SLOPE * s
    if force_law == "legacy":
        return eta * eta * s
    return eta * eta * s + eta * ETA_SLOPE * s * s


def _rest_twist_energy_density(node_count, tuned_length, twists, GJ, s):
    """kappa0^T K_bt kappa0 at rest (only twist curvature is non-zero there), for rest stretch s."""
    if twists == 0.0:
        return 0.0
    dl = tuned_length / ((node_count - 1) * (1.0 + s))
    half_step = 0.5 * (2.0 * np.pi * twists / (node_count - 1))
    kappa3 = 2.0 * np.sin(half_step) / dl
    return GJ * kappa3 ** 2


def rest_stretch(tension_force, EA, force_law, node_count=None, tuned_length=None, twists=0.0, GJ=0.0):
    """Rest stretch s0 such that the axial force in the interior of the string equals the
    configured tension. For the variational law that axial force includes the stretch
    dependence of the rest twist energy: each interior edge belongs to two bending regions,
    each contributing eta' eta_v^3 kappa0^T K_bt kappa0."""
    coeffs = [0.09, -0.6, 1.0, -tension_force / EA] if force_law == "legacy" else \
             [0.18, -0.9, 1.0, -tension_force / EA]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    s0 = float(np.min(real_roots[real_roots >= 0]))
    if force_law == "legacy" or twists == 0.0:
        return s0

    def residual(s):
        B0 = _rest_twist_energy_density(node_count, tuned_length, twists, GJ, s)
        eta = 1.0 + ETA_SLOPE * s
        return EA * axial_force_per_EA(s, force_law) + 2.0 * ETA_SLOPE * eta ** 3 * B0 - tension_force

    for _ in range(50):
        h = 1e-9 * max(s0, 1e-6)
        step = residual(s0) / ((residual(s0 + h) - residual(s0 - h)) / (2.0 * h))
        s0 -= step
        if abs(step) < 1e-16:
            break
    return s0


def static_axial_equilibrium(ref_spacing, dl, EA, B0, force_law, node_count, tol=1e-12):
    """Longitudinal displacements u_x (node_count,) of the straight rest state for which every
    interior node force vanishes, in float64 with the float32 values the GPU holds (ref_spacing,
    dl). Only needed when the rest twist energy depends on stretch (variational law, twisted
    string): the end edges belong to one bending region instead of two, so they carry a
    different axial force at uniform spacing. Returns zeros otherwise."""
    u = np.zeros(node_count)
    if force_law == "legacy" or B0 == 0.0:
        return u
    from scipy.linalg import solve_banded
    h, dl = float(ref_spacing), float(dl)

    def edge_force(u):
        s = (h + np.diff(u)) / dl - 1.0
        eta = 1.0 + ETA_SLOPE * s
        f = EA * axial_force_per_EA(s, force_law)
        eta_v = 1.0 + ETA_SLOPE * 0.5 * (s[:-1] + s[1:])
        c = ETA_SLOPE * eta_v ** 3 * B0                       # per bending region (edges i, i+1)
        f[:-1] += c
        f[1:] += c
        # d f_e / d s_e, own stretch only (the region cross-terms are small; used as preconditioner)
        k = EA * (eta * eta - 1.2 * eta * s + 0.09 * s * s) if force_law == "variational" else \
            EA * (eta * eta - 0.6 * eta * s)
        return f, k

    for _ in range(100):
        f, k = edge_force(u)
        F = f[1:] - f[:-1]                                    # interior nodes 1..N-2
        if np.max(np.abs(F)) < tol * EA * 1e-3:
            break
        k = k / dl
        ab = np.zeros((3, node_count - 2))
        ab[0, 1:] = k[1:-1]                                   # dF_j/du_{j+1} = k_j
        ab[1, :] = -(k[1:] + k[:-1])                          # dF_j/du_j
        ab[2, :-1] = k[1:-1]                                  # dF_j/du_{j-1} = k_{j-1}
        u[1:-1] -= solve_banded((1, 1), ab, F)
    return u


def generate_straight_rod(
    node_count, tuned_length, tension_force, core_radius, winding_radius,
    E_core, G_core, rho_core, E_winding, G_winding, rho_winding, packing_factor, mu, twists,
    force_law="variational", torsional_wave_speed=None
):
    """force_law: "variational" (force = gradient of the stored energy, including the
    stretch dependence of the cross-section) or "legacy" (runs before 2026-10-18).
    torsional_wave_speed (m/s, measured on the string at tension): if given, the torsional
    stiffness is GJ = c_t^2 rho I_p (rho I_p: the rod's polar inertia per length from core and
    winding) instead of the core's G J, i.e. the winding's contribution is taken from the
    measurement. Then torsional_fundamental gives c_t / (2 L)."""
    assert force_law in FORCE_LAWS, force_law
    core_area = np.pi * core_radius**2
    EA = E_core * core_area

    I_core = rho_core * core_radius**4 / 4.0
    I_winding = rho_winding * ((core_radius + 2.0 * winding_radius)**4 - core_radius**4) * packing_factor / 4.0
    I1 = I_core + I_winding
    I3 = I1 * 2.0
    if torsional_wave_speed is None:
        GJ = G_core * (np.pi * core_radius**4 / 2.0)   # same evaluation order as before
    else:
        GJ = float(torsional_wave_speed) ** 2 * np.pi * I3

    dilatation = rest_stretch(tension_force, EA, force_law, node_count, tuned_length, twists, GJ)
    rest_length = tuned_length / (1 + dilatation)
    dl = rest_length / float(node_count - 1)

    core_mass = float(rho_core * core_area * dl)
    winding_area = np.pi * ((core_radius + 2.0 * winding_radius)**2 - core_radius**2)
    winding_mass = float(rho_winding * winding_area * dl * packing_factor)
    mass = core_mass + winding_mass

    inertia = (np.pi * dl * np.array([I1, I1, I3], dtype=np.float32))

    alpha = 4.0 / 3.0
    I_c = np.pi * core_radius**4 / 4.0
    I_w = np.pi * ((core_radius + 2.0 * winding_radius)**4 - core_radius**4) * packing_factor / 4.0
    EI = E_core * I_c + (E_winding * I_w * mu)

    K_se = np.array([G_core * core_area * alpha,
                     G_core * core_area * alpha,
                     E_core * core_area], dtype=np.float32)

    K_bt = np.array([EI, EI, GJ], dtype=np.float32)

    # Edge orientations are stored as the deviation p from the rest orientation
    # q0 (q = q0 * p), so they start at the identity; the rest frame itself is
    # handed over as directors and as the relative rest rotation per node.
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    frames = rest_frames(node_count, twists)
    ref_vecs = reference_vectors(node_count, tuned_length)
    offsets = reference_offsets(ref_vecs, frames, dl)

    edges = [(identity, v, np.zeros(3), np.zeros(3), o, [f[:, 0], f[:, 1], f[:, 2]])
             for v, o, f in zip(ref_vecs, offsets, frames)]
    # Ghost edge (never strained; its frame is only read for safety)
    edges.append((identity, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), edges[-1][5]))

    rel = rest_relative_orientations(node_count, twists)
    # Start from the static equilibrium of the discrete model (zero displacement except
    # for a twisted string under the variational law, see static_axial_equilibrium).
    kappa0 = 2.0 / np.float32(dl) * rel.astype(np.float32)[0, :3].astype(np.float64) if len(rel) else np.zeros(3)
    B0 = float(np.sum(K_bt.astype(np.float64) * kappa0 ** 2)) if twists else 0.0
    u_x = static_axial_equilibrium(ref_vecs[0, 0], np.float32(dl), float(K_se[2]), B0, force_law, node_count)
    nodes = [
        (np.array([u_x[i], 0.0, 0.0]), np.zeros(3), identity, rel[i] if i < len(rel) else identity)
        for i in range(node_count)
    ]

    return nodes, edges, dl, mass, inertia, K_se, K_bt


def resolve_rod_params(rod_params, node_count, force_law="variational"):
    """Copy of rod_params with derived entries filled in.

    "linear_density" (kg/m, of the string at tension): the winding packing factor is solved
    so that the rod's mass per tuned length equals it. Mass is linear in the packing factor
    and is conserved under the rest stretch, so the per-node mass m gives m (N - 1) / L
    per tuned length. The entry is removed and "packing_factor" set; any given packing
    factor must then be None.
    """
    params = dict(rod_params)
    f_target = params.pop("fundamental_frequency", None)
    if f_target is not None:
        # "fundamental_frequency" (Hz, plain strings): the tension is solved so that the flexible-
        # string fundamental of the rod, sqrt(T / rho_L) / (2 L) with its own mass per tuned length,
        # equals it; rho_L depends on T only through the rest stretch, so the fixed point converges.
        assert params.get("tension_force") is None, "give fundamental_frequency or tension_force, not both"
        assert "linear_density" not in params
        L = params["tuned_length"]
        tension = 1.0
        for _ in range(100):
            mass = generate_straight_rod(node_count, **{**params, "tension_force": tension}, force_law=force_law)[3]
            new = mass * (node_count - 1) / L * (2.0 * L * f_target) ** 2
            if abs(new - tension) <= 1e-12 * new:
                break
            tension = new
        params["tension_force"] = float(new)
        return params
    target = params.pop("linear_density", None)
    if target is None:
        return params
    assert params.get("packing_factor") is None, "give linear_density or packing_factor, not both"

    def per_length(phi):
        mass = generate_straight_rod(node_count, **{**params, "packing_factor": phi}, force_law=force_law)[3]
        return mass * (node_count - 1) / params["tuned_length"]

    m0, m1 = per_length(0.0), per_length(1.0)
    phi = (target - m0) / (m1 - m0)
    assert 0.0 < phi <= 1.0, f"linear density {target} kg/m needs packing factor {phi}"
    params["packing_factor"] = float(phi)
    return params


def fundamental_frequency(tension_force, node_mass, node_count, length):
    """f1 = sqrt(T / rho_L) / (2 L) of the flexible string, rho_L = m (N - 1) / L."""
    rho_L = float(node_mass) * (node_count - 1) / float(length)
    return np.sqrt(float(tension_force) / rho_L) / (2.0 * float(length))


def constant_q_fit(q, f1, length, modes=(1, 10)):
    """b1, b2 such that the transverse decay rate b1 + b2 k_n^2 (k_n = n pi / L) equals the
    constant-Q rate pi f_n / Q, f_n = n f1, at the two given mode numbers. Between and
    beyond them the Q of this two-parameter law differs from q (see damping_q_profile)."""
    n_a, n_b = modes
    k_a, k_b = n_a * np.pi / length, n_b * np.pi / length
    r_a, r_b = np.pi * n_a * f1 / q, np.pi * n_b * f1 / q
    b2 = (r_b - r_a) / (k_b ** 2 - k_a ** 2)
    b1 = r_a - b2 * k_a ** 2
    return float(b1), float(b2)


def two_mode_rate_fit(rates, length, modes=(1, 10)):
    """b1, b2 such that b1 + b2 k_n^2 (k_n = n pi / L) equals the given amplitude decay rates
    (s^-1) at the two mode numbers."""
    n_a, n_b = modes
    k_a, k_b = n_a * np.pi / length, n_b * np.pi / length
    r_a, r_b = rates
    b2 = (r_b - r_a) / (k_b ** 2 - k_a ** 2)
    b1 = r_a - b2 * k_a ** 2
    return float(b1), float(b2)


def valette_q(n, f1, linear_density, tension, EI, length, model):
    """Quality factor of transverse mode n from the loss model of Valette & Cuesta (Mecanique de la
    corde vibrante, 1993) in the form used by Issanchou, Bilbao, Le Carrou, Touze & Doare (JSV 393,
    2017, Eq. 12):
        1/Q = R / (2 pi mu nu) + 4 pi^2 mu EI delta_ve nu^2 / T^2 + 1/Q_te,
        R = 2 pi eta_air + 2 pi d sqrt(pi eta_air rho_air nu),
    with nu the stiff-string frequency n f1 sqrt(1 + B n^2), B = pi^2 EI / (T L^2).
    model: {"diameter", "delta_ve", "q_te_inv", "eta_air", "rho_air"}."""
    n = np.asarray(n, dtype=float)
    B = np.pi ** 2 * EI / (tension * length ** 2)
    nu = n * f1 * np.sqrt(1.0 + B * n ** 2)
    mu = float(linear_density)
    eta, rho_a, d = model["eta_air"], model["rho_air"], model["diameter"]
    R = 2.0 * np.pi * eta + 2.0 * np.pi * d * np.sqrt(np.pi * eta * rho_a * nu)
    q_inv = (R / (2.0 * np.pi * mu * nu)
             + 4.0 * np.pi ** 2 * mu * EI * model["delta_ve"] * nu ** 2 / tension ** 2
             + model["q_te_inv"])
    return 1.0 / q_inv, nu


def damping_q_profile(b1, b2, f1, length, modes):
    """Q_n = pi f_n / (b1 + b2 k_n^2) with f_n = n f1."""
    n = np.asarray(modes, dtype=float)
    return np.pi * n * f1 / (b1 + b2 * (n * np.pi / length) ** 2)


def damping_coefficients(damping, node_mass, inertia, K_bt=None, dl=None, length=None,
                         tension=None, node_count=None):
    """Viscous damping coefficients for the shader from physical decay rates.

    damping: {"b1": s^-1, "b2": m^2/s, "torsion_rate": s^-1}
      transverse_q  instead of b1, b2: a constant transverse Q, matched at the mode
                    numbers "q_fit_modes" (default (1, 10)) by constant_q_fit; needs
                    tension and node_count for the fundamental
      b1            frequency-independent loss; transverse mode n decays at
      b2            ... b1 + b2 k_n^2 (k_n = n pi / L). The tilt damping acts on the
                    slope rate because the frame follows the tangent, i.e. as the
                    2 b2 d^3y/dt dx^2 term of Bensa, Bilbao, Kronland-Martinet & Smith,
                    JASA 114(2), 2003 (their Table I: C2 b1 = 0.25 s^-1,
                    b2 = 7.5e-5 m^2/s; C4 1.1 / 2.7e-4; C7 9.17 / 2.1e-3)
      torsion_rate  amplitude decay rate of torsional motion, or instead
      torsion_q     quality factor of the fundamental torsional mode, converted with
                    f_t = sqrt(GJ dl / I3) / (2 L) (needs K_bt, dl, length): rate = pi f_t / Q.
                    Wound steel double-bass strings: Q ~ 15-20 (Wollman, Smith & Wolfe, ISMA 2010)
    Returns (linear, (tilt, tilt, torsion)): a coefficient c on a degree of freedom
    with mass or inertia M gives the amplitude decay rate c / (2 M).
    """
    m = float(node_mass)
    I3 = float(np.ravel(inertia)[2])
    if "torsion_q_fit_modes" in damping:
        # viscous part of the two-term torsional law (see twist_rate_damping)
        f_t = torsional_fundamental(K_bt, I3, dl, length)
        b1_t, _ = constant_q_fit(damping["torsion_q"], f_t, length, tuple(damping["torsion_q_fit_modes"]))
        c_lin, c_ang = damping_coefficients({k: v for k, v in damping.items()
                                             if k not in ("torsion_q", "torsion_q_fit_modes")}
                                            | {"torsion_rate": b1_t},
                                            node_mass, inertia, K_bt, dl, length, tension, node_count)
        return c_lin, c_ang
    if "transverse_q" in damping:
        f1 = fundamental_frequency(tension, m, node_count, length)
        b1, b2 = constant_q_fit(damping["transverse_q"], f1, length,
                                tuple(damping.get("q_fit_modes", (1, 10))))
    elif "transverse_valette" in damping:
        # decay rate pi nu_n / Q_n of the Valette model, matched at the fit modes
        modes = tuple(damping.get("q_fit_modes", (1, 10)))
        f1 = fundamental_frequency(tension, m, node_count, length)
        rho_L = m * (node_count - 1) / float(length)
        q, nu = valette_q(modes, f1, rho_L, tension, float(np.ravel(K_bt)[0]), length,
                          damping["transverse_valette"])
        b1, b2 = two_mode_rate_fit(np.pi * nu / q, length, modes)
    else:
        b1, b2 = damping["b1"], damping["b2"]
    c_lin = 2.0 * b1 * m
    c_tilt = 2.0 * b2 * m
    if "torsion_q" in damping:
        f_t = torsional_fundamental(K_bt, I3, dl, length)
        torsion_rate = np.pi * f_t / damping["torsion_q"]
    else:
        torsion_rate = damping["torsion_rate"]
    c_tor = 2.0 * torsion_rate * I3
    return c_lin, (c_tilt, c_tilt, c_tor)


def twist_rate_damping(damping, inertia, K_bt, dl, length):
    """Kelvin-Voigt twist-rate damping D3 (N m^2 s) for the shader (0 if not requested).

    With "torsion_q_fit_modes" the torsional decay rate is b1_t + b2_t k_n^2, matched to the
    constant torsional Q ("torsion_q") at the two mode numbers like the transverse law
    (constant_q_fit with the torsional fundamental). The viscous torsion coefficient gives
    b1_t (damping_coefficients); the moment D3 d(kappa_3)/dt gives b2_t = D3 / (2 rho I_p),
    rho I_p = I3_edge / dl, so D3 = 2 b2_t I3_edge / dl.
    """
    if "torsion_q_fit_modes" not in damping:
        return 0.0
    I3 = float(np.ravel(inertia)[2])
    f_t = torsional_fundamental(K_bt, I3, dl, length)
    _, b2_t = constant_q_fit(damping["torsion_q"], f_t, length, tuple(damping["torsion_q_fit_modes"]))
    return 2.0 * b2_t * I3 / float(dl)


def torsional_fundamental(K_bt, I3_edge, dl, length):
    """Fundamental torsional frequency (Hz) of the discrete rod: wave speed
    sqrt(GJ / (rho I3)) with GJ = K_bt[2] and rho I3 per length = I3_edge / dl."""
    GJ = float(np.ravel(K_bt)[2])
    return np.sqrt(GJ * float(dl) / float(I3_edge)) / (2.0 * float(length))


def legacy_damping_coefficients(dampening):
    """Runs before per-axis damping: [c_lin, c_ang] with c_ang on all three axes."""
    c_lin, c_ang = (float(x) for x in dampening)
    return c_lin, (c_ang, c_ang, c_ang)


def make_weights(node_count, node_index=None, width_count=4, order=10):
    x = np.arange(node_count, dtype=np.float32)

    sigma = width_count / 2.0
    data = np.exp(-((np.abs(x - node_index) / sigma) ** order))

    data[data < 1e-4] = 0.0

    return data
