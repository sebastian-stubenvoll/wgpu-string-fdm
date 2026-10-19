"""Bow contact parameters: friction curve, Woodhouse's thermal friction model and its kernels.

Sources
  Woodhouse, Acta Acustica 89 (2003) 355-368: Eq. 1 (heat balance of the contact), Eq. 2 (friction
    curve), Eq. 3 (thermal parameter lambda0 = sqrt(K rho c)), Table I (material and contact values),
    sec. 2.1 (calibration conditions of Fig. 3), sec. 2.3 (order: temperature, then limiting
    friction, then force), sec. 4 (the bow is the moving surface).
  Smith & Woodhouse, J. Mech. Phys. Solids 48 (2000) 1633-1681: Eqs. 12-17 (discrete Green's
    function of one-dimensional heat flow, moving-surface history cut at 8a / (3 pi v)).
"""
import numpy as np

# Woodhouse 2003 Eq. 2: mu(v) = 0.35 + 0.4 exp(-v / 0.01) + 0.45 exp(-v / 0.1); shader order
# (mu_d, a1, v1, a2, v2)
FRICTION_CURVE = (0.35, 0.4, 0.01, 0.45, 0.1)

# Woodhouse 2003 Table I (string, bow, rosin) and the Fig. 3 calibration conditions (sec. 2.1)
TABLE_I = {
    "rho": {"string": 3000.0, "bow": 600.0, "rosin": 1010.0},
    "c": {"string": 1000.0, "bow": 1680.0, "rosin": 250.0},
    "K": {"string": 1.4, "bow": 0.2, "rosin": 0.113},
    "layer_thickness": 1e-6,
    "contact_radius_at_1N": 250e-6,      # contact area proportional to normal force
}
CALIBRATION = {"normal_force": 3.0, "contact_radius": 0.5e-3, "layer_thickness": 10e-6,
               "surface": "bow"}          # PMMA properties, "as listed under bow in Table I"

EXACT_TERMS = 32                         # f(n) summed exactly for n < EXACT_TERMS
MAX_SOE_TERMS = 32
RING = 65536                             # temperature history per contact (shader ring buffer)


def friction_curve(v, mu=FRICTION_CURVE):
    v = np.abs(v)
    return mu[0] + mu[1] * np.exp(-v / mu[2]) + mu[3] * np.exp(-v / mu[4])


def thermal_parameter(surface):
    """lambda0 = sqrt(K rho c) (Woodhouse 2003 Eq. 3)."""
    return np.sqrt(TABLE_I["K"][surface] * TABLE_I["rho"][surface] * TABLE_I["c"][surface])


def calibration_constants(mu=FRICTION_CURVE):
    """Steady sliding at the Fig. 3 conditions: the stationary surface's flux has decayed, the
    moving surface (PMMA) has been in contact for t_c = 8a / (3 pi v) (Smith & Woodhouse Eq. 16),
    so Eq. 1 gives mu(v) N v = (lambda / sqrt(t_c) + v a delta rho_r c_r) T with
    lambda = A lambda0 / sqrt(pi), A = pi a^2 (Smith & Woodhouse Eq. 14). Written as
    T(v) = mu(v) N sqrt(v) / (k1 + k2 sqrt(v)), returns (k1, k2, N)."""
    a = CALIBRATION["contact_radius"]
    lam = np.pi * a * a * thermal_parameter(CALIBRATION["surface"]) / np.sqrt(np.pi)
    k1 = lam * np.sqrt(3.0 * np.pi / (8.0 * a))
    k2 = a * CALIBRATION["layer_thickness"] * TABLE_I["rho"]["rosin"] * TABLE_I["c"]["rosin"]
    return k1, k2, CALIBRATION["normal_force"]


def calibration_temperature(v, mu=FRICTION_CURVE):
    k1, k2, N = calibration_constants(mu)
    u = np.sqrt(np.abs(v))
    return friction_curve(v, mu) * N * u / (k1 + k2 * u)


def calibrated_mu(T, mu=FRICTION_CURVE, iterations=200):
    """mu(T): the friction coefficient of the steady sliding speed whose contact temperature is T
    (T(v) is increasing). Float64 reference of the shader's bisection."""
    T = float(T)
    if T <= 0.0:
        return friction_curve(0.0, mu)
    hi = 1.0
    while calibration_temperature(hi * hi, mu) < T:
        hi *= 2.0
    lo = 0.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if calibration_temperature(mid * mid, mu) < T:
            lo = mid
        else:
            hi = mid
    return friction_curve(hi * hi, mu)


def step_response_shape(n):
    """f(n) = 2 (sqrt(n + 1) - sqrt(n)): Smith & Woodhouse's step response G_n = lambda h^(-1/2) f(n)
    (the running sum of their g_n, Eq. 13), so that sum_n g_n T_(m-n) = sum_n G_n (T_(m-n) - T_(m-n-1))."""
    n = np.asarray(n, dtype=np.float64)
    return 2.0 / (np.sqrt(n + 1.0) + np.sqrt(n))


def soe_fit(n_end, n0=EXACT_TERMS, du=0.35):
    """Sum of exponentials f(n) ~ sum_k w_k d_k^(n - n0) on n0 <= n <= n_end.

    Nodes from the exact representation f(n) = (2 / sqrt(pi)) int exp(-n x^2) (1 - exp(-x^2)) / x^2 dx
    (x = e^u, step du); decay factors d_k rounded to float32 (the shader's recursion then uses exactly
    these), weights by non-negative least squares in relative error. Returns (d32, w, max relative
    error checked on every n < n0 + 2e5 and 2e5 log-spaced n up to n_end)."""
    from scipy.optimize import nnls
    u = np.arange(0.5 * np.log(1e-3 / n_end), 0.5 * np.log(60.0 / n0) + du, du)
    d32 = np.exp(-np.exp(2.0 * u)).astype(np.float32)
    d32 = np.unique(d32[(d32 > 0.0) & (d32 < 1.0)])
    beta = -np.log(d32.astype(np.float64))
    n = np.unique(np.round(np.logspace(np.log10(n0), np.log10(n_end), 4000)))
    A = np.exp(-np.outer(n - n0, beta)) / step_response_shape(n)[:, None]
    w, _ = nnls(A, np.ones_like(n), maxiter=20000)
    keep = w > 0.0
    d32, beta, w = d32[keep], beta[keep], w[keep]
    nc = np.unique(np.concatenate([np.arange(n0, min(n0 + 200000, int(n_end)) + 1),
                                   np.round(np.logspace(np.log10(n0), np.log10(n_end), 200000))]))
    err = float(np.max(np.abs(np.exp(-np.outer(nc - n0, beta)) @ w / step_response_shape(nc) - 1.0)))
    return d32, w, err


def thermal_contact(normal_force, weight, bow_velocity, dt, run_steps, max_error=1e-3):
    """Shader parameters of the thermal model for contacts carrying weight * normal_force each.

    Per contact (Woodhouse 2003 Eq. 1 with Smith & Woodhouse's kernel): contact radius
    a = a(1 N) sqrt(w N) (area proportional to force, Table I), heat capacity C = rho_r pi a^2 delta c_r,
    convection a delta rho_r c_r |v_rel|, conduction lambda h^(-1/2) f(n) per surface with
    lambda = pi a^2 lambda0 / sqrt(pi); the bow (moving surface) history ends after
    M = round(8 a / (3 pi v_b) / h) steps. The contact size and M are those of the steady normal
    force and bow speed (also during the attack and release ramps).
    """
    N_c = float(normal_force) * float(weight)
    a = TABLE_I["contact_radius_at_1N"] * np.sqrt(N_c)
    A = np.pi * a * a
    delta, rc = TABLE_I["layer_thickness"], TABLE_I["rho"]["rosin"] * TABLE_I["c"]["rosin"]
    lam_s = A * thermal_parameter("string") / np.sqrt(np.pi)
    lam_b = A * thermal_parameter("bow") / np.sqrt(np.pi)
    t_c = 8.0 * a / (3.0 * np.pi * float(bow_velocity))
    M = int(round(t_c / dt))
    assert EXACT_TERMS <= M and M + EXACT_TERMS + 2 < RING, f"bow history window {M} steps"
    d32, w, err = soe_fit(max(run_steps, M + EXACT_TERMS + 1))
    assert len(w) <= MAX_SOE_TERMS and err < max_error, (len(w), err)
    beta = -np.log(d32.astype(np.float64))
    w_window = w * np.exp(-beta * (M - EXACT_TERMS))
    k1, k2, Nc = calibration_constants()
    return {
        "contact_radius": a, "window_steps": M, "window_time": t_c, "soe_terms": len(w), "soe_error": err,
        "c_over_h": rc * A * delta / dt,
        "conv_coef": a * delta * rc,
        "lam_string": lam_s / np.sqrt(dt),
        "lam_bow": lam_b / np.sqrt(dt),
        "f_window": float(step_response_shape(M)),
        "exact": step_response_shape(np.arange(EXACT_TERMS)),
        "decay": d32, "weight": w, "weight_window": w_window,
        "cal_k1": k1, "cal_k2": k2, "cal_force": Nc,
    }


class ContactTemperature:
    """Float64 reference of the shader's contact temperature update (one contact).

    Backward difference of Woodhouse's Eq. 1 at step m, with the friction power P and slip speed
    of step m - 1 (temperature first, then friction):
      C (T_m - T_(m-1)) / h = P - conv |v| T_m - Q_m,
      Q_m = lam_s [f(0) dT_m + F_m] + lam_b [f(0) dT_m + F_m - W_m + f(M) T_(m-M)],
    F_m = sum_(n>=1) f(n) dT_(m-n), W_m = sum_(n>=M) f(n) dT_(m-n), dT_j = T_j - T_(j-1), T_j = 0 for j <= 0.
    direct=True evaluates F and W as plain sums (Smith & Woodhouse Eq. 17); otherwise exact for
    n < n0 and by the sum of exponentials beyond, as the shader does."""

    def __init__(self, p, direct=False):
        self.p, self.direct = p, direct
        self.T = [0.0]                                     # T_0 = 0
        K = len(p["weight"])
        self.S, self.D = np.zeros(K), np.zeros(K)

    def dT(self, j):
        if j <= 0:
            return 0.0
        return self.T[j] - self.T[j - 1]

    def step(self, heat_power, slip_speed):
        p = self.p
        m = len(self.T)                                    # index of the new temperature
        n0, M = EXACT_TERMS, p["window_steps"]
        if self.direct:
            F = sum(step_response_shape(n) * self.dT(m - n) for n in range(1, m))
            W = sum(step_response_shape(n) * self.dT(m - n) for n in range(M, m))
        else:
            F = sum(p["exact"][n] * self.dT(m - n) for n in range(1, n0))
            self.S = p["decay"].astype(np.float64) * self.S + p["weight"] * self.dT(m - n0)
            self.D = p["decay"].astype(np.float64) * self.D + p["weight_window"] * self.dT(m - M)
            F += self.S.sum()
            W = self.D.sum()
        T_prev = self.T[-1]
        T_M = self.T[m - M] if m - M >= 0 else 0.0
        lam = p["lam_string"] + p["lam_bow"]
        rhs = (p["c_over_h"] * T_prev + heat_power + 2.0 * lam * T_prev - lam * F
               + p["lam_bow"] * (W - p["f_window"] * T_M))
        T = rhs / (p["c_over_h"] + p["conv_coef"] * abs(slip_speed) + 2.0 * lam)
        self.T.append(T)
        return T


FRICTION_MODELS = {"curve": 1, "thermal": 2}


def contact_nodes(node_count, beta, width_nodes=1):
    """Contact nodes centred on the bow position beta (fraction of the length from node 0, the
    bridge end) and equal weights; width_nodes = 1 is a point contact."""
    j = int(round(beta * (node_count - 1)))
    nodes = list(range(j - (width_nodes - 1) // 2, j - (width_nodes - 1) // 2 + width_nodes))
    return nodes, [1.0 / width_nodes] * width_nodes


def configure_bow(sim, bow, node_count, dt, run_steps, string_radius):
    """Set up the bow on a Simulation from a parameter dict:
      friction  "curve" (Woodhouse 2003 Eq. 2) or "thermal" (Eq. 1 with the calibrated mu(T))
      torsion   True: the contact sees the surface speed of the string's rotation and exerts the
                friction couple; False: inhibited
      beta, width_nodes, normal_force (N), velocity (m/s), ramp_time, start_time, stop_time (s)
    Returns the derived values to record (contact nodes, thermal parameters)."""
    nodes, weights = contact_nodes(node_count, bow["beta"], bow.get("width_nodes", 1))
    ramp = int(round(bow.get("ramp_time", 1e-3) / dt))
    start = int(round(bow.get("start_time", 0.0) / dt))
    stop = int(round(bow["stop_time"] / dt)) if bow.get("stop_time") is not None else 2**31 - ramp - 1
    derived = {"contact_nodes": nodes, "contact_weights": weights, "ramp_steps": ramp,
               "start_step": start, "stop_step": stop, "friction_curve": list(FRICTION_CURVE)}
    model = FRICTION_MODELS[bow["friction"]]
    if model == 2:
        p = thermal_contact(bow["normal_force"], weights[0], bow["velocity"], dt, run_steps)
        sim.set_bow_thermal(p["c_over_h"], p["conv_coef"], p["lam_string"], p["lam_bow"], p["window_steps"],
                            p["f_window"], list(p["exact"]), list(p["decay"].astype(float)), list(p["weight"]),
                            list(p["weight_window"]), [p["cal_k1"], p["cal_k2"], p["cal_force"]])
        derived["thermal"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    sim.set_bow(model, bool(bow.get("torsion", True)), nodes, weights, string_radius, bow["normal_force"],
                bow["velocity"], list(FRICTION_CURVE), start, ramp, stop)
    return derived
