"""Synthetic stand-in for extract_data(), used by `post_process.py --test`.

The string is an analytic modal sum on a coarse grid using the real rod
parameters from config.py: a hammer-like initial-velocity pulse on a stiff
string (Y, with a weaker, slightly detuned Z polarisation), plus a few
longitudinal (X) modes whose amplitude is fed from the transverse motion with
a slow modulation so energy visibly moves back and forth. All energies are
computed modally, so they are self-consistent with the displacement field.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from config import SIM_CONFIG, ROD_PARAMS
from utils import generate_straight_rod
from plots.energy import ENERGY_DENSITY_BLOCK, block_mean, block_centers
from plots.dispersion import (modal_projection, kspace_grid, kspace_projection, StreamingModalSpectra,
                              StreamingKOmegaSpectra, finalize_dispersion)


def make_preview_data(node_count=128, duration=1.0, chunk=512, n_exc_chunks=10,
                      inspect_nodes=(3, 64, 100), n_modes=256, nperseg=8192, seed=0):
    rng = np.random.default_rng(seed)

    dt = SIM_CONFIG["dt"]
    oversamp = SIM_CONFIG["oversampling_factor"]
    fs = 1.0 / (dt * oversamp)

    _, _, dl, m_node, inertia, K_se, K_bt = generate_straight_rod(node_count, **ROD_PARAMS)
    L = ROD_PARAMS["tuned_length"]
    T = ROD_PARAMS["tension_force"]
    mu = m_node * (node_count - 1) / L
    EI, EA = float(K_bt[0]), float(K_se[2])
    rhoI1, rhoI3 = float(inertia[0]) / dl, float(inertia[2]) / dl

    x = np.linspace(0.0, L, node_count)
    weights = np.ones(node_count)
    weights[[0, -1]] = 0.5

    # --- Transverse modes (Y and Z) --------------------------------------
    n_t = np.arange(1, node_count - 1)
    k_t = n_t * np.pi / L
    w_t = np.sqrt(T / mu * k_t ** 2 + EI / mu * k_t ** 4)
    keep = w_t < 0.45 * 2 * np.pi * fs
    n_t, k_t, w_t = n_t[keep], k_t[keep], w_t[keep]
    sig_t = 0.8 + 2e-9 * w_t ** 2                     # frequency-dependent damping

    x0 = SIM_CONFIG["hammer_node"] / (SIM_CONFIG["node_count"] - 1) * L
    width = 0.01
    v0 = 0.5                                          # peak pulse velocity (m/s)
    b_y = v0 * np.sin(k_t * x0) * np.exp(-(k_t * width) ** 2 / 4) / w_t
    b_z = 0.2 * b_y * (1 + 0.05 * rng.standard_normal(b_y.size))
    w_z = w_t * 1.001                                 # detuned polarisation

    # --- Longitudinal modes (X) ------------------------------------------
    n_l = np.arange(1, 9)
    k_l = n_l * np.pi / L
    w_l = np.sqrt(EA / mu) * k_l
    b_l = 1.0 / n_l                                   # shape; scaled below


    # --- Torsion (single mode) -------------------------------------------
    w_tor = 2 * np.pi * 900.0
    phi0 = 1e-3

    S_t = np.sin(np.outer(x, k_t))                    # (nodes, modes)
    C_t = np.cos(np.outer(x, k_t)) * k_t              # d/dx of mode shapes
    S2_t = -S_t * k_t ** 2                            # d2/dx2
    S_l = np.sin(np.outer(x, k_l))
    C_l = np.cos(np.outer(x, k_l)) * k_l

    # Longitudinal amplitude from the geometric-nonlinearity estimate
    # E_L / E_T ~ (EA / 4T) <u'^2>: the transverse slopes stretch the string,
    # and that stretch is what feeds longitudinal motion. <u'^2> is averaged
    # over the string and the first 5 ms.
    t_probe = np.linspace(0.0, 0.005, 64)[:, None]
    slopes = (b_y * np.exp(-sig_t * t_probe) * np.sin(w_t * t_probe)) @ C_t.T
    ratio = EA / (4.0 * T) * np.mean(slopes ** 2)
    e_trans0 = 0.5 * mu * 0.5 * L * (np.sum((b_y * w_t) ** 2) + np.sum((b_z * w_z) ** 2))
    t_env = np.linspace(0.0, duration, 2000)
    g_max = np.max((1 - np.exp(-t_env / 0.05)) * np.exp(-1.5 * t_env) * (1 + 0.6 * np.sin(2 * np.pi * 3.0 * t_env)))
    e_long_unit = 0.5 * mu * 0.5 * L * np.sum((b_l * w_l) ** 2) * g_max ** 2
    b_l = b_l * np.sqrt(ratio * e_trans0 / e_long_unit)

    def modal_state(t):
        """Modal amplitudes and velocities at times t (frames,)."""
        t = t[:, None]
        env_t = np.exp(-sig_t * t)
        a_y = b_y * env_t * np.sin(w_t * t)
        v_y = b_y * env_t * (w_t * np.cos(w_t * t) - sig_t * np.sin(w_t * t))
        a_z = b_z * env_t * np.sin(w_z * t)
        v_z = b_z * env_t * (w_z * np.cos(w_z * t) - sig_t * np.sin(w_z * t))
        # Longitudinal: builds up from the transverse motion, slowly modulated,
        # then decays (amplitude envelope is slow vs. the carrier).
        g = (1 - np.exp(-t / 0.05)) * np.exp(-1.5 * t) * (1 + 0.6 * np.sin(2 * np.pi * 3.0 * t))
        a_x = b_l * g * np.sin(w_l * t)
        v_x = b_l * g * w_l * np.cos(w_l * t)
        phi = phi0 * np.exp(-3.0 * t[:, 0]) * np.sin(w_tor * t[:, 0])
        dphi = phi0 * np.exp(-3.0 * t[:, 0]) * w_tor * np.cos(w_tor * t[:, 0])
        return a_y, v_y, a_z, v_z, a_x, v_x, phi, dphi

    n_frames = int(duration * fs)
    n_chunks = int(np.ceil(n_frames / chunk))
    half_L = 0.5 * L
    static_ext = 1e-3                                 # static tension baseline (J)

    energy = {k: [] for k in ("t_ke", "r_ke", "bt_pe", "ss_pe",
                              "t_ke_l", "r_ke_t", "bt_pe_tw", "ss_pe_ex")}
    node_data = {n: {k: [] for k in ("pos", "vel", "mom", "quats", "omega")} for n in inspect_nodes}
    field_chunks = []
    density_chunks, density_t, p_damp_chunks, modal_ke_chunks, modal_ke_t = [], [], [], [], []
    dx = L / (node_count - 1)
    modal_acc = StreamingModalSpectra(fs, nperseg=nperseg)
    kw_acc = StreamingKOmegaSpectra(fs, nperseg=nperseg)
    n_modes = min(n_modes, node_count - 2)
    grid = kspace_grid(node_count, L / (node_count - 1), k_max=n_modes * np.pi / L)
    q_base = R.from_rotvec([0.0, np.pi / 2, 0.0])

    for c in range(n_chunks):
        t = np.arange(c * chunk, min((c + 1) * chunk, n_frames)) / fs
        a_y, v_y, a_z, v_z, a_x, v_x, phi, dphi = modal_state(t)

        # Modal energies (sum over orthogonal modes; integral of sin^2 = L/2).
        ke_l = 0.5 * mu * half_L * np.sum(v_x ** 2, axis=1)
        ke_tr = 0.5 * mu * half_L * (np.sum(v_y ** 2, axis=1) + np.sum(v_z ** 2, axis=1))
        tilt_ke = 0.5 * rhoI1 * half_L * (np.sum((k_t * v_y) ** 2, axis=1)
                                          + np.sum((k_t * v_z) ** 2, axis=1))
        tor_ke = 0.5 * rhoI3 * half_L * dphi ** 2
        bend = 0.5 * EI * half_L * (np.sum((k_t ** 2 * a_y) ** 2, axis=1)
                                    + np.sum((k_t ** 2 * a_z) ** 2, axis=1))
        twist = 0.5 * (rhoI3 * w_tor ** 2) * half_L * phi ** 2 + 1e-9
        tens = 0.5 * T * half_L * (np.sum((k_t * a_y) ** 2, axis=1)
                                   + np.sum((k_t * a_z) ** 2, axis=1))
        ext = 0.5 * EA * half_L * np.sum((k_l * a_x) ** 2, axis=1) + static_ext

        energy["t_ke"].append(ke_tr + ke_l)
        energy["t_ke_l"].append(ke_l)
        energy["r_ke"].append(tilt_ke + tor_ke)
        energy["r_ke_t"].append(tor_ke)
        energy["bt_pe"].append(bend + twist)
        energy["bt_pe_tw"].append(twist)
        # As in the simulation (shear stiffness >> tension), the frame follows
        # the tangent, so the transverse wave's tension energy shows up as axial
        # strain; only a tiny residual is genuine shear.
        shear = 1e-3 * tens
        energy["ss_pe"].append(tens + ext)
        energy["ss_pe_ex"].append(tens - shear + ext)

        # Viscous damping power, matching the modal decay: <P> = 2 sigma E = 4 sigma <KE>.
        p_damp_chunks.append(4.0 * 0.5 * mu * half_L * (v_y ** 2 + v_z ** 2) @ sig_t)

        # Full grid field for this chunk only: (frames, nodes, 3) as X, Y, Z.
        pos = np.stack([a_x @ S_l.T, a_y @ S_t.T, a_z @ S_t.T], axis=2).astype(np.float32)
        vel_grid = np.stack([v_x @ S_l.T, v_y @ S_t.T, v_z @ S_t.T], axis=2)

        # Per-element energy (KE + tension, bending and extension PE + an even
        # share of the static baseline), block-averaged like the real worker.
        elem = (0.5 * m_node * np.sum(vel_grid ** 2, axis=2)
                + 0.5 * T * ((a_y @ C_t.T) ** 2 + (a_z @ C_t.T) ** 2) * dx
                + 0.5 * EI * ((a_y @ S2_t.T) ** 2 + (a_z @ S2_t.T) ** 2) * dx
                + 0.5 * EA * (a_x @ C_l.T) ** 2 * dx) * weights + static_ext / node_count
        density_chunks.append(block_mean(elem, ENERGY_DENSITY_BLOCK))
        density_t.append(block_centers(c * chunk, t.size, ENERGY_DENSITY_BLOCK, 1.0 / fs))

        # Chunk-averaged KE per spatial mode, exactly as the real worker does it.
        adot = modal_projection(vel_grid, n_modes).astype(np.float64)
        ke_modes = 0.25 * m_node * (node_count - 1) * np.mean(adot ** 2, axis=0)
        modal_ke_chunks.append(np.stack([ke_modes[:, 1] + ke_modes[:, 2], ke_modes[:, 0]], axis=1))
        modal_ke_t.append((c * chunk + 0.5 * (t.size - 1)) / fs)
        if c >= n_exc_chunks:
            modal_acc.feed(modal_projection(pos, n_modes))
            kw_acc.feed(kspace_projection(pos, grid))
        if c < n_exc_chunks:
            field_chunks.append(pos[:, :, 1:3].copy())

        for n in inspect_nodes:
            vel = np.stack([v_x @ S_l[n], v_y @ S_t[n], v_z @ S_t[n]], axis=1)
            dy, dz = a_y @ C_t[n], a_z @ C_t[n]
            dvy, dvz = v_y @ C_t[n], v_z @ C_t[n]
            curv = np.stack([-(a_z @ S2_t[n]), a_y @ S2_t[n], np.zeros_like(t)], axis=1)
            rotvec = np.stack([phi, -dz, dy], axis=1)
            quats = (R.from_rotvec(rotvec) * q_base).as_quat()  # (x, y, z, w)
            omega = np.stack([dphi, -dvz, dvy], axis=1)
            nd = node_data[n]
            nd["pos"].append(pos[:, n, :])
            nd["vel"].append(vel)
            nd["mom"].append(curv)
            nd["quats"].append(quats)
            nd["omega"].append(omega)

    data = {
        "time": np.arange(n_frames) / fs,
        "fs": fs,
        "excitation_cutoff_frame": min(n_exc_chunks * chunk, n_frames),
        "field_exc": np.concatenate(field_chunks, axis=0),
        "dispersion": finalize_dispersion(modal_acc, kw_acc, grid["k"], node_count, L),
        "node_data": {n: {k: np.concatenate(v).astype(np.float32) for k, v in nd.items()}
                      for n, nd in node_data.items()},
        "energy_density": np.concatenate(density_chunks, axis=0),
        "energy_density_time": np.concatenate(density_t),
        "p_damp": np.concatenate(p_damp_chunks).astype(np.float32),
        "modal_ke": np.stack(modal_ke_chunks).astype(np.float32),
        "modal_ke_time": np.asarray(modal_ke_t),
    }
    for k, v in energy.items():
        data[k] = np.concatenate(v).astype(np.float32)

    meta = {
        "dt": dt, "oversamp": oversamp, "dl": dl, "m_node": m_node,
        "K_se": K_se, "K_bt": K_bt, "node_count": node_count,
        "tension": T, "length": L, "inspect_nodes": list(inspect_nodes),
    }
    return data, meta
