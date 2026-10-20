import os
import sys
import pickle
import gzip
import json
from pathlib import Path
from datetime import datetime
import smtplib
from email.message import EmailMessage
import concurrent.futures
import gc

import numpy as np

# Worker processes for extraction and plotting (32 suits the container; each extraction worker holds
# about 0.75 GB with 512-node chunks, so a 16 GB desktop needs a handful).
POST_WORKERS = int(os.environ.get("POST_WORKERS", 32))

# Import Plotting Modules
from plots.pos_fft import plot_node_pos_moment_fft
from plots.rotation import plot_angular_velocity, plot_axis_angle_over_time
from plots.energy import (plot_energies, plot_energy_components, plot_energy_totals,
                          plot_kinetic_trans_long, plot_kinetic_transfer, energy_components,
                          plot_energy_share, plot_modal_energy,
                          ENERGY_DENSITY_BLOCK, block_mean, block_centers)
from plots.energy_flow import (exact_flow, roundoff_forces, modal_transfer, edge_forces, edges_to_nodes,
                               dst_parseval,
                               plot_energy_flow, plot_transfer_power, plot_transfer_cumulative,
                               plot_element_map, plot_element_transfer_integral,
                               plot_modal_transfer, plot_modal_transfer_sum, plot_transfer_spectrum,
                               plot_energy_balance, plot_energy_unexplained, plot_roundoff_force,
                               N_TRANSFER_MODES)
from plots.spectrogram import plot_node_spectrogram
from plots.audio import generate_node_audio
from plots.inharmonicity import plot_inharmonicity_comparison, plot_inharmonicity_deviation
from plots.polarization import plot_polarization
from plots.decay import plot_partial_decay, plot_spectral_decay
from plots.transient import plot_spacetime_heatmap, plot_waterfall_3d, plot_node_transient
from plots.dispersion import (modal_projection, kspace_grid, kspace_projection, StreamingModalSpectra,
                              StreamingKOmegaSpectra, StreamingCrossSpectrum, finalize_dispersion,
                              find_resonances, plot_dispersion)
from utils import (reference_vectors, rest_orientations, rest_frames, reference_offsets,
                   quat_mul, node_displacement)
from plots.gallery import write_gallery

# Dispersion analysis: modes kept per frame and Welch segment length (frames).
DISPERSION_MODES = 256
DISPERSION_NPERSEG = 8192
# Finer time blocks (frames) for the transient phase, excitation chunks only.
FINE_MAP_BLOCK = 4
FINE_MODAL_BLOCK = 32

def _quad(v, K):
    """0.5 * v^T K v over the last axis of v (..., 3), for a diagonal (3,) or
    full (3, 3) K."""
    K = np.asarray(K, dtype=np.float64)
    if K.ndim == 2:
        return 0.5 * np.einsum('...i,ij,...j->...', v, K, v)
    return 0.5 * np.sum(K * v * v, axis=-1)


def _diag(K):
    K = np.asarray(K, dtype=np.float64)
    return K if K.ndim == 1 else np.diag(K)


# Parallel Extraction Worker
def _process_single_chunk(args):
    (nf, ef, inspect_nodes, m_node, inertia, K_se, K_bt, dl, capture_field, want_spectra,
     n_modes, length, dampening, reference, dt_sim) = args
    ref_vecs, q_rest = reference["ref"], reference["q_rest"]

    with gzip.open(nf, "rb") as f:
        n_chunk = pickle.load(f)
    with gzip.open(ef, "rb") as f:
        e_chunk = pickle.load(f)

    # Whole chunk as arrays: nodes (F, N, 5, 3) = [displacement (high part), velocity,
    # node[2], curvature, displacement low part] (4 rows for runs before the low part was
    # saved); edges: quaternions (F, N, 4) and vectors (F, N, 3, 3)
    # = [angular velocity, ., strain]. The last edge is the ghost edge.
    nodes = np.asarray(n_chunk, dtype=np.float32)
    quats = np.asarray([[e[0] for e in frame] for frame in e_chunk], dtype=np.float32)
    p_dev = None
    if reference["relative"]:
        # The GPU stores each orientation as its deviation p from the rest
        # orientation q0 (q = q0 * p); everything below works with q, except the
        # exact-flow forces, which use p as the shader does.
        p_dev = quats[:, :-1].copy()
        quats = quat_mul(reference["q_rest_ghost"], quats)
    evecs = np.asarray([[e[1] for e in frame] for frame in e_chunk], dtype=np.float32)
    del n_chunk, e_chunk
    n_frames, node_count = nodes.shape[:2]
    # Displacement as the shader holds it, hi + lo (float64); pos_hi / pos_lo for the
    # float32 rounding diagnostic, which repeats the shader's own arithmetic.
    pos = node_displacement(nodes)
    pos_hi = nodes[:, :, 0]
    pos_lo = nodes[:, :, 4] if nodes.shape[2] > 4 else None
    vel, curv = nodes[:, :, 1], nodes[:, :, 3]
    omega, strain = evecs[:, :-1, 0], evecs[:, :-1, 2]          # real edges 0..N-2

    # Inspected nodes (edge index clamped to the last edge, as before).
    out_node_data = {}
    for n in inspect_nodes:
        e_idx = min(n, quats.shape[1] - 1)
        out_node_data[n] = {
            'pos': pos[:, n].copy(), 'vel': vel[:, n].copy(), 'mom': nodes[:, n, 2].copy(),
            'quats': quats[:, e_idx].copy(), 'omega': evecs[:, e_idx, 0].copy(),
        }

    # Spatial transforms of the full field (only these leave the worker).
    spectra = None
    if want_spectra:
        # Physical node spacing is the tuned (stretched) length, not rest dl.
        L = length if length is not None else dl * (node_count - 1)
        grid = kspace_grid(node_count, L / (node_count - 1), k_max=n_modes * np.pi / L)
        spectra = (modal_projection(pos, n_modes), kspace_projection(pos, grid),
                   grid["k"], node_count, L)

    # Optional full-field transverse (Y,Z) displacement for the transient plots.
    field = pos[:, :, 1:3].astype(np.float32) if capture_field else None

    # Edge stretch exactly as the shader computes it: e = |r| / dl, where r is
    # the edge's reference vector (the pre-stretched spacing, exactly as handed
    # to the simulation, see utils.reference_vectors) plus the displacement
    # difference. dl is the rest length, so e ~ 1 + pre-strain at rest.
    r = ref_vecs + (pos[:, 1:] - pos[:, :-1])                 # (F, N-1, 3)
    stretch = np.linalg.norm(r.astype(np.float64), axis=-1) / dl
    eta_e = 1.0 - 0.3 * (stretch - 1.0)                        # per edge
    eta_n = 1.0 - 0.3 * (0.5 * (stretch[:, :-1] + stretch[:, 1:]) - 1.0)  # interior nodes

    # Axial parts: translational motion is in the world frame where the string
    # lies along x (index 0 is longitudinal), while the material-frame terms
    # (rotation, curvature, strain) put the axial direction at index 2
    # (inertia [I1,I1,I3], K_bt [EI,EI,GJ], K_se [.,.,EA]).
    v64 = vel.astype(np.float64)
    w_ends = np.ones(node_count)
    w_ends[[0, -1]] = 0.5
    ke_node = 0.5 * m_node * np.sum(v64 * v64, axis=-1) * w_ends            # (F, N)
    ke_node_l = 0.5 * m_node * v64[..., 0] ** 2 * w_ends

    K_bt_d, K_se_d, I_d = _diag(K_bt), _diag(K_se), _diag(inertia)
    k64 = curv[:, 1:-1].astype(np.float64)
    # The stretch scaling multiplies each quadratic form: E = eta^p * 0.5 v^T K v.
    bt_node = np.zeros((n_frames, node_count))
    bt_node[:, 1:-1] = eta_n ** 4 * _quad(k64, K_bt) * dl
    bt_node_tw = np.zeros((n_frames, node_count))
    bt_node_tw[:, 1:-1] = 0.5 * K_bt_d[2] * eta_n ** 4 * k64[..., 2] ** 2 * dl

    w64 = omega.astype(np.float64)
    s64 = strain.astype(np.float64)
    rot_edge = eta_e ** 2 * _quad(w64, inertia)
    rot_edge_t = 0.5 * I_d[2] * eta_e ** 2 * w64[..., 2] ** 2
    ss_edge = eta_e ** 2 * _quad(s64, K_se) * dl
    ss_edge_ex = 0.5 * K_se_d[2] * eta_e ** 2 * s64[..., 2] ** 2 * dl

    # Per-frame totals (axial parts of each family alongside; the transverse
    # remainder is derived at plot time).
    sums = [ke_node.sum(1), rot_edge.sum(1), bt_node.sum(1), ss_edge.sum(1),
            ke_node_l.sum(1), rot_edge_t.sum(1), bt_node_tw.sum(1), ss_edge_ex.sum(1)]

    # Exact energy flow from the shader's force law (see plots/energy_flow.py):
    # longitudinal energy, transfer power into it by channel, and the float32
    # rounding of the node force for the energy-source diagnostic.
    c_lin, c_ang = dampening
    q_e = quats[:, :-1]
    law = reference.get("force_law", "legacy")
    offset_form = None if p_dev is None else (p_dev, reference["frames"], reference["offsets"])
    fl = exact_flow(pos, vel, q_e, ref_vecs, q_rest, dl, m_node, K_se_d, c_lin, force_law=law,
                    omega=omega, inertia=I_d, K_bt=K_bt_d, offset_form=offset_form)
    d_old, d_new = roundoff_forces(pos_hi, q_e, ref_vecs, q_rest, reference["frames"],
                                   reference["offsets"], dl, K_se_d, pos_lo=pos_lo)
    # The formulation the run actually used is the one whose rounding matters.
    d_round, d_other = (d_new, d_old) if reference["relative"] else (d_old, d_new)
    d_round, d_other = d_round[:, 1:-1], d_other[:, 1:-1]
    d_perp2 = np.sum(d_round[..., 1:] ** 2, axis=-1)
    flow = {
        "KE_T": fl["ke_t"].sum(1), "KE_R": (rot_edge - rot_edge_t).sum(1), "KE_tor": rot_edge_t.sum(1),
        "E_L": fl["ke_l"].sum(1) + fl["u_l_quad"].sum(1) + fl["u_l_lin"].sum(1) + fl["u_l_bend"].sum(1),
        "P_disp": fl["p_disp"].sum(1), "P_rot": fl["p_rot"].sum(1), "D_L": fl["d_l"].sum(1),
        # Damping power of the whole state (linear on interior nodes, angular on edges).
        "D_all": c_lin * np.sum(v64[:, 1:-1] ** 2, axis=(1, 2))
                 + np.sum(np.asarray(c_ang) * w64 ** 2, axis=(1, 2)),
        "dF_x_rms": np.sqrt(np.mean(d_round[..., 0] ** 2, axis=1)),
        "dF_perp_rms": np.sqrt(np.mean(d_perp2, axis=1)),
        # The same for the other shader formulation (offset vs absolute orientation)
        "dF_x_rms_other": np.sqrt(np.mean(d_other[..., 0] ** 2, axis=1)),
        "dF_perp_rms_other": np.sqrt(np.mean(np.sum(d_other[..., 1:] ** 2, axis=-1), axis=1)),
        # Heating if the rounding were independent each solver step: dF^2 dt / (2m) per node.
        "H_x": np.sum(d_round[..., 0] ** 2, axis=1) * dt_sim / (2.0 * m_node),
        "H_perp": np.sum(d_perp2, axis=1) * dt_sim / (2.0 * m_node),
    }
    flow = {k: v.astype(np.float32) for k, v in flow.items()}
    del d_old, d_new, d_round, d_other, d_perp2

    # Force-port check against the saved internal force. The saved value was
    # computed at the mid-step state two solver steps before the saved one
    # (buffer swap), i.e. 1.5 steps behind the saved positions, so expect
    # agreement to the change over 1.5 steps, not to rounding. Saved every
    # step, the port matches that mid-step state to float32 rounding.
    f_port = edge_forces(r, q_e, dl, K_se_d, force_law=law, omega=omega, inertia=I_d)
    f_saved = evecs[:, :-1, 1].astype(np.float64)
    force_check = float(np.median(np.linalg.norm(f_saved - f_port, axis=-1)
                                  / np.maximum(np.linalg.norm(f_port, axis=-1), 1e-30)))
    del f_port, f_saved

    # Per-element maps, block-averaged in time; the excitation chunks also get
    # a finer version for the transient phase.
    per_elem = {"trans": fl["ke_t"], "long_ke": fl["ke_l"],
                "long": fl["ke_l"] + edges_to_nodes(fl["u_l_quad"]),
                "transfer": fl["p_disp"] + fl["p_rot"]}
    maps = {k: block_mean(v, ENERGY_DENSITY_BLOCK) for k, v in per_elem.items()}
    maps_fine = ({k: block_mean(v, FINE_MAP_BLOCK) for k, v in per_elem.items()}
                 if capture_field else None)

    # Transfer per longitudinal mode (exact sine-basis split) and the mode
    # coefficients for the streamed transfer spectrum.
    p_modes, t_coeffs = modal_transfer(v64[..., 0], fl["f_c"], N_TRANSFER_MODES)
    modal_tr = p_modes.mean(axis=0).astype(np.float32)
    modal_tr_fine = block_mean(p_modes, FINE_MODAL_BLOCK) if capture_field else None
    del fl, per_elem, p_modes

    # Kinetic energy per spatial mode (~half each mode's energy):
    # KE_n = 0.5 * (total mass) / 2 * adot_n^2, transverse (Y+Z) and longitudinal (X).
    adot = modal_projection(vel, n_modes).astype(np.float64)
    ke_modes = 0.25 * m_node * (node_count - 1) * adot ** 2                  # (F, n_modes, 3)
    ke_modes = np.stack([ke_modes[..., 1] + ke_modes[..., 2], ke_modes[..., 0]], axis=-1)
    modal_ke = ke_modes.mean(axis=0).astype(np.float32)                      # chunk average
    modal_ke_fine = block_mean(ke_modes, FINE_MODAL_BLOCK) if capture_field else None

    return {
        "n_frames": n_frames, "node_data": out_node_data,
        "energy": [np.asarray(a, dtype=np.float32) for a in sums],
        "field": field, "spectra": spectra, "flow": flow,
        "maps": maps, "maps_fine": maps_fine,
        "modal_ke": modal_ke, "modal_ke_fine": modal_ke_fine,
        "modal_transfer": modal_tr, "modal_transfer_fine": modal_tr_fine,
        "transfer_coeffs": t_coeffs if want_spectra else None, "force_check": force_check,
    }

def _format_parameters(metadata):
    """Render parameters.json as readable, sectioned plain text for the email body."""
    lines = []
    for key, val in metadata.items():
        if isinstance(val, dict):
            lines.append(f"\n[{key}]")
            for k, v in val.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"{key}: {val}")
    return "\n".join(lines)

ENERGY_KEYS = ("t_ke", "r_ke", "bt_pe", "ss_pe", "t_ke_l", "r_ke_t", "bt_pe_tw", "ss_pe_ex")

# Core Memory-Efficient Extraction
def rod_reference(node_count, length, dl, rod_params=None, uniform_reference=None,
                  relative_orientations=False):
    """Edge reference vectors and rest orientations exactly as the simulation
    got them (utils.generate_straight_rod). uniform_reference=False (runs made
    before the bit-identical spacing) reproduces the old float32 spacings.
    relative_orientations: the run stored orientations as deviations from the
    rest orientation (runs made with the offset formulation of the shader).

    Returns a dict: ref (N-1, 3), q_rest (N-1, 4) the rest orientation as the
    GPU effectively holds it (float32 absolute quaternions for older runs, the
    exact rest orientation for relative runs, where p = identity at rest),
    q_rest_ghost (N, 4) float64 incl. the ghost edge, frames (N-1, 3, 3),
    offsets (N-1, 3) and relative.
    """
    L = length if length is not None else dl * (node_count - 1)
    ref = reference_vectors(node_count, L, legacy=not uniform_reference)
    twists = (rod_params or {}).get("twists")
    if twists is None:
        print(f"[{datetime.now()}] Warning: rod twist unknown; assuming untwisted rest orientations.")
        twists = 0.0
    q0 = rest_orientations(node_count, twists)
    frames = rest_frames(node_count, twists)
    return {
        "ref": ref,
        "q_rest": q0 if relative_orientations else q0.astype(np.float32),
        "q_rest_ghost": np.concatenate([q0, q0[-1:]], axis=0),
        "frames": frames,
        "offsets": reference_offsets(ref, frames, dl),
        "relative": bool(relative_orientations),
    }


def extract_data(sim_path, config, m_node, dl, inertia, K_se, K_bt, inspect_nodes, length=None,
                 n_modes=DISPERSION_MODES, nperseg=DISPERSION_NPERSEG, rod_params=None,
                 uniform_reference=None, relative_orientations=False):
    """Read all chunks in parallel and reduce them to the arrays the plots need.

    Returns a dict with 'time', 'fs', the energy arrays (ENERGY_KEYS),
    'node_data', 'field_exc', 'excitation_cutoff_frame', 'dispersion',
    'flow' (per-frame motion energies and transfer power), the per-element
    'maps' / 'maps_fine' (+ '_time') and 'modal_ke' / 'modal_ke_fine' (+ '_time').
    """
    sim_path = Path(sim_path)
    node_files = sorted(sim_path.glob("n_*.pkl.gz"))
    edge_files = sorted(sim_path.glob("e_*.pkl.gz"))

    dt = config.get("dt", 1e-5)
    oversamp = config.get("oversampling_factor", 100)
    fs_native = 1.0 / (dt * oversamp)  # stored-frame rate == audio sample rate

    print(f"[{datetime.now()}] Dispatching data extraction for {len(node_files)} chunks across workers...")
    print(f"[{datetime.now()}] Target nodes for inspection: {inspect_nodes}")

    # 1. Fully Parallelized Extraction
    #    The first N chunks (the excitation window) also return the full spatial
    #    field for the transient plots.
    n_exc_chunks = min(10, len(node_files))
    if length is None:
        print(f"[{datetime.now()}] Warning: tuned length unknown; edge stretch (and the eta-scaled "
              "energies) will assume rest spacing dl, which ignores the string's pre-strain.")
    if "damping_coefficients" in config:
        dc = config["damping_coefficients"]
        dampening = (float(dc["linear"]), tuple(float(x) for x in dc["angular"]))
    else:   # runs before per-axis damping: one angular coefficient for all axes
        c_lin, c_ang = config.get("dampening", (0.0, 0.0))
        dampening = (float(c_lin), (float(c_ang),) * 3)
    node_count = config.get("node_count")
    if node_count is None:
        with gzip.open(node_files[0], "rb") as f:
            node_count = len(pickle.load(f)[0])
    reference = rod_reference(node_count, length, dl, rod_params, uniform_reference,
                              relative_orientations)
    # Runs before the variational force law carry no "force_law" entry.
    reference["force_law"] = config.get("force_law", "legacy")
    ref_vecs = reference["ref"]
    print(f"[{datetime.now()}] Edge reference spacing: "
          f"{'uniform' if uniform_reference else 'legacy float32 differences'} "
          f"({np.unique(ref_vecs[:, 0]).size} distinct values); orientations stored "
          f"{'relative to the rest frame' if relative_orientations else 'absolute'}.")
    args_list = [(nf, ef, inspect_nodes, m_node, inertia, K_se, K_bt, dl,
                  idx < n_exc_chunks, idx >= n_exc_chunks, n_modes, length, dampening,
                  reference, dt)
                 for idx, (nf, ef) in enumerate(zip(node_files, edge_files))]

    all_node_data = {n: {'pos': [], 'vel': [], 'mom': [], 'quats': [], 'omega': []} for n in inspect_nodes}
    all_energy = {key: [] for key in ENERGY_KEYS}
    all_flow = {}
    field_chunks = []
    maps, maps_t, maps_fine, maps_fine_t = {}, [], {}, []
    modal_ke_chunks, modal_ke_t, modal_fine, modal_fine_t = [], [], [], []
    modal_tr_chunks, modal_tr_fine, force_checks = [], [], []
    frame_dt = dt * oversamp
    # Modal spectra are accumulated as chunks arrive (in time order), so only
    # one Welch segment of modal data is ever held. Only free-vibration chunks
    # are fed, so the hammer's forced response doesn't smear the spectrum.
    modal_acc = StreamingModalSpectra(fs_native, nperseg=nperseg)
    kw_acc = StreamingKOmegaSpectra(fs_native, nperseg=nperseg)
    transfer_acc = StreamingCrossSpectrum(fs_native, nperseg=nperseg)
    spectra_meta = None

    excitation_cutoff_frame = 0
    frames_processed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=POST_WORKERS) as executor:
        for file_idx, res in enumerate(executor.map(_process_single_chunk, args_list)):
            n_frames = res["n_frames"]
            for n in inspect_nodes:
                for key in ('pos', 'vel', 'mom', 'quats', 'omega'):
                    all_node_data[n][key].append(res["node_data"][n][key])
            for key, arr in zip(ENERGY_KEYS, res["energy"]):
                all_energy[key].append(arr)
            for key, arr in res["flow"].items():
                all_flow.setdefault(key, []).append(arr)
            if res["field"] is not None:
                field_chunks.append(res["field"])
            if res["spectra"] is not None:
                c_modal, c_kspace, k_grid, n_nodes, grid_len = res["spectra"]
                modal_acc.feed(c_modal)
                kw_acc.feed(c_kspace)
                transfer_acc.feed(res["transfer_coeffs"])
                spectra_meta = (k_grid, n_nodes, grid_len)

            for key, arr in res["maps"].items():
                maps.setdefault(key, []).append(arr)
            maps_t.append(block_centers(frames_processed, n_frames, ENERGY_DENSITY_BLOCK, frame_dt))
            modal_ke_chunks.append(res["modal_ke"])
            modal_tr_chunks.append(res["modal_transfer"])
            force_checks.append(res["force_check"])
            modal_ke_t.append((frames_processed + 0.5 * (n_frames - 1)) * frame_dt)
            if res["maps_fine"] is not None:
                for key, arr in res["maps_fine"].items():
                    maps_fine.setdefault(key, []).append(arr)
                maps_fine_t.append(block_centers(frames_processed, n_frames, FINE_MAP_BLOCK, frame_dt))
                modal_fine.append(res["modal_ke_fine"])
                modal_tr_fine.append(res["modal_transfer_fine"])
                modal_fine_t.append(block_centers(frames_processed, n_frames, FINE_MODAL_BLOCK, frame_dt))
            del res

            frames_processed += n_frames
            if file_idx <= 9: # Files 0 through 9 represent the first 10 dispatches
                excitation_cutoff_frame += n_frames

    print(f"[{datetime.now()}] Extraction complete. Stitching arrays...")
    if force_checks:
        print(f"[{datetime.now()}] Force port vs saved internal force: median relative difference "
              f"{np.median(force_checks):.2e} (the saved force is 1.5 solver steps older than the saved positions).")
    if "dF_x_rms" in all_flow:
        med = {k: float(np.median(np.concatenate(all_flow[k])))
               for k in ("dF_x_rms", "dF_perp_rms", "dF_x_rms_other", "dF_perp_rms_other")}
        run, other = ("offset", "absolute") if relative_orientations else ("absolute", "offset")
        print(f"[{datetime.now()}] Float32 node-force rounding (median RMS over frames): "
              f"{run} formulation (this run) x {med['dF_x_rms']:.2e} N, y/z {med['dF_perp_rms']:.2e} N; "
              f"{other} formulation x {med['dF_x_rms_other']:.2e} N, y/z {med['dF_perp_rms_other']:.2e} N.")
    p_spec = transfer_acc.finalize()
    if p_spec is not None:
        p_spec["p_nf"] = p_spec["p_nf"] * dst_parseval(node_count - 2)
    data = {
        "time": np.arange(frames_processed) * (dt * oversamp),
        "fs": fs_native,
        "excitation_cutoff_frame": excitation_cutoff_frame,
        "field_exc": np.concatenate(field_chunks, axis=0) if field_chunks else None,
        "dispersion": finalize_dispersion(modal_acc, kw_acc, *spectra_meta) if spectra_meta else None,
        "flow": {k: np.concatenate(v) for k, v in all_flow.items()},
        "maps": {k: np.concatenate(v, axis=0) for k, v in maps.items()},
        "maps_time": np.concatenate(maps_t),
        "maps_fine": {k: np.concatenate(v, axis=0) for k, v in maps_fine.items()} if maps_fine else None,
        "maps_fine_time": np.concatenate(maps_fine_t) if maps_fine_t else None,
        "modal_ke": np.stack(modal_ke_chunks),          # (chunks, n_modes, 2)
        "modal_ke_time": np.asarray(modal_ke_t),
        "modal_ke_fine": np.concatenate(modal_fine, axis=0) if modal_fine else None,
        "modal_ke_fine_time": np.concatenate(modal_fine_t) if modal_fine_t else None,
        "modal_transfer": np.stack(modal_tr_chunks),    # (chunks, n_modes + 1)
        "modal_transfer_fine": np.concatenate(modal_tr_fine, axis=0) if modal_tr_fine else None,
        "transfer_spectrum": p_spec,
        "dt_sim": dt,
        "relative_orientations": bool(relative_orientations),
        "node_data": {
            n: {key: np.concatenate(vals) for key, vals in all_node_data[n].items()}
            for n in inspect_nodes
        },
    }
    for key in ENERGY_KEYS:
        data[key] = np.concatenate(all_energy[key])

    # Free up collection lists aggressively
    del all_node_data, all_energy, all_flow, field_chunks, modal_acc, kw_acc, transfer_acc
    del maps, maps_fine, modal_ke_chunks, modal_fine
    gc.collect()
    return data


def make_string_params(m_node, dl, K_se, K_bt, node_count=None, tension=None, length=None,
                       force_law="legacy", twists=None):
    """Continuum parameters for the reference overlays (T is None -> none).

    T_eff, EA_eff, EI_eff are what the shader applies at the rest state: with
    rest strain s0 = h/dl - 1 and eta0 = 1 - 0.3 s0, the axial force is
    f(e) = K_z eta^2 (e - 1), so the tension is f(e0) = K_z eta0^2 s0. For
    longitudinal waves m u'' = f'(e0)/dl (u_{j+1} - 2u_j + u_{j-1}) gives
    c^2 = f'(e0) h / (dl mu), i.e. an effective EA of
    K_z (eta0^2 - 0.6 eta0 s0)(1 + s0); bending is scaled by eta0^4.
    Variational force law: f(e) = K_z (eta^2 s + eta eta' s^2) plus, for a pre-twisted string,
    2 eta' eta^3 B0 from the rest twist energy (B0 = GJ kappa0^2), so T_eff and EA_eff include
    those terms (EA_eff from f'(e0): K_z (eta0^2 - 1.2 eta0 s0 + 0.09 s0^2) + 0.54 eta0^2 B0).
    """
    if length is None:
        length = dl * (node_count - 1)
    n_seg = node_count - 1 if node_count else length / dl
    K_z, EI = float(np.ravel(K_se)[2]), float(np.ravel(K_bt)[0])
    s0 = (length / n_seg) / dl - 1.0
    eta0 = 1.0 - 0.3 * s0
    if force_law == "variational":
        B0 = 0.0
        if twists and node_count:
            kappa0 = 2.0 * np.sin(np.pi * twists / (node_count - 1)) / dl
            B0 = float(np.ravel(K_bt)[2]) * kappa0 ** 2
        T_eff = K_z * (eta0 ** 2 * s0 - 0.3 * eta0 * s0 ** 2) - 0.6 * eta0 ** 3 * B0
        EA_eff = (K_z * (eta0 ** 2 - 1.2 * eta0 * s0 + 0.09 * s0 ** 2) + 0.54 * eta0 ** 2 * B0) * (1.0 + s0)
    else:
        T_eff = K_z * eta0 ** 2 * s0
        EA_eff = K_z * (eta0 ** 2 - 0.6 * eta0 * s0) * (1.0 + s0)
    return {
        "L": float(length),
        "mu": float(m_node * n_seg / length),
        "T": None if tension is None else float(tension),
        "EI": EI,
        "EA": K_z,
        "T_eff": T_eff,
        "EA_eff": EA_eff,
        "EI_eff": EI * eta0 ** 4,
    }


def strike_onset(time, kinetic, rel=1e-3):
    """Time of the first frame whose kinetic energy exceeds rel * its maximum
    (robust against the small noise while the string is still at rest)."""
    kinetic = np.asarray(kinetic, dtype=np.float64)
    if not kinetic.size or kinetic.max() <= 0:
        return None
    return float(time[int(np.argmax(kinetic > rel * kinetic.max()))])


def _rows_in(times, t0, t1):
    if times is None:
        return None
    rows = (times >= t0) & (times <= t1)
    return rows if rows.sum() > 1 else None


def build_plot_tasks(data, output_dir, run_id, dt, oversamp, dl, inspect_nodes, string_params):
    """Queue every plot as (func, args, kwargs) for run_plot_tasks."""
    output_dir = Path(output_dir)
    time = data["time"]
    fs_native = data["fs"]
    excitation_cutoff_frame = data["excitation_cutoff_frame"]
    stitched_node_data = data["node_data"]
    field_exc = data["field_exc"]
    flow = data.get("flow")
    string_len = string_params["L"]
    run_tag = f"run_{run_id:03d}"

    # Fundamental: measured (lowest resonance of the dispersion spectra) when
    # available, else ideal-string theory, else unknown.
    f1 = None
    disp = data.get("dispersion")
    if disp is not None:
        _, f_res = find_resonances(disp["freqs"], disp["psd_trans"])
        if f_res.size:
            f1 = float(f_res[0])
    if f1 is None and string_params.get("T") is not None:
        f1 = np.sqrt(string_params["T"] / string_params["mu"]) / (2.0 * string_params["L"])
    period = 1.0 / f1 if f1 else 0.01
    # KE-based transfer plots average over two periods (removes the KE<->PE
    # swing); the share/family/flow plots of the long phases over one.
    transfer_window = 2.0 * period

    # Transient: from just before the strike to three fundamental periods after.
    onset = strike_onset(time, data["t_ke"].astype(np.float64) + data["r_ke"])
    phases = {}
    if onset is not None:
        t0, t1 = onset - 1e-3, onset + 3.0 * period
        phases["Transient"] = (int(np.searchsorted(time, t0)), int(np.searchsorted(time, t1)))
    phases["Excitation"] = (0, excitation_cutoff_frame)
    phases["Free_Vibration"] = (excitation_cutoff_frame, None)
    phases["Full_Simulation"] = (0, None)
    transient_label = f"Transient (strike + 3 periods, {3 * period * 1e3:.0f} ms)"

    plot_tasks = []
    for phase_name, (start, end) in phases.items():
        p_time = time[start:end]
        if p_time.size < 3:
            print(f"[{datetime.now()}] Skipping empty phase '{phase_name}' (no frames in this window).")
            continue
        p_tke, p_rke, p_bpe, p_spe = (data[k][start:end] for k in ("t_ke", "r_ke", "bt_pe", "ss_pe"))
        p_tke_l, p_rke_t = data["t_ke_l"][start:end], data["r_ke_t"][start:end]
        p_bpe_tw, p_spe_ex = data["bt_pe_tw"][start:end], data["ss_pe_ex"][start:end]
        p_flow = {k: v[start:end] for k, v in flow.items()} if flow else None

        transient = phase_name == "Transient"
        long_phase = phase_name in ("Free_Vibration", "Full_Simulation")
        smooth = period if long_phase else None
        energy_prefix = str(output_dir / f"{run_tag}_{phase_name}")
        phase_title = transient_label if transient else phase_name.replace("_", " ")
        p_onset = onset if onset is not None and p_time[0] <= onset <= p_time[-1] else None

        # Energy overview plots.
        plot_tasks.append((plot_energy_totals, (p_time, p_tke, p_rke, p_bpe, p_spe),
                           {"title_prefix": phase_title, "filename": f"{energy_prefix}_energy_totals.png"}))
        plot_tasks.append((plot_energy_components,
                           (p_time, p_tke, p_rke, p_bpe, p_spe, p_tke_l, p_rke_t, p_bpe_tw, p_spe_ex),
                           {"window_s": smooth, "title_prefix": phase_title,
                            "filename": f"{energy_prefix}_energy_components.png"}))
        if p_flow is not None:
            plot_tasks.append((plot_energy_flow, (p_time, p_flow),
                               {"window_s": smooth, "onset": p_onset,
                                "title_prefix": phase_title, "filename": f"{energy_prefix}_energy_flow.png"}))
            plot_tasks.append((plot_transfer_power, (p_time, p_flow),
                               {"window_s": smooth, "onset": p_onset, "period": period,
                                "title_prefix": phase_title,
                                "filename": f"{energy_prefix}_transfer_power.png"}))
            plot_tasks.append((plot_transfer_cumulative, (p_time, p_flow),
                               {"onset": p_onset, "title_prefix": phase_title,
                                "filename": f"{energy_prefix}_transfer_cumulative.png"}))

        if not transient:
            for kw, suffix in (({}, "energy"), ({"normalized": True}, "energy_norm"),
                               ({"mode_transfer": True}, "energy_mode")):
                plot_tasks.append((plot_energies, (p_time, p_tke, p_rke, p_bpe, p_spe),
                                   {**kw, "window_s": smooth, "title_prefix": phase_title,
                                    "filename": f"{energy_prefix}_{suffix}.png"}))
            p_comps = energy_components(p_tke, p_rke, p_bpe, p_spe, p_tke_l, p_rke_t, p_bpe_tw, p_spe_ex)
            plot_tasks.append((plot_energy_share, (p_time, p_comps),
                               {"window_s": transfer_window, "title_prefix": phase_title,
                                "filename": f"{energy_prefix}_energy_share.png"}))
            plot_tasks.append((plot_kinetic_trans_long, (p_time, p_tke, p_tke_l),
                               {"window_s": transfer_window, "title_prefix": phase_title,
                                "filename": f"{energy_prefix}_ke_trans_long.png"}))
        if long_phase:
            plot_tasks.append((plot_kinetic_transfer, (p_time, p_tke, p_tke_l),
                               {"window_s": transfer_window, "title_prefix": phase_title,
                                "filename": f"{energy_prefix}_ke_transfer.png"}))

        # Per-element maps: where along the string energy sits and transfers.
        if transient or long_phase:
            src, src_t = ((data.get("maps_fine"), data.get("maps_fine_time")) if transient
                          else (data.get("maps"), data.get("maps_time")))
            rows = _rows_in(src_t, p_time[0], p_time[-1]) if src is not None else None
            if rows is not None:
                for kind, suffix in (("trans", "element_energy_trans"),
                                     ("long_ke", "element_energy_long_ke"),
                                     ("long", "element_energy_long"),
                                     ("transfer", "element_transfer")):
                    if kind not in src:
                        continue
                    plot_tasks.append((plot_element_map, (src_t[rows], src[kind][rows], string_len, kind),
                                       {"onset": p_onset, "title_prefix": phase_title,
                                        "filename": f"{energy_prefix}_{suffix}.png"}))
                plot_tasks.append((plot_element_transfer_integral,
                                   (src_t[rows], src["transfer"][rows], string_len),
                                   {"title_prefix": phase_title,
                                    "filename": f"{energy_prefix}_element_transfer_integral.png"}))

        # Is energy created? Free vibration only (no hammer work to account for).
        if phase_name == "Free_Vibration" and p_flow is not None and "D_all" in p_flow:
            plot_tasks.append((plot_energy_balance,
                               (p_time, p_tke.astype(np.float64) + p_rke + p_bpe + p_spe, p_flow),
                               {"title_prefix": phase_title, "filename": f"{energy_prefix}_energy_balance.png"}))
            plot_tasks.append((plot_energy_unexplained, (p_time, p_flow),
                               {"title_prefix": phase_title,
                                "filename": f"{energy_prefix}_energy_unexplained.png"}))
            plot_tasks.append((plot_roundoff_force, (p_time, p_flow),
                               {"relative": data.get("relative_orientations", False),
                                "title_prefix": phase_title,
                                "filename": f"{energy_prefix}_roundoff_force.png"}))

        # Transfer into each longitudinal mode.
        if transient or phase_name == "Full_Simulation":
            src, src_t = ((data.get("modal_transfer_fine"), data.get("modal_ke_fine_time"))
                          if transient else (data.get("modal_transfer"), data.get("modal_ke_time")))
            rows = _rows_in(src_t, p_time[0], p_time[-1]) if src is not None else None
            if rows is not None:
                plot_tasks.append((plot_modal_transfer, (src[rows], src_t[rows]),
                                   {"onset": p_onset, "title_prefix": phase_title,
                                    "filename": f"{energy_prefix}_modal_transfer.png"}))
                plot_tasks.append((plot_modal_transfer_sum, (src[rows], src_t[rows]),
                                   {"onset": p_onset, "title_prefix": phase_title,
                                    "filename": f"{energy_prefix}_modal_transfer_sum.png"}))

        # Whole-string displacement and modal energy during the transient.
        if transient:
            if field_exc is not None:
                stop = min(end, field_exc.shape[0])
                if stop - start > 2:
                    spacing = string_len / (field_exc.shape[1] - 1)
                    args = (time[start:stop], field_exc[start:stop], spacing)
                    plot_tasks.append((plot_spacetime_heatmap, args,
                                       {"title_prefix": phase_title, "filename": f"{energy_prefix}_heatmap.png"}))
                    plot_tasks.append((plot_waterfall_3d, args,
                                       {"title_prefix": phase_title, "filename": f"{energy_prefix}_waterfall.png"}))
            rows = _rows_in(data.get("modal_ke_fine_time"), p_time[0], p_time[-1])
            if rows is not None:
                plot_tasks.append((plot_modal_energy,
                                   (data["modal_ke_fine"][rows], data["modal_ke_fine_time"][rows]),
                                   {"title_prefix": phase_title,
                                    "filename": f"{energy_prefix}_modal_energy.png"}))

        # Node-specific plots.
        for n in inspect_nodes:
            nd = stitched_node_data[n]
            p_pos, p_vel, p_mom = nd['pos'][start:end], nd['vel'][start:end], nd['mom'][start:end]
            p_quats, p_omega = nd['quats'][start:end], nd['omega'][start:end]
            node_prefix = str(output_dir / f"{run_tag}_node_{n:03d}_{phase_name}")
            node_title = f"Node {n} | {phase_title}"

            if transient:
                plot_tasks.append((plot_node_transient, (p_time, p_pos, p_vel),
                                   {"onset": p_onset, "title_prefix": node_title,
                                    "filename": f"{node_prefix}_motion.png"}))
            else:
                plot_tasks.append((plot_node_pos_moment_fft, (p_time, p_pos, p_mom, dt, oversamp),
                                   {"moments": True, "title_prefix": node_title,
                                    "filename_prefix": f"{node_prefix}_fft"}))
            plot_tasks.append((plot_axis_angle_over_time, (p_time, p_quats),
                               {"title_prefix": node_title, "filename": f"{node_prefix}_angles.png"}))
            plot_tasks.append((plot_angular_velocity, (p_time, p_omega),
                               {"title_prefix": node_title, "filename": f"{node_prefix}_omega.png",
                                "fs": fs_native}))

            # Full-run-only analyses (audio + spectral/decay/polarization).
            if phase_name == "Full_Simulation":
                plot_tasks.append((generate_node_audio, (p_pos, p_vel, p_mom, fs_native),
                                   {"title_prefix": node_title, "filename_prefix": f"{node_prefix}_audio"}))
                plot_tasks.append((plot_inharmonicity_comparison, (p_time, p_pos, dt, oversamp),
                                   {"title_prefix": node_title, "filename_prefix": node_prefix}))
                plot_tasks.append((plot_inharmonicity_deviation, (p_time, p_pos, dt, oversamp),
                                   {"title_prefix": node_title, "filename": f"{node_prefix}_inharm_deviation.png"}))
                plot_tasks.append((plot_polarization, (p_time, p_pos),
                                   {"title_prefix": node_title, "filename": f"{node_prefix}_polarization.png"}))
                plot_tasks.append((plot_partial_decay, (p_time, p_pos, dt, oversamp),
                                   {"title_prefix": node_title, "filename": f"{node_prefix}_partial_decay.png"}))
                plot_tasks.append((plot_spectral_decay, (p_time, p_pos, dt, oversamp),
                                   {"title_prefix": node_title, "filename": f"{node_prefix}_spectral_decay.png"}))
                plot_tasks.append((plot_node_spectrogram, (p_pos, fs_native),
                                   {"title_prefix": node_title, "filename": f"{node_prefix}_spectrogram.png"}))

    # Whole-string displacement over the full excitation window
    if field_exc is not None and field_exc.shape[0] > 2:
        exc_time = time[:field_exc.shape[0]]
        spacing = string_len / (field_exc.shape[1] - 1)
        trans_prefix = str(output_dir / f"{run_tag}_Excitation")
        plot_tasks.append((plot_spacetime_heatmap, (exc_time, field_exc, spacing),
                           {"title_prefix": "Excitation", "filename": f"{trans_prefix}_heatmap.png"}))
        plot_tasks.append((plot_waterfall_3d, (exc_time, field_exc, spacing),
                           {"title_prefix": "Excitation", "filename": f"{trans_prefix}_waterfall.png"}))

    # Energy per spatial mode over the whole run
    if data.get("modal_ke") is not None and len(data["modal_ke"]) > 1:
        plot_tasks.append((plot_modal_energy, (data["modal_ke"], data["modal_ke_time"]),
                           {"title_prefix": "Full Simulation",
                            "filename": str(output_dir / f"{run_tag}_Full_Simulation_modal_energy.png")}))

    # Transfer spectrum (free vibration), marked with measured resonances.
    if data.get("transfer_spectrum") is not None:
        f_trans = find_resonances(disp["freqs"], disp["psd_trans"])[1] if disp is not None else ()
        f_long = find_resonances(disp["freqs"], disp["psd_long"])[1] if disp is not None else ()
        plot_tasks.append((plot_transfer_spectrum, (data["transfer_spectrum"], f_trans, f_long),
                           {"title_prefix": "Free Vibration",
                            "filename": str(output_dir / f"{run_tag}_Free_Vibration_transfer_spectrum.png")}))

    # Dispersion relation from the streamed modal spectra (free vibration),
    # in physical units and normalised (mode number, v / c0).
    if disp is not None:
        for normalized, suffix in ((False, "dispersion"), (True, "dispersion_normalized")):
            plot_tasks.append((plot_dispersion, (disp, string_params),
                               {"normalized": normalized, "f1": f1, "title_prefix": "Free Vibration",
                                "filename": str(output_dir / f"{run_tag}_{suffix}.png")}))

    return plot_tasks


def run_plot_tasks(plot_tasks, max_workers=None):
    max_workers = max_workers or POST_WORKERS
    """Execute plotting tasks in parallel; returns the generated file paths."""
    print(f"[{datetime.now()}] Dispatching {len(plot_tasks)} plotting tasks to concurrent workers...")
    generated_images = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, *args, **kwargs) for func, args, kwargs in plot_tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                generated_images.extend(future.result())
            except Exception as e:
                print(f"[{datetime.now()}] A plot worker encountered an error: {e}")
                import traceback
                traceback.print_exc()
    return generated_images


def send_email(email_config, sim_path, run_id, inspect_nodes, generated_images, output_dir):
    # Email Dispatch (optional - skipped when email is not configured, e.g. local CLI runs)
    email_ready = bool(email_config and email_config.get('sender_email')
                       and email_config.get('sender_password') and email_config.get('receiver_email'))
    if not email_ready:
        print(f"[{datetime.now()}] Plotting complete. Generated {len(generated_images)} files in {output_dir}. "
              f"Email not configured - skipping dispatch.")
        return

    print(f"[{datetime.now()}] Plotting complete. Attached {len(generated_images)} files. Dispatching email...")
    msg = EmailMessage()
    msg['Subject'] = f"Piano Sim Run {run_id:03d} Completed - Plot Suite"
    msg['From'] = email_config['sender_email']
    msg['To'] = email_config['receiver_email']
    body_lines = [
        f"Simulation {run_id:03d} plots attached.",
        f"Memory-efficient parallel generation successful for {len(inspect_nodes)} node(s): {inspect_nodes}.",
        f"Attached {len(generated_images)} artefacts.",
        "",
    ]
    param_file = Path(sim_path) / "parameters.json"
    if param_file.exists():
        try:
            with open(param_file) as pf:
                meta = json.load(pf)
            body_lines.append("=== Simulation Parameters ===")
            body_lines.append(_format_parameters(meta))
        except Exception as e:
            body_lines.append(f"(could not read parameters.json: {e})")
    msg.set_content("\n".join(body_lines))

    for img_path in generated_images:
        try:
            with open(img_path, 'rb') as f:
                img_data = f.read()
                # Attach with the right MIME type per artefact
                path_str = str(img_path)
                if path_str.endswith('.json'):
                    msg.add_attachment(img_data, maintype='application', subtype='json', filename=Path(img_path).name)
                elif path_str.endswith('.wav'):
                    msg.add_attachment(img_data, maintype='audio', subtype='wav', filename=Path(img_path).name)
                else:
                    msg.add_attachment(img_data, maintype='image', subtype='png', filename=Path(img_path).name)
        except Exception as e:
            print(f"Could not attach {img_path}: {e}")

    try:
        with smtplib.SMTP_SSL(email_config['smtp_server'], email_config['smtp_port']) as smtp:
            smtp.login(email_config['sender_email'], email_config['sender_password'])
            smtp.send_message(msg)
        print(f"[{datetime.now()}] Email sent successfully.")
    except Exception as e:
        print(f"[{datetime.now()}] Failed to send email: {e}")


def bow_post_process(sim_path, output_dir):
    """Bowed runs: regime, playing frequency, contact temperature and the bow figures
    (plots/bowing.py) from the bow output b_*.pkl.gz and the bridge-end edge force.
    Writes bow_summary.json to output_dir and returns the figure paths."""
    from plots import bowing as bp
    import bowing
    sim_path, output_dir = Path(sim_path), Path(output_dir)
    files = sorted(sim_path.glob("b_*.pkl.gz"))
    if not files:
        return []
    P = json.load(open(sim_path / "parameters.json"))
    cfg, bow_params, bow_derived = P["config"], P["bow_params"], P["bow_derived"]
    frames = []
    bridge = []
    for bf in files:
        with gzip.open(bf, "rb") as f:
            frames.extend(pickle.load(f))
        compact = sim_path / bf.name.replace("b_", "s_").replace(".pkl.gz", ".npz")
        if compact.exists():          # runs with save_state = False
            bridge.extend(np.load(compact)["bridge_force"][:, 1])
            continue
        with gzip.open(sim_path / bf.name.replace("b_", "e_"), "rb") as f:
            e_chunk = pickle.load(f)
        # transverse force in the first edge (lab y, the bowing direction): the bridge force
        bridge.extend(fr[0][1][1][1] for fr in e_chunk)
        del e_chunk
    B = bp.bow_arrays(frames)
    bridge = np.asarray(bridge, np.float64)
    fdt = cfg["dt"] * cfg["oversampling_factor"]
    t = (np.arange(len(frames)) + 1) * fdt
    stuck = np.all(B["sticking"] > 0.5, axis=1)
    mid = B["slip_velocity"].shape[1] // 2
    v_rel = B["slip_velocity"][:, mid]
    v_surface = B["velocity"] + B["slip_velocity"].mean(axis=1)      # mean over the contact nodes
    # playing frequency: bridge-force fundamental over the second half (zero-padded peak)
    half = t > t[-1] / 2
    x = bridge[half] - bridge[half].mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), 16 * len(x)))
    fr = np.fft.rfftfreq(16 * len(x), fdt)
    rod = P["rod_derived"]
    f_free = 1.0 / (2.0 * P["rod_params"]["tuned_length"]) * np.sqrt(
        P["rod_params"]["tension_force"] * P["rod_params"]["tuned_length"] / (rod["m_node"] * (cfg["node_count"] - 1)))
    band = (fr > 0.7 * f_free) & (fr < 1.3 * f_free)
    f_play = float(fr[band][np.argmax(spec[band])])
    period = 1.0 / f_play
    summary = {"playing_frequency": f_play, "flexible_string_f1": f_free,
               **bp.regime(t, v_surface, B["velocity"], stuck, period)}
    summary["stuck_fraction_per_node"] = np.mean(B["sticking"][t >= t[-1] - 21 * period] > 0.5, axis=0).tolist()
    N_c = B["normal_force"] * np.asarray(bow_derived["contact_weights"])[mid]
    on = N_c > 0
    mu_eff = np.where(on, np.abs(B["force"][:, mid]) / np.where(on, N_c, 1.0), np.nan)
    last = t > t[-1] - 10 * period
    summary["mu_eff_max_last_10_periods"] = float(np.nanmax(mu_eff[last]))
    thermal = bow_params.get("friction") == "thermal"
    if thermal:
        T = B["temperature"][:, mid]
        summary["temperature_range_last_10_periods"] = [float(T[last].min()), float(T[last].max())]
    with open(output_dir / "bow_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{datetime.now()}] Bow: {summary['regime']}, playing frequency {f_play:.2f} Hz, stuck fraction {summary['stuck_fraction']:.3f}"
        + (f", contact temperature {summary['temperature_range_last_10_periods'][0]:.1f}-"
           f"{summary['temperature_range_last_10_periods'][1]:.1f} K" if thermal else ""))
    w = t > t[-1] - 3 * period
    out = [bp.plot_time_series(t[w], v_surface[w], "String velocity at the bow", "Velocity (m/s)",
                               output_dir / "bow_string_velocity.png", reference=bow_params["velocity"],
                               reference_label="bow"),
           bp.plot_time_series(t[w], bridge[w], "Bridge force", "Force (N)", output_dir / "bow_bridge_force.png"),
           bp.plot_regime(t, v_surface, B["velocity"], stuck, period, "Flybacks per period", output_dir / "bow_regime.png")]
    if thermal:
        out.append(bp.plot_time_series(t[w], B["temperature"][w, mid], "Contact temperature",
                                       "Temperature above ambient (K)", output_dir / "bow_temperature.png"))
        vs = np.linspace(0.0, max(0.05, float(np.nanmax(np.abs(v_rel[last])))), 400)
        curve = (np.concatenate([-vs[::-1], vs]), bowing.friction_curve(np.concatenate([-vs[::-1], vs])))
        label = "steady sliding"
    else:
        vs = np.linspace(0.0, max(0.05, float(np.nanmax(np.abs(v_rel[last])))), 400)
        curve = (np.concatenate([-vs[::-1], vs]), bowing.friction_curve(np.concatenate([-vs[::-1], vs])))
        label = "friction curve"
    out.append(bp.plot_friction_plane(v_rel[last], mu_eff[last], "Friction coefficient", output_dir / "bow_friction.png",
                                      curve=curve, curve_label=label))
    return [str(o) for o in out]


def pluck_post_process(sim_path, output_dir):
    """Plucked runs: release time and displacement, contact forces, works, twist at the pick node,
    and the pluck figures (plots/plucking.py) from the pick records p_*.pkl.gz, the pick node's
    displacement and the bridge-end edge force. Writes pluck_summary.json to output_dir and returns
    the figure paths."""
    from plots import plucking as pp
    import plucking
    sim_path, output_dir = Path(sim_path), Path(output_dir)
    P = json.load(open(sim_path / "parameters.json"))
    if not P.get("pick_params"):
        return []
    files = sorted(sim_path.glob("p_*.pkl.gz"))
    if not files:
        return []
    cfg, derived = P["config"], P["pick_derived"]
    j = int(derived["pick_node"])
    frames, bridge, disp, twist = [], [], [], []
    for pf in files:
        with gzip.open(pf, "rb") as f:
            frames.extend(pickle.load(f))
        compact = sim_path / pf.name.replace("p_", "s_").replace(".pkl.gz", ".npz")
        if compact.exists():          # runs with save_state = False
            c = np.load(compact)
            bridge.extend(c["bridge_force"][:, 1])
            disp.extend(c["contact_displacement"][:, 0, :])
            twist.extend(np.full(len(c["bridge_force"]), np.nan))
            continue
        with gzip.open(sim_path / pf.name.replace("p_", "e_"), "rb") as f:
            e_chunk = pickle.load(f)
        bridge.extend(fr[0][1][1][1] for fr in e_chunk)
        # twist of edge j: the axial (material 3) component of the rotation vector of its
        # deviation p from the rest orientation
        for fr in e_chunk:
            q = np.asarray(fr[j][0], np.float64)
            s = np.linalg.norm(q[:3])
            twist.append(0.0 if s == 0 else 2.0 * np.arctan2(s, abs(q[3])) * np.sign(q[3]) * q[2] / s)
        del e_chunk
        with gzip.open(sim_path / pf.name.replace("p_", "n_"), "rb") as f:
            n_chunk = pickle.load(f)
        disp.extend(node_displacement(np.asarray([fr[j] for fr in n_chunk], np.float32)))
        del n_chunk
    A = plucking.pick_arrays(frames)
    bridge, disp, twist = np.asarray(bridge, np.float64), np.asarray(disp, np.float64), np.asarray(twist)
    dt = cfg["dt"]
    fdt = dt * cfg["oversampling_factor"]
    t = (np.arange(len(frames)) + 1) * fdt
    on = A["in_contact"]
    summary = {"contact": bool(on.any())}
    if on.any():
        # contact intervals (frames): the rigid pick touches and leaves the string a few times while
        # it accelerates, then pushes it in one long contact that ends in the release; a slow pick can
        # be struck again by the returning string. The release is the end of the contact interval
        # with the largest normal impulse; its exact solver step is the last contact step recorded at
        # the end of the following frame.
        edges_ = np.diff(np.r_[0, on.astype(int), 0])
        starts, ends = np.nonzero(edges_ == 1)[0], np.nonzero(edges_ == -1)[0] - 1
        impulse = [A["normal_force"][a:b + 1].sum() for a, b in zip(starts, ends)]
        main = int(np.argmax(impulse))
        last = int(ends[main])
        release_step = int(A["last_contact_step"][min(last + 1, len(t) - 1)])
        summary.update({
            "first_contact_time": float(A["first_contact_step"][-1] * dt),
            "release_time": float(release_step * dt),
            "release_frame": last,
            "main_contact_start_time": float(starts[main] * fdt),
            "contact_intervals_before_release": main,
            "contacts_after_release": [[float(a * fdt), float((b + 1) * fdt)] for a, b in zip(starts[main + 1:], ends[main + 1:])],
            "release_displacement_y": float(disp[last, 1]),
            "release_displacement_z": float(disp[last, 2]),
            "max_normal_force": float(A["normal_force"][on].max()),
            "stuck_fraction_in_contact": float(A["sticking"][on].mean()),
            "max_penetration": float(A["max_penetration"][-1]),
            "pick_work": float(A["body_work"].sum()),
            "friction_heat": float(-A["friction_work"].sum()),
            "normal_work": float(A["normal_work"].sum()),
            "energy_to_string": float((A["body_work"] + A["friction_work"] + A["normal_work"]).sum()),
            "twist_at_release": float(twist[last]),
            "max_twist": float(np.nanmax(np.abs(twist))) if np.isfinite(twist).any() else None,
        })
    else:
        last, summary["release_time"] = 0, 0.0
    # fundamental of the bridge force after the release (zero-padded peak near the flexible-string f1)
    rod = P["rod_derived"]
    tension = rod.get("tension_force", P["rod_params"]["tension_force"])
    L = P["rod_params"]["tuned_length"]
    f_free = np.sqrt(tension * L / (rod["m_node"] * (cfg["node_count"] - 1))) / (2.0 * L)
    after = np.arange(len(t)) > last
    if after.sum() > 64:
        x = bridge[after] - bridge[after].mean()
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), 16 * len(x)))
        fr = np.fft.rfftfreq(16 * len(x), fdt)
        band = (fr > 0.7 * f_free) & (fr < 1.3 * f_free)
        summary["fundamental"] = float(fr[band][np.argmax(spec[band])])
    summary["flexible_string_f1"] = float(f_free)
    with open(output_dir / "pluck_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if on.any():
        print(f"[{datetime.now()}] Pluck: release at {summary['release_time']*1e3:.3f} ms, displacement "
              f"({summary['release_displacement_y']*1e3:.3f}, {summary['release_displacement_z']*1e3:.3f}) mm, "
              f"energy to the string {summary['energy_to_string']*1e3:.4f} mJ")
    t_rel = summary["release_time"]
    w = t <= max(3.0 * t_rel, t_rel + 20e-3)
    c = t <= t_rel + 2.0 * fdt
    out = [pp.plot_displacement(t[w], disp[w, 1], disp[w, 2], t_rel, "Pick-node displacement",
                                output_dir / "pluck_displacement.png"),
           pp.plot_time_series(t, bridge, "Bridge force", "Force (N)", output_dir / "pluck_bridge_force.png")]
    if on.any():
        out.append(pp.plot_contact_forces(t[c], A["normal_force"][c], A["friction_force"][c], t_rel,
                                          "Pick contact forces", output_dir / "pluck_contact_forces.png"))
        orbit = (t > t_rel) & (t <= t_rel + 20.0 / f_free)
        out.append(pp.plot_orbit(disp[orbit, 1], disp[orbit, 2], "Pick-node orbit after release (20 periods)",
                                 output_dir / "pluck_orbit.png"))
    if np.isfinite(twist).any():
        out.append(pp.plot_time_series(t[w], twist[w] * 1e3, "Twist at the pick node", "Twist (mrad)",
                                       output_dir / "pluck_twist.png"))
    return [str(o) for o in out]


def hammer_contact_summary(sim_path, output_dir):
    """Hammer runs with a curved crown or felt friction: contact angle, stick fraction, friction heat
    and the twist impulse from the hammer records (p_*.pkl.gz). Writes hammer_contact_summary.json."""
    sim_path, output_dir = Path(sim_path), Path(output_dir)
    P = json.load(open(sim_path / "parameters.json"))
    if P.get("pick_params"):
        return []
    files = sorted(sim_path.glob("p_*.pkl.gz"))
    if not files:
        return []
    hammer = []
    for pf in files:
        with gzip.open(pf, "rb") as f:
            hammer.extend(fr[1] for fr in pickle.load(f))
    summary = {"hammer_crown_radius": P["hammer_params"].get("hammer_crown_radius", 0.0),
               "hammer_felt_friction": P["hammer_params"].get("hammer_felt_friction", 0.0),
               "hammer_offset_y": P["hammer_params"].get("hammer_offset_y", 0.0)}
    if hammer[0][3]:
        C = np.array([[c[2] for c in h[3]] for h in hammer], np.float64)
        ic = np.array([[c[0] for c in h[3]] for h in hammer], bool)
        st = np.array([[c[1] for c in h[3]] for h in hammer], bool)
        R = P["rod_params"]["core_radius"] + 2.0 * P["rod_params"]["winding_radius"]
        psi = np.degrees(np.arctan2(C[..., 3], C[..., 4]))[ic]
        summary.update({
            "contact_node_frames": int(ic.sum()),
            "stuck_fraction": float(st[ic].mean()) if ic.any() else None,
            "contact_angle_deg": [float(psi.min()), float(psi.max())] if ic.any() else None,
            "max_friction_ratio": float(np.max(np.abs(C[..., 2][ic]) / C[..., 1][ic])) if ic.any() else None,
            "friction_heat": float(-C[..., 6].sum()),
            "work_at_felt_surface": float(C[..., 5].sum()),
            "friction_work_on_hammer": float(sum(h[2] for h in hammer)),
            # couple -R F_t e_x on each contact edge
            "twist_impulse": float(-R * C[..., 8].sum()),
        })
    with open(output_dir / "hammer_contact_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return []


def process_and_plot(sim_path, run_id, config, email_config, m_node, dl, inertia, K_se, K_bt, inspect_nodes,
                     tension=None, length=None, rod_params=None, uniform_reference=None,
                     relative_orientations=False):
    sim_path = Path(sim_path)
    if config.get("post_process") in ("bow_only", "pluck_only"):
        output_dir = sim_path / "plots"
        output_dir.mkdir(exist_ok=True)
        generated_images = bow_post_process(sim_path, output_dir) + pluck_post_process(sim_path, output_dir)
        send_email(email_config, sim_path, run_id, inspect_nodes, generated_images, output_dir)
        return
    data = extract_data(sim_path, config, m_node, dl, inertia, K_se, K_bt, inspect_nodes, length=length,
                        rod_params=rod_params, uniform_reference=uniform_reference,
                        relative_orientations=relative_orientations)

    output_dir = sim_path / "plots"
    output_dir.mkdir(exist_ok=True)
    string_params = make_string_params(m_node, dl, K_se, K_bt, node_count=config.get("node_count"),
                                       tension=tension, length=length,
                                       force_law=config.get("force_law", "legacy"),
                                       twists=(rod_params or {}).get("twists"))

    plot_tasks = build_plot_tasks(data, output_dir, run_id, config.get("dt", 1e-5),
                                  config.get("oversampling_factor", 100), dl, inspect_nodes, string_params)
    del data
    gc.collect()
    generated_images = run_plot_tasks(plot_tasks)
    generated_images = (list(generated_images) + bow_post_process(sim_path, output_dir)
                        + pluck_post_process(sim_path, output_dir) + hammer_contact_summary(sim_path, output_dir))
    send_email(email_config, sim_path, run_id, inspect_nodes, generated_images, output_dir)


def preview_plots(out_dir, max_workers=8):
    """Run the full plot suite on synthetic data (no chunk files, no email)."""
    from preview_data import make_preview_data

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now()}] Generating synthetic preview data...")
    data, meta = make_preview_data()
    string_params = make_string_params(meta["m_node"], meta["dl"], meta["K_se"], meta["K_bt"],
                                       node_count=meta["node_count"], tension=meta["tension"],
                                       length=meta["length"])
    plot_tasks = build_plot_tasks(data, out_dir, 0, meta["dt"], meta["oversamp"], meta["dl"],
                                  meta["inspect_nodes"], string_params)
    generated = run_plot_tasks(plot_tasks, max_workers=max_workers)
    index = write_gallery(out_dir, title="Plot Preview (synthetic data)")
    print(f"[{datetime.now()}] Preview complete. Generated {len(generated)} files in {out_dir}.")
    print(f"[{datetime.now()}] Open {index.as_uri()} in a browser to view them.")

# ---------------------------------------------------------
# CLI Execution Wrapper
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("sim_path", type=str, nargs="?")
    parser.add_argument("--test", action="store_true",
                        help="Render all plots from synthetic data (no extraction, no email).")
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "plot_preview"),
                        help="Output directory for --test.")
    args = parser.parse_args()

    if args.test:
        preview_plots(args.out)
        sys.exit(0)
    if args.sim_path is None:
        parser.error("sim_path is required unless --test is given")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print(f"[{datetime.now()}] Warning: 'python-dotenv' is not installed. Falling back to system environment variables.")

    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = os.environ.get("SMTP_PORT")

    if not sender_email or not sender_password or not receiver_email:
        print(f"[{datetime.now()}] Note: email env vars not fully set (SENDER_EMAIL, SENDER_PASSWORD, "
              "RECEIVER_EMAIL). Plots and audio will be generated locally; email dispatch will be skipped.")

    target_dir = Path(args.sim_path).resolve()

    param_file = target_dir / "parameters.json"
    with open(param_file, "r") as f:
        metadata = json.load(f)

    # Config setup
    sim_config = metadata.get("config", {})
    inspect_nodes = sim_config.get("inspect_nodes", [64])
    if not inspect_nodes:
        inspect_nodes = [64]  # Safety fallback if list is empty

    rod = metadata["rod_derived"]
    rod_params = metadata.get("rod_params", {})
    email_config_dict = {
        "sender_email": sender_email,
        "sender_password": sender_password,
        "receiver_email": receiver_email,
        "smtp_server": smtp_server,
        "smtp_port": smtp_port,
    }

    process_and_plot(
        sim_path=target_dir,
        run_id=metadata.get("run_id", 0),
        config=sim_config,
        email_config=email_config_dict,
        m_node=rod["m_node"], dl=rod["dl"],
        inertia=np.array(rod["inertia"]), K_se=np.array(rod["K_se"]), K_bt=np.array(rod["K_bt"]),
        inspect_nodes=inspect_nodes,
        tension=rod.get("tension_force", rod_params.get("tension_force")), length=rod_params.get("tuned_length"),
        rod_params=rod_params, uniform_reference=bool(rod.get("uniform_reference_spacing", False)),
        relative_orientations=rod.get("orientation_representation") == "relative_to_rest",
    )
