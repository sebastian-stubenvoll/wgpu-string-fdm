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

# Import Plotting Modules
from plots.pos_fft import plot_node_pos_moment_fft
from plots.rotation import plot_angular_velocity, plot_axis_angle_over_time
from plots.energy import (plot_energies, plot_energy_envelope, plot_energy_components,
                          plot_energy_totals, plot_kinetic_trans_long, plot_kinetic_transfer,
                          energy_components, plot_energy_density, plot_energy_balance,
                          plot_motion_power, plot_energy_share, plot_modal_energy,
                          ENERGY_DENSITY_BLOCK, block_mean, block_centers)
from plots.spectrogram import plot_node_spectrogram
from plots.audio import generate_node_audio
from plots.inharmonicity import plot_inharmonicity_comparison, plot_inharmonicity_deviation
from plots.polarization import plot_polarization
from plots.decay import plot_partial_decay, plot_spectral_decay
from plots.transient import plot_spacetime_heatmap, plot_waterfall_3d
from plots.dispersion import (modal_projection, kspace_grid, kspace_projection, StreamingModalSpectra,
                              StreamingKOmegaSpectra, finalize_dispersion, find_resonances,
                              plot_dispersion)
from plots.gallery import write_gallery

# Dispersion analysis: modes kept per frame and Welch segment length (frames).
DISPERSION_MODES = 256
DISPERSION_NPERSEG = 8192

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
     n_modes, length, dampening) = args

    with gzip.open(nf, "rb") as f:
        n_chunk = pickle.load(f)
    with gzip.open(ef, "rb") as f:
        e_chunk = pickle.load(f)

    # Whole chunk as arrays: nodes (F, N, 4, 3) = [displacement, velocity,
    # node[2], curvature]; edges: quaternions (F, N, 4) and vectors (F, N, 3, 3)
    # = [angular velocity, ., strain]. The last edge is the ghost edge.
    nodes = np.asarray(n_chunk, dtype=np.float32)
    quats = np.asarray([[e[0] for e in frame] for frame in e_chunk], dtype=np.float32)
    evecs = np.asarray([[e[1] for e in frame] for frame in e_chunk], dtype=np.float32)
    del n_chunk, e_chunk
    n_frames, node_count = nodes.shape[:2]
    pos, vel, curv = nodes[:, :, 0], nodes[:, :, 1], nodes[:, :, 3]
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
    field = pos[:, :, 1:3].copy() if capture_field else None

    # Edge stretch exactly as the shader computes it: e = |r| / dl, where r is
    # the edge's reference vector (the pre-stretched spacing tuned_length/(N-1)
    # along x, see utils.generate_straight_rod) plus the displacement
    # difference. dl is the rest length, so e ~ 1 + pre-strain at rest.
    ref = np.zeros(3, dtype=np.float32)
    ref[0] = length / (node_count - 1) if length is not None else dl
    r = ref + (pos[:, 1:] - pos[:, :-1])                      # (F, N-1, 3)
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

    # Per-element total energy: node terms plus half of each adjacent edge, so
    # the row sum equals the system total. Block-averaged in time.
    edge_e = rot_edge + ss_edge
    elem = ke_node + bt_node
    elem[:, :-1] += 0.5 * edge_e
    elem[:, 1:] += 0.5 * edge_e
    energy_density = block_mean(elem, ENERGY_DENSITY_BLOCK)

    # Damping power, exactly where the shader applies it: linear on interior
    # nodes, angular on edges 1..N-3.
    c_lin, c_ang = dampening
    p_damp = (c_lin * np.sum(v64[:, 1:-1] ** 2, axis=(1, 2))
              + c_ang * np.sum(w64[:, 1:-1] ** 2, axis=(1, 2)))

    # Chunk-averaged kinetic energy per spatial mode (~half each mode's energy):
    # KE_n = 0.5 * (total mass) / 2 * adot_n^2, transverse (Y+Z) and longitudinal (X).
    adot = modal_projection(vel, n_modes).astype(np.float64)
    ke_modes = 0.25 * m_node * (node_count - 1) * np.mean(adot ** 2, axis=0)   # (n_modes, 3)
    modal_ke = np.stack([ke_modes[:, 1] + ke_modes[:, 2], ke_modes[:, 0]], axis=1).astype(np.float32)

    return (n_frames, out_node_data, *[np.asarray(a, dtype=np.float32) for a in sums],
            field, spectra, energy_density, p_damp.astype(np.float32), modal_ke)

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
def extract_data(sim_path, config, m_node, dl, inertia, K_se, K_bt, inspect_nodes, length=None,
                 n_modes=DISPERSION_MODES, nperseg=DISPERSION_NPERSEG):
    """Read all chunks in parallel and reduce them to the arrays the plots need.

    Returns a dict with 'time', 'fs', the energy arrays (ENERGY_KEYS),
    'node_data', 'field_exc', 'excitation_cutoff_frame', 'dispersion',
    'energy_density' (+ '_time'), 'p_damp' and 'modal_ke' (+ '_time').
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
    dampening = tuple(config.get("dampening", (0.0, 0.0)))
    args_list = [(nf, ef, inspect_nodes, m_node, inertia, K_se, K_bt, dl,
                  idx < n_exc_chunks, idx >= n_exc_chunks, n_modes, length, dampening)
                 for idx, (nf, ef) in enumerate(zip(node_files, edge_files))]

    all_node_data = {n: {'pos': [], 'vel': [], 'mom': [], 'quats': [], 'omega': []} for n in inspect_nodes}
    all_energy = {key: [] for key in ENERGY_KEYS}
    field_chunks = []
    density_chunks, density_t, p_damp_chunks = [], [], []
    modal_ke_chunks, modal_ke_t = [], []
    frame_dt = dt * oversamp
    # Modal spectra are accumulated as chunks arrive (in time order), so only
    # one Welch segment of modal data is ever held. Only free-vibration chunks
    # are fed, so the hammer's forced response doesn't smear the spectrum.
    modal_acc = StreamingModalSpectra(fs_native, nperseg=nperseg)
    kw_acc = StreamingKOmegaSpectra(fs_native, nperseg=nperseg)
    spectra_meta = None

    excitation_cutoff_frame = 0
    frames_processed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        for file_idx, res in enumerate(executor.map(_process_single_chunk, args_list)):
            n_frames, nd_dict = res[0], res[1]
            c_energy = res[2:10]
            c_field, c_spectra = res[10], res[11]
            c_density, c_p_damp, c_modal_ke = res[12], res[13], res[14]

            for n in inspect_nodes:
                for key in ('pos', 'vel', 'mom', 'quats', 'omega'):
                    all_node_data[n][key].append(nd_dict[n][key])

            for key, arr in zip(ENERGY_KEYS, c_energy):
                all_energy[key].append(arr)
            if c_field is not None:
                field_chunks.append(c_field)
            if c_spectra is not None:
                c_modal, c_kspace, k_grid, n_nodes, grid_len = c_spectra
                modal_acc.feed(c_modal)
                kw_acc.feed(c_kspace)
                spectra_meta = (k_grid, n_nodes, grid_len)
            density_chunks.append(c_density)
            density_t.append(block_centers(frames_processed, n_frames, ENERGY_DENSITY_BLOCK, frame_dt))
            p_damp_chunks.append(c_p_damp)
            modal_ke_chunks.append(c_modal_ke)
            modal_ke_t.append((frames_processed + 0.5 * (n_frames - 1)) * frame_dt)
            del res, c_spectra

            frames_processed += n_frames
            if file_idx <= 9: # Files 0 through 9 represent the first 10 dispatches
                excitation_cutoff_frame += n_frames

    print(f"[{datetime.now()}] Extraction complete. Stitching arrays...")
    data = {
        "time": np.arange(frames_processed) * (dt * oversamp),
        "fs": fs_native,
        "excitation_cutoff_frame": excitation_cutoff_frame,
        "field_exc": np.concatenate(field_chunks, axis=0) if field_chunks else None,
        "dispersion": finalize_dispersion(modal_acc, kw_acc, *spectra_meta) if spectra_meta else None,
        "energy_density": np.concatenate(density_chunks, axis=0),
        "energy_density_time": np.concatenate(density_t),
        "p_damp": np.concatenate(p_damp_chunks),
        "modal_ke": np.stack(modal_ke_chunks),          # (chunks, n_modes, 2)
        "modal_ke_time": np.asarray(modal_ke_t),
        "node_data": {
            n: {key: np.concatenate(vals) for key, vals in all_node_data[n].items()}
            for n in inspect_nodes
        },
    }
    for key in ENERGY_KEYS:
        data[key] = np.concatenate(all_energy[key])

    # Free up collection lists aggressively
    del all_node_data, all_energy, field_chunks, modal_acc, kw_acc
    del density_chunks, p_damp_chunks, modal_ke_chunks
    gc.collect()
    return data


def make_string_params(m_node, dl, K_se, K_bt, node_count=None, tension=None, length=None):
    """Continuum parameters for the theory overlays (T is None -> no theory)."""
    if length is None:
        length = dl * (node_count - 1)
    n_seg = node_count - 1 if node_count else length / dl
    return {
        "L": float(length),
        "mu": float(m_node * n_seg / length),
        "T": None if tension is None else float(tension),
        "EI": float(np.ravel(K_bt)[0]),
        "EA": float(np.ravel(K_se)[2]),
    }


def build_plot_tasks(data, output_dir, run_id, dt, oversamp, dl, inspect_nodes, string_params):
    """Queue every plot as (func, args, kwargs) for run_plot_tasks."""
    output_dir = Path(output_dir)
    time = data["time"]
    fs_native = data["fs"]
    excitation_cutoff_frame = data["excitation_cutoff_frame"]
    stitched_node_data = data["node_data"]
    field_exc = data["field_exc"]

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
    # Smoothing window for the transfer/flow plots: two fundamental periods.
    transfer_window = 2.0 / f1 if f1 else 0.02
    # Early transient window: the first three fundamental periods.
    early_window = 3.0 / f1 if f1 else 0.03

    release_time = time[excitation_cutoff_frame] if 0 < excitation_cutoff_frame < time.size else None
    density = data.get("energy_density")
    density_t = data.get("energy_density_time")
    density_base = density.min(axis=0) if density is not None else None
    string_len = string_params["L"]

    phases = {
        "Excitation": (0, excitation_cutoff_frame),
        "Free_Vibration": (excitation_cutoff_frame, None),
        "Full_Simulation": (0, None)
    }

    plot_tasks = []

    for phase_name, (start, end) in phases.items():
        p_time = time[start:end]
        p_tke, p_rke, p_bpe, p_spe = (data[k][start:end] for k in ("t_ke", "r_ke", "bt_pe", "ss_pe"))
        p_tke_l, p_rke_t = data["t_ke_l"][start:end], data["r_ke_t"][start:end]
        p_bpe_tw, p_spe_ex = data["bt_pe_tw"][start:end], data["ss_pe_ex"][start:end]

        if p_time.size == 0:
            print(f"[{datetime.now()}] Skipping empty phase '{phase_name}' (no frames in this window).")
            continue

        energy_prefix = str(output_dir / f"run_{run_id:03d}_{phase_name}")
        phase_title = phase_name.replace("_", " ")

        # Queue Global Energy Plots (Only once per phase). Totals have their
        # own plot so the component breakdown stays readable.
        plot_tasks.append((plot_energies, (p_time, p_tke, p_rke, p_bpe, p_spe),
                           {"title_prefix": phase_title, "filename": f"{energy_prefix}_energy.png"}))
        plot_tasks.append((plot_energy_totals, (p_time, p_tke, p_rke, p_bpe, p_spe),
                           {"title_prefix": phase_title, "filename": f"{energy_prefix}_energy_totals.png"}))
        plot_tasks.append((plot_energies, (p_time, p_tke, p_rke, p_bpe, p_spe),
                           {"normalized": True, "title_prefix": phase_title, "filename": f"{energy_prefix}_energy_norm.png"}))
        plot_tasks.append((plot_energies, (p_time, p_tke, p_rke, p_bpe, p_spe),
                           {"mode_transfer": True, "title_prefix": phase_title, "filename": f"{energy_prefix}_energy_mode.png"}))

        # Fine-grained 8-way component split (all phases).
        plot_tasks.append((plot_energy_components,
                           (p_time, p_tke, p_rke, p_bpe, p_spe, p_tke_l, p_rke_t, p_bpe_tw, p_spe_ex),
                           {"title_prefix": phase_title, "filename": f"{energy_prefix}_energy_components.png"}))
        plot_tasks.append((plot_kinetic_trans_long, (p_time, p_tke, p_tke_l),
                           {"window_s": transfer_window, "title_prefix": phase_title, "filename": f"{energy_prefix}_ke_trans_long.png"}))

        # Energy flows: bookkeeping, per-component share, per-element map.
        p_damp = data["p_damp"][start:end]
        p_comps = energy_components(p_tke, p_rke, p_bpe, p_spe, p_tke_l, p_rke_t, p_bpe_tw, p_spe_ex)
        plot_tasks.append((plot_energy_balance, (p_time, p_tke, p_rke, p_bpe, p_spe, p_damp),
                           {"release_time": release_time, "title_prefix": phase_title,
                            "filename": f"{energy_prefix}_energy_balance.png"}))
        plot_tasks.append((plot_energy_share, (p_time, p_comps),
                           {"window_s": transfer_window, "title_prefix": phase_title,
                            "filename": f"{energy_prefix}_energy_share.png"}))
        if density is not None:
            rows = (density_t >= p_time[0]) & (density_t <= p_time[-1])
            if rows.sum() > 1:
                plot_tasks.append((plot_energy_density,
                                   (density_t[rows], density[rows], density_base, string_len),
                                   {"title_prefix": phase_title,
                                    "filename": f"{energy_prefix}_energy_density.png"}))

        # Decay envelopes only for the long phases (free vibration + full run).
        if phase_name in ("Free_Vibration", "Full_Simulation"):
            plot_tasks.append((plot_energy_envelope, (p_time, p_tke, p_rke, p_bpe, p_spe),
                               {"title_prefix": phase_title, "filename": f"{energy_prefix}_energy_envelope.png"}))
            plot_tasks.append((plot_kinetic_transfer, (p_time, p_tke, p_tke_l),
                               {"window_s": transfer_window, "title_prefix": phase_title,
                                "filename": f"{energy_prefix}_ke_transfer.png"}))
            plot_tasks.append((plot_motion_power, (p_time, p_comps, p_damp),
                               {"window_s": transfer_window, "title_prefix": phase_title,
                                "filename": f"{energy_prefix}_motion_power.png"}))

        # Queue Node-Specific Plots
        for n in inspect_nodes:
            p_pos = stitched_node_data[n]['pos'][start:end]
            p_vel = stitched_node_data[n]['vel'][start:end]
            p_mom = stitched_node_data[n]['mom'][start:end]
            p_quats = stitched_node_data[n]['quats'][start:end]
            p_omega = stitched_node_data[n]['omega'][start:end]

            node_prefix = str(output_dir / f"run_{run_id:03d}_node_{n:03d}_{phase_name}")
            node_title = f"Node {n} | {phase_title}"

            plot_tasks.append((plot_node_pos_moment_fft, (p_time, p_pos, p_mom, dt, oversamp),
                               {"moments": True, "title_prefix": node_title, "filename_prefix": f"{node_prefix}_fft"}))

            plot_tasks.append((plot_axis_angle_over_time, (p_time, p_quats),
                               {"title_prefix": node_title, "filename": f"{node_prefix}_angles.png"}))

            plot_tasks.append((plot_angular_velocity, (p_time, p_omega),
                               {"title_prefix": node_title, "filename": f"{node_prefix}_omega.png", "fs": fs_native}))

            # Full-run-only analyses (audio + spectral/decay/polarization);
            # redundant to repeat these per phase.
            if phase_name == "Full_Simulation":
                plot_tasks.append((generate_node_audio, (p_pos, p_vel, p_mom, fs_native),
                                   {"title_prefix": node_title,
                                    "filename_prefix": f"{node_prefix}_audio"}))

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

    # Transient (whole-string) plots for the excitation window
    if field_exc is not None and field_exc.shape[0] > 2:
        exc_time = time[:field_exc.shape[0]]
        # Physical node spacing (tuned length), so x matches the energy map.
        spacing = string_len / (field_exc.shape[1] - 1)
        trans_prefix = str(output_dir / f"run_{run_id:03d}_transient")
        trans_title = f"Run {run_id} | Excitation"
        plot_tasks.append((plot_spacetime_heatmap, (exc_time, field_exc, spacing),
                           {"title_prefix": trans_title, "filename": f"{trans_prefix}_heatmap.png"}))
        plot_tasks.append((plot_waterfall_3d, (exc_time, field_exc, spacing),
                           {"title_prefix": trans_title, "filename": f"{trans_prefix}_waterfall.png"}))

        # Early window: the first three fundamental periods only.
        n_early = int(min(field_exc.shape[0], np.searchsorted(exc_time, early_window)))
        if n_early > 2:
            early_title = f"Run {run_id} | First {early_window * 1e3:.0f} ms (3 periods)" if f1 \
                else f"Run {run_id} | First {early_window * 1e3:.0f} ms"
            plot_tasks.append((plot_spacetime_heatmap, (exc_time[:n_early], field_exc[:n_early], spacing),
                               {"title_prefix": early_title, "filename": f"{trans_prefix}_early_heatmap.png"}))
            plot_tasks.append((plot_waterfall_3d, (exc_time[:n_early], field_exc[:n_early], spacing),
                               {"title_prefix": early_title, "filename": f"{trans_prefix}_early_waterfall.png"}))
            if density is not None:
                rows = density_t <= early_window
                if rows.sum() > 1:
                    plot_tasks.append((plot_energy_density,
                                       (density_t[rows], density[rows], density_base, string_len),
                                       {"title_prefix": early_title,
                                        "filename": f"{trans_prefix}_early_energy_density.png"}))

    # Energy per spatial mode over the whole run
    if data.get("modal_ke") is not None and len(data["modal_ke"]) > 1:
        plot_tasks.append((plot_modal_energy, (data["modal_ke"], data["modal_ke_time"]),
                           {"title_prefix": f"Run {run_id} | Full Simulation",
                            "filename": str(output_dir / f"run_{run_id:03d}_modal_energy.png")}))

    # Dispersion relation from the streamed modal spectra (free vibration)
    if data.get("dispersion") is not None:
        plot_tasks.append((plot_dispersion, (data["dispersion"], string_params),
                           {"title_prefix": f"Run {run_id} | Free Vibration",
                            "filename": str(output_dir / f"run_{run_id:03d}_dispersion.png")}))

    return plot_tasks


def run_plot_tasks(plot_tasks, max_workers=32):
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


def process_and_plot(sim_path, run_id, config, email_config, m_node, dl, inertia, K_se, K_bt, inspect_nodes,
                     tension=None, length=None):
    sim_path = Path(sim_path)
    data = extract_data(sim_path, config, m_node, dl, inertia, K_se, K_bt, inspect_nodes, length=length)

    output_dir = sim_path / "plots"
    output_dir.mkdir(exist_ok=True)
    string_params = make_string_params(m_node, dl, K_se, K_bt, node_count=config.get("node_count"),
                                       tension=tension, length=length)

    plot_tasks = build_plot_tasks(data, output_dir, run_id, config.get("dt", 1e-5),
                                  config.get("oversampling_factor", 100), dl, inspect_nodes, string_params)
    del data
    gc.collect()
    generated_images = run_plot_tasks(plot_tasks)
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
        tension=rod_params.get("tension_force"), length=rod_params.get("tuned_length"),
    )
