import os
import sys
import pickle
import gzip
from pathlib import Path
from datetime import datetime
import smtplib
from email.message import EmailMessage
import concurrent.futures

import numpy as np
import matplotlib
matplotlib.use('Agg') # MUST be set before pyplot import for multiprocessing safety
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R
import gc

# ---------------------------------------------------------
# Helper Functions (Energy)
# ---------------------------------------------------------
def E_T(node, mass):
    velocity = np.array(node[1])
    return 0.5 * mass * np.dot(velocity, velocity)

def E_R(omega_vec, inertia):
    w = np.array(omega_vec)
    I = np.array(inertia)
    if I.ndim == 1: 
        return 0.5 * np.dot(w, I * w)
    return 0.5 * np.dot(w, I @ w)

def E_PB(kappa, K_bt):
    k = np.array(kappa)
    K = np.array(K_bt)
    if K.ndim == 1:
        return 0.5 * np.dot(k, K * k)
    return 0.5 * np.dot(k, K @ k)

def E_PS(strain, K_se):
    s = np.array(strain)
    K = np.array(K_se)
    if K.ndim == 1:
        return 0.5 * np.dot(s, K * s)
    return 0.5 * np.dot(s, K @ s)

# ---------------------------------------------------------
# Parallel Extraction Worker
# ---------------------------------------------------------
def _process_single_chunk(args):
    nf, ef, inspect_nodes, m_node, inertia, K_se, K_bt, dl = args
    
    node_data = {n: {'pos': [], 'vel': [], 'mom': [], 'quats': [], 'omega': []} for n in inspect_nodes}
    t_ke, r_ke, bt_pe, ss_pe = [], [], [], []

    with gzip.open(nf, "rb") as f: n_chunk = pickle.load(f)
    with gzip.open(ef, "rb") as f: e_chunk = pickle.load(f)

    for i in range(len(n_chunk)):
        n_frame, e_frame = n_chunk[i], e_chunk[i]
        
        # Extract specific nodes
        for n in inspect_nodes:
            target_node = n_frame[n]
            node_data[n]['pos'].append(target_node[0])
            node_data[n]['vel'].append(target_node[1])
            node_data[n]['mom'].append(target_node[2])
            
            target_edge_idx = min(n, len(e_frame)-1)
            node_data[n]['quats'].append(e_frame[target_edge_idx][0])
            node_data[n]['omega'].append(e_frame[target_edge_idx][1][0])
        
        # Global energy sums
        t_sum, bt_sum = 0, 0
        for j, n_node in enumerate(n_frame):
            weight = 0.5 if (j == 0 or j == len(n_frame)-1) else 1.0
            t_sum += E_T(n_node, m_node) * weight
            bt_sum += E_PB(n_node[3], K_bt) * weight
            
        r_sum, ss_sum = 0, 0
        for e in e_frame:
            _, e_vecs = e
            r_sum += E_R(e_vecs[0], inertia)
            ss_sum += E_PS(e_vecs[2], K_se)
            
        t_ke.append(t_sum)
        bt_pe.append(bt_sum * dl)
        r_ke.append(r_sum)
        ss_pe.append(ss_sum * dl)

    # Convert to numpy arrays directly to minimize IPC overhead
    out_node_data = {}
    for n in inspect_nodes:
        out_node_data[n] = {
            'pos': np.array(node_data[n]['pos'], dtype=np.float32),
            'vel': np.array(node_data[n]['vel'], dtype=np.float32),
            'mom': np.array(node_data[n]['mom'], dtype=np.float32),
            'quats': np.array(node_data[n]['quats'], dtype=np.float32),
            'omega': np.array(node_data[n]['omega'], dtype=np.float32)
        }

    return (len(n_chunk), out_node_data,
            np.array(t_ke, dtype=np.float32), np.array(r_ke, dtype=np.float32), 
            np.array(bt_pe, dtype=np.float32), np.array(ss_pe, dtype=np.float32))

# ---------------------------------------------------------
# Plotting Functions (Designed for Parallel Workers)
# ---------------------------------------------------------
def plot_node_pos_moment_fft(time, pos, mom, dt, oversampling_factor, cutoff=20_000, 
                             moments=False, fundamental_weight=0.2, 
                             title_prefix="", filename_prefix=""):
    T = len(time)
    freqs = np.fft.rfftfreq(T, dt * oversampling_factor)
    components = [("x", pos[:, 0], mom[:, 0]),
                  ("y", pos[:, 1], mom[:, 1]),
                  ("z", pos[:, 2], mom[:, 2])]

    def fft_mag(signal):
        sig = signal - np.mean(signal)
        window = np.hanning(len(sig))
        return np.abs(np.fft.rfft(sig * window))

    def stiff_string_model(n, f1_val, B_val):
        return n * f1_val * np.sqrt(1 + B_val * (n**2))

    generated_files = []

    for label, p_data, m_data in components:
        n_plots = 4 if moments else 3
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.8 * n_plots), sharex=False)

        # Displacement
        ax = axes[0]
        p_max = np.max(np.abs(p_data))
        p_scaled = p_data / p_max if p_max > 0 else p_data
        ax.plot(time, p_scaled, color="tab:blue", label=f"{label} Disp")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Normalized Disp.", color="tab:blue")
        ax.set_xlabel("Time (s)")
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([f"{-p_max:.3e}", "0", f"{p_max:.3e}"])
        ax.set_title(f"[{title_prefix}] {label}-axis Displacement over Time")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plot_idx = 1
        
        # Moments
        if moments:
            axm = axes[plot_idx]
            m_max = np.max(np.abs(m_data))
            m_scaled = m_data / m_max if m_max > 0 else m_data
            axm.plot(time, m_scaled, color="tab:green", label=f"{label} Moment")
            axm.set_ylim(-1.05, 1.05)
            axm.set_xlabel("Time (s)")
            axm.set_ylabel("Normalized Moment")
            axm.set_yticks([-1, 0, 1])
            axm.set_title(f"[{title_prefix}] {label}-axis Moment over Time")
            axm.legend(loc='upper right')
            axm.grid(True, alpha=0.3)
            plot_idx += 1

        # FFT (No Fit)
        bins = fft_mag(p_data)
        mag_norm = bins / np.max(bins) if np.max(bins) > 0 else bins
        
        ax_fft = axes[plot_idx]
        ax_fft.plot(freqs, mag_norm, color='tab:cyan', label='Spectrum')
        ax_fft.set_xlim(0, cutoff)
        ax_fft.set_title(f"[{title_prefix}] {label}-axis FFT Spectrum (Raw)")
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Normalized Magnitude")
        ax_fft.legend(loc='upper right')
        ax_fft.grid(True, alpha=0.3)
        
        plot_idx += 1

        # FFT (With Fit)
        ax_fit = axes[plot_idx]
        peak_indices, _ = find_peaks(mag_norm, height=0.02, distance=5)
        peak_freqs = freqs[peak_indices]
        peak_mags = mag_norm[peak_indices]

        if len(peak_freqs) >= 2:
            sig_idx = np.where(peak_mags > fundamental_weight)[0]
            f1_est = peak_freqs[sig_idx[0]] if len(sig_idx) > 0 else peak_freqs[np.argmax(peak_mags)]
            
            measured_partials, harmonic_indices = [], []
            consecutive_misses = 0
            for n in range(1, 20):
                if consecutive_misses > 2: break
                target = n * f1_est
                idx_closest = np.argmin(np.abs(peak_freqs - target))
                if np.abs(peak_freqs[idx_closest] - target) < 0.45 * f1_est:
                    measured_partials.append(peak_freqs[idx_closest])
                    harmonic_indices.append(n)
                    consecutive_misses = 0
                else:
                    consecutive_misses += 1

            ns, fns = np.array(harmonic_indices), np.array(measured_partials)
            f1_fit, B_fit = f1_est, 0.0
            
            if len(ns) >= 4:
                try:
                    popt, _ = curve_fit(stiff_string_model, ns, fns, p0=[f1_est, 0.0001])
                    f1_fit, B_fit = popt
                except RuntimeError: pass

            ax_fit.plot(freqs, mag_norm, color='lightgray', label='Spectrum')
            ax_fit.plot(peak_freqs, peak_mags, 'x', color='tab:blue', label='Detected Peaks')
            
            fitted_freqs = stiff_string_model(ns, f1_fit, B_fit)
            if len(fitted_freqs) > 0:
                ax_fit.scatter(fitted_freqs, np.interp(fitted_freqs, freqs, mag_norm), 
                               color='tab:red', zorder=5, label=f'Fit (B={B_fit:.2e})')

            for f_ideal in (ns * f1_fit):
                ax_fit.axvline(f_ideal, color='green', linestyle=':', alpha=0.5)

            ax_fit.set_xlim(0, cutoff) 
            if len(fns) > 0: ax_fit.set_xlim(0, max(fns) * 1.2)
            ax_fit.set_title(f"[{title_prefix}] {label}-axis FFT Inharmonicity Fit (B = {B_fit:.5f})")
        else:
            ax_fit.plot(freqs, mag_norm, color='lightgray', label='Spectrum')
            ax_fit.set_xlim(0, cutoff)
            ax_fit.set_title(f"[{title_prefix}] {label}-axis FFT (Not enough peaks for fit)")

        ax_fit.set_xlabel("Frequency (Hz)")
        ax_fit.set_ylabel("Normalized Magnitude")
        ax_fit.legend(loc='upper right')
        ax_fit.grid(True, alpha=0.3)
        plt.tight_layout()
        
        file_path = f"{filename_prefix}_{label}.png"
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close(fig) 
        generated_files.append(file_path)

    return generated_files

def plot_angular_velocity(time, omega, title_prefix="", filename=""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    omega_mag = np.linalg.norm(omega, axis=1)
    
    ax.plot(time, np.abs(omega[:, 0]), label="|Omega X|", alpha=0.6)
    ax.plot(time, np.abs(omega[:, 1]), label="|Omega Y|", alpha=0.6)
    ax.plot(time, np.abs(omega[:, 2]), label="|Omega Z|", alpha=0.6)
    ax.plot(time, omega_mag, label="Magnitude", color='black', linewidth=1.5)
    
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"[{title_prefix}] Absolute Angular Velocity Over Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]

def plot_axis_angle_over_time(time, quats, title_prefix="", filename=""):
    axes_arr, angles = [], []
    for q in quats:
        q_norm = q / np.linalg.norm(q)
        rotvec = R.from_quat(q_norm).as_rotvec()
        angle = np.linalg.norm(rotvec)
        axis = rotvec / angle if angle > 1e-8 else np.array([0.0, 0.0, 0.0])
        axes_arr.append(axis)
        angles.append(angle)

    axes_arr, angles = np.array(axes_arr), np.array(angles)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(time, np.degrees(angles), label="Rotation Angle", color='tab:purple')
    ax1.set_ylabel("Angle (degrees)")
    ax1.set_title(f"[{title_prefix}] Axis-Angle Orientation Over Time")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2.plot(time, axes_arr[:, 0], label="Axis X", color="tab:red")
    ax2.plot(time, axes_arr[:, 1], label="Axis Y", color="tab:green")
    ax2.plot(time, axes_arr[:, 2], label="Axis Z", color="tab:blue")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Axis Component")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]

def plot_energies(time, trans_ke, rot_ke, bend_pe, shear_pe, print_totals=False, 
                  normalized=False, mode_transfer=False, title_prefix="", filename=""):
    # Remove the potential energy offset from frame 0
    bend_pe = bend_pe - bend_pe[0]
    shear_pe = shear_pe - shear_pe[0]
    
    total_kin = trans_ke + rot_ke
    total_pot = bend_pe + shear_pe
    total_energy = total_kin + total_pot
    
    fig = plt.figure(figsize=(12, 6))
    
    if normalized:
        title = f"[{title_prefix}] Normalized Energy Breakdown (Stacked)"
        plt.stackplot(time, trans_ke, rot_ke, bend_pe, shear_pe, 
                      labels=["Translational KE", "Rotational KE", "Bend/Twist PE", "Shear/Stretch PE"], 
                      colors=["tab:blue", "tab:purple", "tab:orange", "tab:green"], alpha=0.8)
        plt.plot(time, total_energy, label="TOTAL SYSTEM ENERGY", color='black', linestyle='--', linewidth=1.5)
        plt.ylabel("Energy (Joules)")
    elif mode_transfer:
        with np.errstate(divide='ignore', invalid='ignore'):
            trans_ke = trans_ke / total_energy
            rot_ke = rot_ke / total_energy
            bend_pe = bend_pe / total_energy
            shear_pe = shear_pe / total_energy
            total_kin = total_kin / total_energy
            total_pot = total_pot / total_energy
        title = f"[{title_prefix}] Energy Mode Transfer (Fraction of Total)"
        plt.plot(time, trans_ke, label="Translational KE", alpha=0.4, linestyle=':')
        plt.plot(time, rot_ke, label="Rotational KE", alpha=0.4, linestyle=':')
        plt.plot(time, bend_pe, label="Bend/Twist PE", alpha=0.4, linestyle=':')
        plt.plot(time, shear_pe, label="Shear/Stretch PE", alpha=0.4, linestyle=':')
        plt.plot(time, total_kin, label="TOTAL Kinetic", linewidth=2, color='blue')
        plt.plot(time, total_pot, label="TOTAL Potential", linewidth=2, color='orange')
        plt.ylabel("Fraction of Total Energy")
        plt.ylim(-0.1, 1.1)
    else:
        title = f"[{title_prefix}] Cosserat Rod Energy Breakdown (Offset Removed)"
        plt.plot(time, trans_ke, label="Translational KE", alpha=0.4, linestyle=':')
        plt.plot(time, rot_ke, label="Rotational KE", alpha=0.4, linestyle=':')
        plt.plot(time, bend_pe, label="Bend/Twist PE", alpha=0.4, linestyle=':')
        plt.plot(time, shear_pe, label="Shear/Stretch PE", alpha=0.4, linestyle=':')
        
        plt.plot(time, total_kin, label="TOTAL Kinetic", linewidth=2, color='blue')
        plt.plot(time, total_pot, label="TOTAL Potential", linewidth=2, color='orange')
        
        plt.plot(time, total_energy, label="TOTAL SYSTEM ENERGY", color='black', linestyle='--', linewidth=1.5)
        plt.ylabel("Energy (Joules)")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if print_totals and not normalized and not mode_transfer:
        print(f"Total Energy Mean: {np.mean(total_energy):.6e} J")
        print(f"Energy Variation: {np.std(total_energy):.6e} J")

    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]

# ---------------------------------------------------------
# Core Memory-Efficient Extraction
# ---------------------------------------------------------
def process_and_plot(sim_path, run_id, config, email_config, m_node, dl, inertia, K_se, K_bt, inspect_nodes):
    sim_path = Path(sim_path)
    node_files = sorted(sim_path.glob("n_*.pkl.gz"))
    edge_files = sorted(sim_path.glob("e_*.pkl.gz"))
    
    dt = config.get("dt", 1e-5)
    oversamp = config.get("oversampling_factor", 100)
    
    print(f"[{datetime.now()}] Dispatching data extraction for {len(node_files)} chunks across 16 workers...")
    print(f"[{datetime.now()}] Target nodes for inspection: {inspect_nodes}")
    
    # 1. Fully Parallelized Extraction
    args_list = [(nf, ef, inspect_nodes, m_node, inertia, K_se, K_bt, dl) for nf, ef in zip(node_files, edge_files)]
    
    all_node_data = {n: {'pos': [], 'vel': [], 'mom': [], 'quats': [], 'omega': []} for n in inspect_nodes}
    all_tke, all_rke, all_bpe, all_spe = [], [], [], []
    
    excitation_cutoff_frame = 0
    frames_processed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        for file_idx, res in enumerate(executor.map(_process_single_chunk, args_list)):
            n_frames, nd_dict, c_tke, c_rke, c_bpe, c_spe = res
            
            for n in inspect_nodes:
                all_node_data[n]['pos'].append(nd_dict[n]['pos'])
                all_node_data[n]['vel'].append(nd_dict[n]['vel'])
                all_node_data[n]['mom'].append(nd_dict[n]['mom'])
                all_node_data[n]['quats'].append(nd_dict[n]['quats'])
                all_node_data[n]['omega'].append(nd_dict[n]['omega'])
                
            all_tke.append(c_tke)
            all_rke.append(c_rke)
            all_bpe.append(c_bpe)
            all_spe.append(c_spe)
            
            frames_processed += n_frames
            if file_idx <= 9: # Files 0 through 9 represent the first 10 dispatches
                excitation_cutoff_frame += n_frames

    print(f"[{datetime.now()}] Extraction complete. Stitching arrays...")
    time = np.arange(frames_processed) * (dt * oversamp)
    
    stitched_node_data = {}
    for n in inspect_nodes:
        stitched_node_data[n] = {
            'pos': np.concatenate(all_node_data[n]['pos']),
            'vel': np.concatenate(all_node_data[n]['vel']),
            'mom': np.concatenate(all_node_data[n]['mom']),
            'quats': np.concatenate(all_node_data[n]['quats']),
            'omega': np.concatenate(all_node_data[n]['omega'])
        }
        
    t_ke = np.concatenate(all_tke)
    r_ke = np.concatenate(all_rke)
    bt_pe = np.concatenate(all_bpe)
    ss_pe = np.concatenate(all_spe)

    # Free up collection lists aggressively
    del all_node_data, all_tke, all_rke, all_bpe, all_spe
    gc.collect()

    # ---------------------------------------------------------
    # Prepare Plotting Tasks for Multiprocessing
    # ---------------------------------------------------------
    output_dir = sim_path / "plots"
    output_dir.mkdir(exist_ok=True)
    
    phases = {
        "Excitation": (0, excitation_cutoff_frame),
        "Free_Vibration": (excitation_cutoff_frame, None),
        "Full_Simulation": (0, None)
    }

    plot_tasks = []

    for phase_name, (start, end) in phases.items():
        p_time = time[start:end]
        p_tke, p_rke, p_bpe, p_spe = t_ke[start:end], r_ke[start:end], bt_pe[start:end], ss_pe[start:end]
        
        energy_prefix = str(output_dir / f"run_{run_id:03d}_{phase_name}")
        phase_title = phase_name.replace("_", " ")

        # Queue Global Energy Plots (Only once per phase)
        plot_tasks.append((plot_energies, (p_time, p_tke, p_rke, p_bpe, p_spe), 
                           {"print_totals": (phase_name == "Full_Simulation"), "title_prefix": phase_title, "filename": f"{energy_prefix}_energy.png"}))
        plot_tasks.append((plot_energies, (p_time, p_tke, p_rke, p_bpe, p_spe), 
                           {"normalized": True, "title_prefix": phase_title, "filename": f"{energy_prefix}_energy_norm.png"}))
        plot_tasks.append((plot_energies, (p_time, p_tke, p_rke, p_bpe, p_spe), 
                           {"mode_transfer": True, "title_prefix": phase_title, "filename": f"{energy_prefix}_energy_mode.png"}))

        # Queue Node-Specific Plots
        for n in inspect_nodes:
            p_pos = stitched_node_data[n]['pos'][start:end]
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
                               {"title_prefix": node_title, "filename": f"{node_prefix}_omega.png"}))

    # ---------------------------------------------------------
    # Execute Plotting in Parallel
    # ---------------------------------------------------------
    print(f"[{datetime.now()}] Dispatching {len(plot_tasks)} plotting tasks to 16 concurrent workers...")
    generated_images = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(func, *args, **kwargs) for func, args, kwargs in plot_tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                generated_images.extend(future.result()) 
            except Exception as e:
                print(f"[{datetime.now()}] A plot worker encountered an error: {e}")

    # ---------------------------------------------------------
    # Email Dispatch
    # ---------------------------------------------------------
    print(f"[{datetime.now()}] Plotting complete. Attached {len(generated_images)} images. Dispatching email...")
    msg = EmailMessage()
    msg['Subject'] = f"Piano Sim Run {run_id:03d} Completed - Plot Suite"
    msg['From'] = email_config['sender_email']
    msg['To'] = email_config['receiver_email']
    msg.set_content(f"Simulation {run_id:03d} plots attached. Memory-efficient parallel generation successful for {len(inspect_nodes)} nodes.")

    for img_path in generated_images:
        try:
            with open(img_path, 'rb') as f:
                img_data = f.read()
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

# ---------------------------------------------------------
# CLI Execution Wrapper
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    
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
        print(f"[{datetime.now()}] FATAL ERROR: Email environment variables not set.")
        print("Please ensure SENDER_EMAIL, SENDER_PASSWORD, and RECEIVER_EMAIL are defined in your .env file.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("sim_path", type=str)
    args = parser.parse_args()
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
        inspect_nodes=inspect_nodes
    )
