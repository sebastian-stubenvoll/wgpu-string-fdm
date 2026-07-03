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
# Plotting Functions (Designed for Parallel Workers)
# ---------------------------------------------------------
def plot_node_pos_vel_moment_fft(time, pos, vel, mom, dt, oversampling_factor, cutoff=20_000, 
                                 show_velocities=False, moments=False, fundamental_weight=0.2, 
                                 title_prefix="", filename_prefix=""):
    T = len(time)
    freqs = np.fft.rfftfreq(T, dt * oversampling_factor)
    components = [("x", pos[:, 0], vel[:, 0], mom[:, 0]),
                  ("y", pos[:, 1], vel[:, 1], mom[:, 1]),
                  ("z", pos[:, 2], vel[:, 2], mom[:, 2])]

    def fft_mag(signal):
        sig = signal - np.mean(signal)
        window = np.hanning(len(sig))
        return np.abs(np.fft.rfft(sig * window))

    def stiff_string_model(n, f1_val, B_val):
        return n * f1_val * np.sqrt(1 + B_val * (n**2))

    generated_files = []

    for label, p_data, v_data, m_data in components:
        n_plots = 3 if moments else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.8 * n_plots), sharex=False)

        # Displacement (& optionally Velocity)
        ax = axes[0]
        p_max = np.max(np.abs(p_data))
        p_scaled = p_data / p_max if p_max > 0 else p_data
        l1 = ax.plot(time, p_scaled, color="tab:blue", label=f"{label} Disp")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Normalized Disp.", color="tab:blue")
        ax.set_xlabel("Time (s)")
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([f"{-p_max:.3e}", "0", f"{p_max:.3e}"])
        ax.set_title(f"[{title_prefix}] {label}-axis Displacement over Time")
        
        lines = l1
        labels = [l.get_label() for l in l1]

        if show_velocities:
            v_max = np.max(np.abs(v_data))
            v_scaled = v_data / v_max if v_max > 0 else v_data
            ax2 = ax.twinx()
            l2 = ax2.plot(time, v_scaled, "--", color="tab:orange", label=f"{label} Vel")
            ax2.set_ylim(-1.05, 1.05)
            ax2.set_ylabel("Normalized Vel.", color="tab:orange")
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels([f"{-v_max:.3e}", "0", f"{v_max:.3e}"])
            ax.set_title(f"[{title_prefix}] {label}-axis Displacement & Velocity over Time")
            lines += l2
            labels += [l.get_label() for l in l2]
        
        ax.legend(lines, labels, loc='upper right')
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

        # FFT
        bins = fft_mag(p_data)
        ax_fit = axes[plot_idx]
        mag_norm = bins / np.max(bins) if np.max(bins) > 0 else bins
        
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
        plt.close(fig) # Prevent memory leaks
        generated_files.append(file_path)

    return generated_files

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
    total_kin = trans_ke + rot_ke
    total_pot = bend_pe + shear_pe
    total_energy = total_kin + total_pot
    
    if normalized:
        trans_ke = trans_ke - trans_ke[0]
        rot_ke = rot_ke - rot_ke[0]
        bend_pe = bend_pe - bend_pe[0]
        shear_pe = shear_pe - shear_pe[0]
        total_kin = total_kin - total_kin[0]
        total_pot = total_pot - total_pot[0]
        total_energy = total_energy - total_energy[0]
        title = f"[{title_prefix}] Normalized Energy Breakdown (Offset Removed)"
    elif mode_transfer:
        with np.errstate(divide='ignore', invalid='ignore'):
            trans_ke = trans_ke / total_energy
            rot_ke = rot_ke / total_energy
            bend_pe = bend_pe / total_energy
            shear_pe = shear_pe / total_energy
            total_kin = total_kin / total_energy
            total_pot = total_pot / total_energy
        title = f"[{title_prefix}] Energy Mode Transfer (Fraction of Total)"
    else:
        title = f"[{title_prefix}] Cosserat Rod Energy Breakdown"

    fig = plt.figure(figsize=(12, 6))
    plt.plot(time, trans_ke, label="Translational KE", alpha=0.4, linestyle=':')
    plt.plot(time, rot_ke, label="Rotational KE", alpha=0.4, linestyle=':')
    plt.plot(time, bend_pe, label="Bend/Twist PE", alpha=0.4, linestyle=':')
    plt.plot(time, shear_pe, label="Shear/Stretch PE", alpha=0.4, linestyle=':')
    
    plt.plot(time, total_kin, label="TOTAL Kinetic", linewidth=2, color='blue')
    plt.plot(time, total_pot, label="TOTAL Potential", linewidth=2, color='orange')
    
    if not mode_transfer:
        plt.plot(time, total_energy, label="TOTAL SYSTEM ENERGY", color='black', linestyle='--', linewidth=1.5)
        plt.ylabel("Energy (Joules)")
    else:
        plt.ylabel("Fraction of Total Energy")
        plt.ylim(-0.1, 1.1)

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

def plot_phase_space(pos, vel, title_prefix="", filename=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ['X', 'Y', 'Z']
    
    for i in range(3):
        axes[i].plot(pos[:, i], vel[:, i], color='purple', alpha=0.6, linewidth=0.5, label="Trajectory")
        axes[i].set_title(f"[{title_prefix}] {labels[i]}-Axis Phase Space")
        axes[i].set_xlabel("Displacement (m)")
        axes[i].set_ylabel("Velocity (m/s)")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]

# ---------------------------------------------------------
# Core Memory-Efficient Extraction
# ---------------------------------------------------------
def process_and_plot(sim_path, run_id, config, email_config, m_node, dl, inertia, K_se, K_bt, node_idx=10):
    sim_path = Path(sim_path)
    node_files = sorted(sim_path.glob("n_*.pkl.gz"))
    edge_files = sorted(sim_path.glob("e_*.pkl.gz"))
    
    dt = config.get("dt", 1e-5)
    oversamp = config.get("oversampling_factor", 100)
    
    # Pre-allocate specific arrays
    t_ke, r_ke, bt_pe, ss_pe = [], [], [], []
    pos, vel, mom = [], [], []
    quats = []
    
    excitation_cutoff_frame = 0
    frames_processed = 0

    print(f"[{datetime.now()}] Streaming {len(node_files)} chunks for memory-efficient extraction...")
    
    for file_idx, (nf, ef) in enumerate(zip(node_files, edge_files)):
        with gzip.open(nf, "rb") as f: n_chunk = pickle.load(f)
        with gzip.open(ef, "rb") as f: e_chunk = pickle.load(f)
        
        for i in range(len(n_chunk)):
            n_frame, e_frame = n_chunk[i], e_chunk[i]
            
            target_node = n_frame[node_idx]
            pos.append(target_node[0])
            vel.append(target_node[1])
            mom.append(target_node[2])
            
            target_edge_idx = min(node_idx, len(e_frame)-1)
            quats.append(e_frame[target_edge_idx][0])
            
            t_sum, bt_sum = 0, 0
            for j, n in enumerate(n_frame):
                weight = 0.5 if (j == 0 or j == len(n_frame)-1) else 1.0
                t_sum += E_T(n, m_node) * weight
                bt_sum += E_PB(n[3], K_bt) * weight
                
            r_sum, ss_sum = 0, 0
            for e in e_frame:
                _, e_vecs = e
                r_sum += E_R(e_vecs[0], inertia)
                ss_sum += E_PS(e_vecs[2], K_se)
                
            t_ke.append(t_sum)
            bt_pe.append(bt_sum * dl)
            r_ke.append(r_sum)
            ss_pe.append(ss_sum * dl)
            
            frames_processed += 1
            
        if file_idx == 9:
            excitation_cutoff_frame = frames_processed

    time = np.arange(frames_processed) * (dt * oversamp)
    pos, vel, mom = np.array(pos), np.array(vel), np.array(mom)
    quats = np.array(quats)
    t_ke, r_ke, bt_pe, ss_pe = np.array(t_ke), np.array(r_ke), np.array(bt_pe), np.array(ss_pe)

    # Free up memory aggressively before spawning plot workers
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
        p_pos, p_vel, p_mom, p_quats = pos[start:end], vel[start:end], mom[start:end], quats[start:end]
        p_tke, p_rke, p_bpe, p_spe = t_ke[start:end], r_ke[start:end], bt_pe[start:end], ss_pe[start:end]
        
        prefix = str(output_dir / f"run_{run_id:03d}_{phase_name}")
        title = phase_name.replace("_", " ")

        # Queue FFT Task
        plot_tasks.append((
            plot_node_pos_vel_moment_fft, 
            (p_time, p_pos, p_vel, p_mom, dt, oversamp), 
            {"show_velocities": (phase_name == "Excitation"), "moments": True, 
             "title_prefix": title, "filename_prefix": f"{prefix}_fft"}
        ))

        # Queue Axis Angle Task
        plot_tasks.append((
            plot_axis_angle_over_time, 
            (p_time, p_quats), 
            {"title_prefix": title, "filename": f"{prefix}_angles.png"}
        ))

        # Queue Standard Energy
        plot_tasks.append((
            plot_energies, 
            (p_time, p_tke, p_rke, p_bpe, p_spe), 
            {"print_totals": (phase_name == "Full_Simulation"), "title_prefix": title, "filename": f"{prefix}_energy.png"}
        ))

        # Queue Normalized Energy
        plot_tasks.append((
            plot_energies, 
            (p_time, p_tke, p_rke, p_bpe, p_spe), 
            {"normalized": True, "title_prefix": title, "filename": f"{prefix}_energy_norm.png"}
        ))

        # Queue Mode Transfer
        plot_tasks.append((
            plot_energies, 
            (p_time, p_tke, p_rke, p_bpe, p_spe), 
            {"mode_transfer": True, "title_prefix": title, "filename": f"{prefix}_energy_mode.png"}
        ))

        # Queue Phase Space
        if phase_name == "Free_Vibration":
            plot_tasks.append((
                plot_phase_space, 
                (p_pos, p_vel), 
                {"title_prefix": title, "filename": f"{prefix}_phasespace.png"}
            ))

    # ---------------------------------------------------------
    # Execute Plotting in Parallel
    # ---------------------------------------------------------
    print(f"[{datetime.now()}] Dispatching {len(plot_tasks)} plotting tasks to 8 concurrent workers...")
    generated_images = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(func, *args, **kwargs) for func, args, kwargs in plot_tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                # Each function returns a list of filepaths it generated
                generated_images.extend(future.result()) 
            except Exception as e:
                print(f"[{datetime.now()}] A plot worker encountered an error: {e}")

    # ---------------------------------------------------------
    # Email Dispatch
    # ---------------------------------------------------------
    print(f"[{datetime.now()}] Plotting complete. Dispatching email...")
    msg = EmailMessage()
    msg['Subject'] = f"Piano Sim Run {run_id:03d} Completed - Plot Suite"
    msg['From'] = email_config['sender_email']
    msg['To'] = email_config['receiver_email']
    msg.set_content(f"Simulation {run_id:03d} plots attached.\nMemory-efficient parallel generation successful.")

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
    
    # 1. Environment Variable Validation Check
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")
    smpt_server = os.environ.get("SMPT_SERVER")
    smpt_port = os.environ.get("SMPT_PORT")

    if not sender_email or not sender_password or not receiver_email:
        print(f"[{datetime.now()}] FATAL ERROR: Email environment variables not set.")
        print("Please ensure SENDER_EMAIL, SENDER_PASSWORD, and RECEIVER_EMAIL are configured properly.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("sim_path", type=str)
    parser.add_argument("--node_idx", type=int, default=64, help="Node index to track for plots")
    args = parser.parse_args()
    target_dir = Path(args.sim_path).resolve()

    param_file = target_dir / "parameters.json"
    with open(param_file, "r") as f:
        metadata = json.load(f)

    # Config setup
    rod = metadata["rod_derived"]
    email_config_dict = {
        "sender_email": sender_email,
        "sender_password": sender_password,
        "receiver_email": receiver_email,
        "smtp_server": smpt_server,
        "smtp_port": smpt_port
    }

    process_and_plot(
        sim_path=target_dir,
        run_id=metadata.get("run_id", 0),
        config=metadata["config"],
        email_config=email_config_dict,
        m_node=rod["m_node"], dl=rod["dl"], 
        inertia=np.array(rod["inertia"]), K_se=np.array(rod["K_se"]), K_bt=np.array(rod["K_bt"]),
        node_idx=args.node_idx
    )
