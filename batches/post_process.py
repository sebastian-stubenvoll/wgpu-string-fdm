import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.signal import find_peaks, butter, sosfiltfilt
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R
import smtplib
from email.message import EmailMessage
import os
import threading

plot_lock = threading.Lock()

def E_T(node, mass):
    velocity = np.array(node[1])
    return 0.5 * mass * np.dot(velocity, velocity)

def E_R(omega_vec, inertia):
    w = np.array(omega_vec)
    I = np.array(inertia)
    if I.ndim == 1: 
        return 0.5 * np.dot(w, I * w)
    return 0.5 * np.dot(w, np.dot(I, w))

def E_PB(kappa, K_bt):
    k = np.array(kappa)
    K = np.array(K_bt)
    return 0.5 * np.dot(k, K * k)

def E_PS(strain, K_se):
    s = np.array(strain)
    k = np.array(K_se)
    return 0.5 * np.dot(s, k * s)



def plot_node_pos_vel_moment_fft(
    pos, vel, mom, 
    dt,
    oversampling_factor,
    cutoff=20_000,
    moments=False,
    save_output=False,
    windowing=False,
    fundamental_weight=0.2,
    prefix="unkown"
):
    T = len(pos)
    time = np.arange(T) * dt * oversampling_factor
    freqs = np.fft.rfftfreq(T, dt * oversampling_factor)

    components = [
        ("x", pos[:, 0], vel[:, 0], mom[:, 0]),
        ("y", pos[:, 1], vel[:, 1], mom[:, 1]),
        ("z", pos[:, 2], vel[:, 2], mom[:, 2]),
    ]

    def fft_mag(signal):
        sig = signal - np.mean(signal)
        if windowing:
            window = np.hanning(len(sig))
            sig = sig * window
        fft = np.fft.rfft(sig)
        return np.abs(fft)

    def stiff_string_model(n, f1_val, B_val):
        return n * f1_val * np.sqrt(1 + B_val * (n**2))

    saved_files = []

    for label, p_data, v_data, m_data in components:
        n_plots = 3 + int(moments)
        fig, axes = plt.subplots(
            n_plots, 1,
            figsize=(10, 2.8 * n_plots),
            sharex=False,
        )

        ax = axes[0]
        p_max = np.max(np.abs(p_data))
        v_max = np.max(np.abs(v_data))

        p_scaled = p_data / p_max if p_max > 0 else p_data
        v_scaled = v_data / v_max if v_max > 0 else v_data

        ax.plot(time, p_scaled, color="tab:blue")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"{label} disp", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([f"{-p_max:.3e}", "0", f"{p_max:.3e}"])

        ax2 = ax.twinx()
        ax2.plot(time, v_scaled, "--", color="tab:orange")
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_ylabel(f"{label} vel", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels([f"{-v_max:.3e}", "0", f"{v_max:.3e}"])
        ax.set_title(f"{label} — displacement & velocity (time)")
        ax.grid(True, alpha=0.3)

        plot_idx = 1
        if moments:
            axm = axes[plot_idx]
            m_max = np.max(np.abs(m_data))
            m_scaled = m_data / m_max if m_max > 0 else m_data

            axm.plot(time, m_scaled, color="tab:green")
            axm.set_ylim(-1.05, 1.05)
            axm.set_ylabel(f"{label} moment")
            axm.set_yticks([-1, 0, 1])
            axm.set_yticklabels([f"{-m_max:.3e}", "0", f"{m_max:.3e}"])
            axm.set_title(f"{label} — moment (time)")
            axm.grid(True, alpha=0.3)
            plot_idx += 1

        bins = fft_mag(p_data)
        axf = axes[plot_idx]
        axf.plot(freqs, bins, color="tab:blue")
        axf.set_ylabel("FFT |disp|")
        axf.set_title(f"{label} — displacement FFT")
        axf.set_xlim(0, cutoff)
        axf.grid(True, alpha=0.3)
        plot_idx += 1

        ax_fit = axes[plot_idx]
        mag_norm = bins / np.max(bins) if np.max(bins) > 0 else bins
        
        peak_indices, _ = find_peaks(mag_norm, height=0.02, distance=5)
        peak_freqs = freqs[peak_indices]
        peak_mags = mag_norm[peak_indices]

        if len(peak_freqs) >= 2:
            significant_peaks_indices = np.where(peak_mags > fundamental_weight)[0]
            if len(significant_peaks_indices) > 0:
                f1_est = peak_freqs[significant_peaks_indices[0]]
            else:
                f1_est = peak_freqs[np.argmax(peak_mags)]

            measured_partials = []
            harmonic_indices = []
            
            search_tolerance = 0.45 * f1_est 
            consecutive_misses = 0

            for n in range(1, 20):
                if consecutive_misses > 2:
                    break
                target = n * f1_est
                idx_closest = np.argmin(np.abs(peak_freqs - target))
                closest_freq = peak_freqs[idx_closest]
                
                if np.abs(closest_freq - target) < search_tolerance:
                    measured_partials.append(closest_freq)
                    harmonic_indices.append(n)
                    consecutive_misses = 0
                else:
                    consecutive_misses += 1

            ns = np.array(harmonic_indices)
            fns = np.array(measured_partials)

            f1_fit, B_fit = f1_est, 0.0
            
            if len(ns) >= 4:
                try:
                    popt, pcov = curve_fit(stiff_string_model, ns, fns, p0=[f1_est, 0.0001])
                    f1_fit, B_fit = popt
                except RuntimeError:
                    pass
            
            ax_fit.plot(freqs, mag_norm, color='lightgray', label='Spectrum')
            ax_fit.plot(peak_freqs, peak_mags, 'x', color='tab:blue', label='Detected Peaks')

            fitted_freqs = stiff_string_model(ns, f1_fit, B_fit)
            
            if len(fitted_freqs) > 0:
                ax_fit.scatter(fitted_freqs, np.interp(fitted_freqs, freqs, mag_norm), 
                               color='tab:red', zorder=5, label=f'Fit (B={B_fit:.2e})')

            ideal_freqs = ns * f1_fit
            for f_ideal in ideal_freqs:
                ax_fit.axvline(f_ideal, color='green', linestyle=':', alpha=0.5)

            ax_fit.set_xlim(0, cutoff) 
            if len(fns) > 0:
                 ax_fit.set_xlim(0, max(fns) * 1.2)
                 
            ax_fit.set_title(f"{label} — Inharmonicity Fit (B = {B_fit:.5f})")
            ax_fit.set_xlabel("Frequency [Hz]")
            ax_fit.legend(loc='upper right', fontsize='small')
            
            print(f"{label} Inharmonicity Coeff (B): {B_fit:.2e}")

        else:
            ax_fit.text(0.5, 0.5, "Not enough peaks for fit", ha='center', va='center', transform=ax_fit.transAxes)
            ax_fit.set_title(f"{label} — Inharmonicity Fit")

        ax_fit.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_output:
            filename = os.path.join(prefix, f"{label}_pos_vel_fft.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            saved_files.append(filename)

        plt.close()
    return saved_files


def plot_axis_angle_over_time(quats, node_index, components="xyz", save_output=False, prefix="unknown"):
    axes = []
    angles = []

    for quat in quats:
        q = np.array(quat, dtype=float)
        q /= np.linalg.norm(q)
        rot = R.from_quat(q)  
        rotvec = rot.as_rotvec()
        angle = np.linalg.norm(rotvec)

        if angle > 1e-8:
            axis = rotvec / angle
        else:
            axis = np.array([0.0, 0.0, 0.0])

        axes.append(axis)
        angles.append(angle)

    axes = np.array(axes)
    angles = np.array(angles)
    t = np.arange(len(quats))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(t, np.degrees(angles))
    ax1.set_ylabel("Angle (degrees)")
    ax1.set_title(f"Axis–Angle orientation over time (node {node_index})")
    ax1.grid(True)

    if 'x' in components: ax2.plot(t, axes[:, 0], label="Axis X")
    if 'y' in components: ax2.plot(t, axes[:, 1], label="Axis Y")
    if 'z' in components: ax2.plot(t, axes[:, 2], label="Axis Z")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Axis component")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    saved_file = None
    if save_output:
        filename = os.path.join(prefix, "angles.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        saved_file = filename
    plt.close()
    return saved_file


def plot_energies(trans_ke, rot_ke, bend_twist_pe, shear_stretch_pe, dl, show_totals=False, remove_offset=True, save_output=False, prefix="unknown"):
    trans_ke = np.array(trans_ke)
    rot_ke = np.array(rot_ke)
    bend_pe = np.array(bend_twist_pe) * dl
    shear_pe = np.array(shear_stretch_pe) * dl

    if remove_offset:
        sos = butter(4, 0.01, btype='high', output='sos')
        trans_ke = sosfiltfilt(sos, trans_ke)
        rot_ke = sosfiltfilt(sos, rot_ke)
        bend_pe = sosfiltfilt(sos, bend_pe)
        shear_pe = sosfiltfilt(sos, shear_pe)

    total_kin = trans_ke + rot_ke
    total_pot = bend_pe + shear_pe
    total_energy = total_kin + total_pot

    plt.figure(figsize=(12, 8))
    plt.plot(trans_ke, label="Translational KE", alpha=0.4, linestyle=':')
    plt.plot(rot_ke, label="Rotational KE", alpha=0.4, linestyle=':')
    plt.plot(bend_pe, label="Potential: Bend/Twist", alpha=0.4, linestyle=':')
    plt.plot(shear_pe, label="Potential: Shear/Stretch", alpha=0.4, linestyle=':')
    
    if show_totals:
        plt.plot(total_kin, label="TOTAL Kinetic Energy", linewidth=2, color='blue')
        plt.plot(total_pot, label="TOTAL Potential Energy", linewidth=2, color='orange')
        if not remove_offset: 
            plt.plot(total_energy, label="TOTAL SYSTEM ENERGY", color='black', linestyle='--', linewidth=1.5)
    
    plt.title("Cosserat Rod Energy Breakdown")
    plt.ylabel("Energy (Joules)")
    plt.xlabel("Frame Index")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    saved_file = None
    if save_output:        
        filename = os.path.join(prefix, "energy.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        saved_file = filename
        
    plt.close()
    print(f"Total Energy Mean: {np.mean(total_energy):.6e} J")
    print(f"Energy Variation: {np.std(total_energy):.6e} J")
    return saved_file


def process_and_email(sim_dir, run_id, config, email_config, m_node, inertia, K_se, K_bt, dl):
    sim_path = Path(sim_dir)
    print(f"[{datetime.now()}] Streaming chunks to extract lightweight timeseries...")
    
    node_files = sorted(sim_path.glob("n_*.npy"))
    edge_files = sorted(sim_path.glob("e_*.npy"))
    
    node_idx = config.get("inspect_node", 10)
    
    pos_all, vel_all, mom_all, quat_all = [], [], [], []
    t_ke_all, r_ke_all, bt_pe_all, ss_pe_all = [], [], [], []
    
    for nf, ef in zip(node_files, edge_files):
        n_chunk = np.load(nf, allow_pickle=True)
        e_chunk = np.load(ef, allow_pickle=True)
        
        for i in range(len(n_chunk)):
            n_frame = n_chunk[i]
            e_frame = e_chunk[i]
            
            p, v, m, _ = n_frame[node_idx]
            pos_all.append(p); vel_all.append(v); mom_all.append(m)
            
            quat_all.append(e_frame[node_idx][0])
            
            t_sum, bt_sum = 0, 0
            for j, node in enumerate(n_frame):
                weight = 0.5 if (j == 0 or j == len(n_frame)-1) else 1.0
                t_sum += E_T(node, m_node) * weight
                bt_sum += E_PB(node[2], K_bt) * weight
                
            r_sum, ss_sum = 0, 0
            for edge in e_frame:
                r_sum += E_R(edge[1][0], inertia)
                ss_sum += E_PS(edge[1][2], K_se)
                
            t_ke_all.append(t_sum); bt_pe_all.append(bt_sum)
            r_ke_all.append(r_sum); ss_pe_all.append(ss_sum)

    pos_all = np.array(pos_all, dtype=np.float32)
    vel_all = np.array(vel_all, dtype=np.float32)
    mom_all = np.array(mom_all, dtype=np.float32)
    quat_all = np.array(quat_all, dtype=np.float32)
    
    full_dir = sim_path / "full_sim"
    free_dir = sim_path / "free_vib"
    full_dir.mkdir(exist_ok=True)
    free_dir.mkdir(exist_ok=True)
    
    hammer_limit = 2 * config["chunk_size"]

    with plot_lock:
        print(f"[{datetime.now()}] Generating Full Simulation Plots...")
        f1 = plot_node_pos_vel_moment_fft(pos_all, vel_all, mom_all, config["dt"], config["oversampling_factor"], save_output=True, prefix=str(full_dir))
        f2 = plot_axis_angle_over_time(quat_all, node_idx, save_output=True, prefix=str(full_dir))
        f3 = plot_energies(t_ke_all, r_ke_all, bt_pe_all, ss_pe_all, dl, save_output=True, prefix=str(full_dir))
        f4 = plot_energies(t_ke_all, r_ke_all, bt_pe_all, ss_pe_all, dl, save_output=True, show_totals=True, prefix=str(full_dir))
        full_files = f1 + ([f2] if f2 else []) + ([f3] if f3 else []) + ([f4] if f4 else [])

        print(f"[{datetime.now()}] Generating Free Vibration Plots...")
        if len(pos_all) > hammer_limit:
            p1 = plot_node_pos_vel_moment_fft(pos_all[hammer_limit:], vel_all[hammer_limit:], mom_all[hammer_limit:], config["dt"], config["oversampling_factor"], save_output=True, prefix=str(free_dir))
            p2 = plot_axis_angle_over_time(quat_all[hammer_limit:], node_idx, save_output=True, prefix=str(free_dir))
            p3 = plot_energies(t_ke_all[hammer_limit:], r_ke_all[hammer_limit:], bt_pe_all[hammer_limit:], ss_pe_all[hammer_limit:], dl, save_output=True, prefix=str(free_dir))
            p4 = plot_energies(t_ke_all[hammer_limit:], r_ke_all[hammer_limit:], bt_pe_all[hammer_limit:], ss_pe_all[hammer_limit:], dl, save_output=True, show_totals=True, prefix=str(free_dir))
            free_files = p1 + ([p2] if p2 else []) + ([p3] if p3 else []) + ([p4] if p4 else [])
        else:
            free_files = []

        s1 = plot_node_pos_vel_moment_fft(pos_all[:hammer_limit], vel_all[:hammer_limit], mom_all[:hammer_limit], config["dt"], config["oversampling_factor"], save_output=True, prefix=str(free_dir))
        s2 = plot_axis_angle_over_time(quat_all[:hammer_limit], node_idx, save_output=True, prefix=str(free_dir))
        s3 = plot_energies(t_ke_all[:hammer_limit], r_ke_all[:hammer_limit], bt_pe_all[:hammer_limit], ss_pe_all[:hammer_limit], dl, save_output=True, prefix=str(free_dir))
        s4 = plot_energies(t_ke_all[:hammer_limit], r_ke_all[:hammer_limit], bt_pe_all[:hammer_limit], ss_pe_all[:hammer_limit], dl, save_output=True, show_totals=True, prefix=str(free_dir))
        short_files = s1 + ([s2] if s2 else []) + ([s3] if s3 else []) + ([s4] if s4 else [])


    if email_config.get('sender_email') and email_config.get('sender_password'):
        print(f"[{datetime.now()}] Sending email...")
        msg = EmailMessage()
        msg['Subject'] = f"Piano Sim Run {run_id:03d} Completed"
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['receiver_email']
        msg.set_content(f"Simulation {run_id:03d} finished processing.\nPlots are attached.")

        for file_path in (full_files + free_files + short_files):
            path = Path(file_path)
            if path.exists():
                with open(path, 'rb') as f:
                    msg.add_attachment(f.read(), maintype='image', subtype='png', filename=f"{path.parent.name}_{path.name}")

        try:
            with smtplib.SMTP_SSL(email_config['smtp_server'], email_config['smtp_port']) as smtp:
                smtp.login(email_config['sender_email'], email_config['sender_password'])
                smtp.send_message(msg)
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")
