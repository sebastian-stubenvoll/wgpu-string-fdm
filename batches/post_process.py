import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.signal import find_peaks, butter, sosfiltfilt, spectrogram
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R
import smtplib
from email.message import EmailMessage
import os
import pickle
import gzip
import concurrent.futures
from itertools import chain

# ---------------------------------------------------------
# Helper Functions (Must be at top-level for multiprocessing)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Parallel Worker Function for File Extraction
# ---------------------------------------------------------
def _process_single_chunk(args):
    """Worker function to read a single chunk of data and extract timeseries."""
    nf, ef, node_idx, m_node, inertia, K_se, K_bt = args
    
    # Updated accumulators based on new Rust structures
    pos_chunk, vel_chunk, mom_chunk, curv_chunk = [], [], [], []
    quat_chunk, ang_vel_chunk, force_chunk, strain_chunk = [], [], [], []
    t_ke_chunk, r_ke_chunk, bt_pe_chunk, ss_pe_chunk = [], [], [], []

    with gzip.open(nf, "rb") as f:
        n_chunk = pickle.load(f)
    with gzip.open(ef, "rb") as f:
        e_chunk = pickle.load(f)
        
    for i in range(len(n_chunk)):
        n_frame = n_chunk[i]
        e_frame = e_chunk[i]
        
        # --- NEW NODE EXTRACTION ---
        # Rust: [position, velocity, internal_moment, curvature]
        p, v, m, c = n_frame[node_idx]
        pos_chunk.append(p)
        vel_chunk.append(v)
        mom_chunk.append(m)
        curv_chunk.append(c)
        
        # --- NEW EDGE EXTRACTION ---
        # Rust: (orientation, [angular_velocity, internal_force, strain])
        quat, edge_data = e_frame[node_idx]
        w, f_int, s = edge_data
        quat_chunk.append(quat)
        ang_vel_chunk.append(w)
        force_chunk.append(f_int)
        strain_chunk.append(s)
        
        # --- ENERGY CALCULATION ---
        t_sum, bt_sum = 0, 0
        for j, node in enumerate(n_frame):
            weight = 0.5 if (j == 0 or j == len(n_frame)-1) else 1.0
            t_sum += E_T(node, m_node) * weight
            # Curvature is now index 3 in the node array
            bt_sum += E_PB(node[3], K_bt) * weight
            
        r_sum, ss_sum = 0, 0
        for edge in e_frame:
            edge_quat, edge_data_vals = edge
            edge_w, edge_f, edge_s = edge_data_vals
            r_sum += E_R(edge_w, inertia)
            ss_sum += E_PS(edge_s, K_se)
            
        t_ke_chunk.append(t_sum)
        bt_pe_chunk.append(bt_sum)
        r_ke_chunk.append(r_sum)
        ss_pe_chunk.append(ss_sum)

    return (pos_chunk, vel_chunk, mom_chunk, curv_chunk, 
            quat_chunk, ang_vel_chunk, force_chunk, strain_chunk,
            t_ke_chunk, r_ke_chunk, bt_pe_chunk, ss_pe_chunk)


# ---------------------------------------------------------
# Plotting Functions 
# ---------------------------------------------------------
def plot_structural_dynamics(curvature, internal_force, strain, dt, oversampling_factor, save_output=False, prefix="unknown"):
    """Plots Curvature, Internal Force, and Strain over time for X, Y, Z components."""
    T = len(curvature)
    time = np.arange(T) * dt * oversampling_factor
    
    datasets = [
        ("Curvature", curvature, "tab:purple"),
        ("Internal Force", internal_force, "tab:red"),
        ("Strain", strain, "tab:brown")
    ]
    
    saved_files = []
    
    for name, data, color in datasets:
        data = np.array(data)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        components = ['X', 'Y', 'Z']
        
        for i in range(3):
            ax = axes[i]
            comp_data = data[:, i]
            ax.plot(time, comp_data, color=color, alpha=0.8)
            ax.set_ylabel(f"{name} {components[i]}")
            ax.grid(True, alpha=0.3)
            
            # Auto-scale Y to ignore massive initial transient spikes if necessary
            y_min, y_max = np.percentile(comp_data, [1, 99])
            padding = (y_max - y_min) * 0.1
            if y_max - y_min > 1e-10:
                ax.set_ylim(y_min - padding, y_max + padding)

        axes[0].set_title(f"Node {name} Over Time")
        axes[2].set_xlabel("Time [s]")
        plt.tight_layout()
        
        if save_output:
            filename = os.path.join(prefix, f"{name.lower().replace(' ', '_')}.png")
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            saved_files.append(filename)
            
        plt.close(fig)
        
    return saved_files

def plot_node_pos_vel_moment_fft(pos, vel, mom, dt, oversampling_factor, cutoff=4000, 
                                 moments=False, save_output=False, windowing=False, 
                                 fundamental_weight=0.2, prefix="unknown", plot_velocity=False):
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

    saved_files = []

    for label, p_data, v_data, m_data in components:
        n_plots = 3 + int(moments)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.8 * n_plots), sharex=False)

        ax = axes[0]
        p_max = np.max(np.abs(p_data))
        p_scaled = p_data / p_max if p_max > 0 else p_data
        
        ax.plot(time, p_scaled, color="tab:blue")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"{label} disp", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([f"{-p_max:.3e}", "0", f"{p_max:.3e}"])
        ax.set_title(f"{label} — displacement (time)")
        ax.grid(True, alpha=0.3)

        if plot_velocity:
            ax.set_title(f"{label} — displacement & velocity (time)")
            v_max = np.max(np.abs(v_data))
            v_scaled = v_data / v_max if v_max > 0 else v_data
            ax2 = ax.twinx()
            ax2.plot(time, v_scaled, "--", color="tab:orange")
            ax2.set_ylim(-1.05, 1.05)
            ax2.set_ylabel(f"{label} vel", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels([f"{-v_max:.3e}", "0", f"{v_max:.3e}"])

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
        
        peak_indices, _ = find_peaks(mag_norm, height=0.08, distance=5)
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
                if consecutive_misses > 2: break
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
            
            # --- IMPROVED FFT FIT & DYNAMIC WIDTH ---
            if len(ns) >= 4:
                try:
                    # Enforce strict physical bounds: B MUST be >= 0.
                    lower_bounds = [f1_est * 0.8, 0.0] 
                    upper_bounds = [f1_est * 1.2, 0.05] 
                    
                    popt, pcov = curve_fit(
                        lambda n, f1, B: n * f1 * np.sqrt(1 + B * n**2), 
                        ns, fns, 
                        p0=[f1_est, 1e-4],
                        bounds=(lower_bounds, upper_bounds)
                    )
                    f1_fit, B_fit = popt
                except RuntimeError: 
                    pass
            
            ax_fit.plot(freqs, mag_norm, color='lightgray', label='Spectrum')
            ax_fit.plot(peak_freqs, peak_mags, 'x', color='tab:blue', label='Detected Peaks')

            fitted_freqs = ns * f1_fit * np.sqrt(1 + B_fit * ns**2)
            if len(fitted_freqs) > 0:
                ax_fit.scatter(fitted_freqs, np.interp(fitted_freqs, freqs, mag_norm), 
                               color='tab:red', zorder=5, label=f'Fit (B={B_fit:.2e})')

            ideal_freqs = ns * f1_fit
            if len(ideal_freqs) > 0:
                ax_fit.axvline(ideal_freqs[0], color='green', linestyle=':', alpha=0.5, label=f'Harmonics (f1={f1_fit:.2f} Hz)')
                for f_ideal in ideal_freqs[1:]:
                    ax_fit.axvline(f_ideal, color='green', linestyle=':', alpha=0.5)

            # --- DYNAMIC WIDTH CALCULATION ---
            significant_peaks = peak_freqs[peak_mags > 0.01]
            max_sig_freq = np.max(significant_peaks) if len(significant_peaks) > 0 else f1_fit * 10
            dynamic_cutoff = min(cutoff, max_sig_freq * 1.15)
            
            axf.set_xlim(0, dynamic_cutoff) 
            ax_fit.set_xlim(0, dynamic_cutoff)
                  
            ax_fit.set_title(f"{label} — Inharmonicity Fit")
            ax_fit.set_xlabel("Frequency [Hz]")
            ax_fit.legend(loc='upper right', fontsize='small')
            print(f"{label} Inharmonicity (B): {B_fit:.2e} | f1: {f1_fit:.2f} Hz")

        else:
            ax_fit.text(0.5, 0.5, "Not enough peaks for fit", ha='center', va='center', transform=ax_fit.transAxes)
            ax_fit.set_title(f"{label} — Inharmonicity Fit")

        ax_fit.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_output:
            filename = os.path.join(prefix, f"{label}_pos_vel_fft.png")
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            saved_files.append(filename)

        plt.close(fig)
    return saved_files


def plot_axis_angle_over_time(quats, node_index, components="xyz", save_output=False, prefix="unknown"):
    axes, angles = [], []

    for quat in quats:
        q = np.array(quat, dtype=float)
        q /= np.linalg.norm(q)
        rot = R.from_quat(q)  
        rotvec = rot.as_rotvec()
        angle = np.linalg.norm(rotvec)

        if angle > 1e-8: axis = rotvec / angle
        else: axis = np.array([0.0, 0.0, 0.0])

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
    saved_file = []
    if save_output:
        filename = os.path.join(prefix, "angles.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        saved_file = [filename]
    plt.close(fig)
    return saved_file

def plot_euler_unwrapped(quats, node_index, save_output=False, prefix="unknown"):
    """Plots unwrapped Euler angles split into subplots for clarity."""
    q_array = np.array(quats, dtype=float)
    norms = np.linalg.norm(q_array, axis=1, keepdims=True)
    q_array = np.divide(q_array, norms, out=np.zeros_like(q_array), where=norms!=0)
    
    rot = R.from_quat(q_array)
    euler_angles = rot.as_euler('xyz', degrees=False) 
    
    euler_unwrapped = np.unwrap(euler_angles, axis=0)
    euler_deg = np.degrees(euler_unwrapped)
    
    t = np.arange(len(quats))
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    colors = ["tab:red", "tab:green", "tab:blue"]
    
    for i in range(3):
        axes[i].plot(t, euler_deg[:, i], color=colors[i], label=labels[i])
        axes[i].set_ylabel("Degrees")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right")
    
    axes[0].set_title(f"Unwrapped Euler Angles Over Time (Node {node_index})")
    axes[2].set_xlabel("Frame Index")
    
    plt.tight_layout()
    saved_file = []
    if save_output:
        filename = os.path.join(prefix, "euler_angles_unwrapped.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        saved_file = [filename]
    plt.close(fig)
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

    fig = plt.figure(figsize=(12, 8))
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
    
    saved_file = []
    if save_output:        
        suffix = "_totals" if show_totals else ""
        filename = os.path.join(prefix, f"energy{suffix}.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        saved_file = [filename]
        
    plt.close(fig)
    return saved_file

def plot_energy_partition(trans_ke, rot_ke, bend_pe, shear_pe, dl, save_output=False, prefix="unknown"):
    """Plots Normalized Stacked Area & Gradient (Energy Transfer Rate) over time."""
    t_ke = np.array(trans_ke)
    r_ke = np.array(rot_ke)
    b_pe = np.array(bend_pe) * dl
    s_pe = np.array(shear_pe) * dl

    sos = butter(2, 0.05, btype='low', output='sos')
    t_ke_s = np.abs(sosfiltfilt(sos, t_ke))
    r_ke_s = np.abs(sosfiltfilt(sos, r_ke))
    b_pe_s = np.abs(sosfiltfilt(sos, b_pe))
    s_pe_s = np.abs(sosfiltfilt(sos, s_pe))
    
    total_e_s = t_ke_s + r_ke_s + b_pe_s + s_pe_s
    total_e_s[total_e_s == 0] = 1e-10

    t_norm = (t_ke_s / total_e_s) * 100
    r_norm = (r_ke_s / total_e_s) * 100
    b_norm = (b_pe_s / total_e_s) * 100
    s_norm = (s_pe_s / total_e_s) * 100

    frames = np.arange(len(t_ke))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    axes[0].stackplot(frames, t_norm, r_norm, b_norm, s_norm, 
                      labels=['Trans KE', 'Rot KE', 'Bend/Twist PE', 'Shear/Stretch PE'],
                      colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'], alpha=0.8)
    axes[0].set_title("Normalized Energy Partition (Mode Transfer)")
    axes[0].set_ylabel("Percentage of Total Energy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    
    t_grad = np.gradient(t_ke_s)
    r_grad = np.gradient(r_ke_s)
    b_grad = np.gradient(b_pe_s)
    s_grad = np.gradient(s_pe_s)

    axes[1].plot(frames, t_grad, label=r'$\Delta$ Trans KE', color='tab:blue', alpha=0.7)
    axes[1].plot(frames, r_grad, label=r'$\Delta$ Rot KE', color='tab:orange', alpha=0.9, linewidth=1.5)
    axes[1].plot(frames, b_grad, label=r'$\Delta$ Bend/Twist PE', color='tab:green', alpha=0.7)
    axes[1].plot(frames, s_grad, label=r'$\Delta$ Shear/Stretch PE', color='tab:red', alpha=0.7)
    
    axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[1].set_title("Energy Gradients (Power flow in/out of modes)")
    axes[1].set_ylabel("dE/dt (Smoothed)")
    axes[1].set_xlabel("Frame Index")
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    
    saved_file = []
    if save_output:
        filename = os.path.join(prefix, "energy_partition_transfer.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        saved_file = [filename]
    plt.close(fig)
    return saved_file

def plot_spectrogram(pos, dt, oversampling_factor, save_output=False, prefix="unknown"):
    """Generates a Waterfall/Spectrogram plot for node displacement."""
    fs = 1.0 / (dt * oversampling_factor)
    saved_files = []
    
    components = [("x", pos[:, 0]), ("y", pos[:, 1]), ("z", pos[:, 2])]
    
    for label, data in components:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        nperseg = min(256, len(data) // 4)
        if nperseg < 16:
            plt.close(fig)
            continue
            
        f, t, Sxx = spectrogram(data - np.mean(data), fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        
        pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='inferno')
        fig.colorbar(pcm, ax=ax, label='Power/Frequency (dB/Hz)')
        
        ax.set_title(f"Waterfall/Spectrogram: {label.upper()}-Displacement")
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_ylim(0, min(4000, fs/2)) 
        
        plt.tight_layout()
        if save_output:
            filename = os.path.join(prefix, f"{label}_spectrogram.png")
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            saved_files.append(filename)
        plt.close(fig)
        
    return saved_files

def plot_phase_space(pos, vel, save_output=False, prefix="unknown"):
    """Phase space portrait (Displacement vs Velocity) for X, Y, Z."""
    saved_files = []
    components = [("x", pos[:, 0], vel[:, 0]), ("y", pos[:, 1], vel[:, 1]), ("z", pos[:, 2], vel[:, 2])]
    
    for label, p_data, v_data in components:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        time = np.linspace(0, 1, len(p_data))
        scatter = ax.scatter(p_data, v_data, c=time, cmap='viridis', s=1, alpha=0.5)
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Normalized Time (0 = Start, 1 = End)')
        
        ax.set_title(f"Phase Space Portrait ({label.upper()})")
        ax.set_xlabel("Displacement")
        ax.set_ylabel("Velocity")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_output:
            filename = os.path.join(prefix, f"{label}_phase_space.png")
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            saved_files.append(filename)
        plt.close(fig)
        
    return saved_files

def plot_3d_orbit(pos, node_index, save_output=False, prefix="unknown"):
    """Plots the 3D trajectory of the node over time."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    time = np.linspace(0, 1, len(pos))
    p = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=time, cmap='plasma', s=2, alpha=0.6)
    
    ax.set_title(f"3D Node Trajectory (Orbit) - Node {node_index}")
    ax.set_xlabel("X Displacement")
    ax.set_ylabel("Y Displacement")
    ax.set_zlabel("Z Displacement")
    
    cbar = fig.colorbar(p, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Normalized Time')
    
    plt.tight_layout()
    saved_file = []
    if save_output:
        filename = os.path.join(prefix, "3d_trajectory_orbit.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        saved_file = [filename]
    plt.close(fig)
    return saved_file

# ---------------------------------------------------------
# Main Processor
# ---------------------------------------------------------
def process_and_email(sim_dir, run_id, config, email_config, m_node, inertia, K_se, K_bt, dl):
    sim_path = Path(sim_dir)
    print(f"[{datetime.now()}] Reading files and extracting timeseries across CPU cores...")
    
    node_files = sorted(sim_path.glob("n_*.pkl.gz"))
    edge_files = sorted(sim_path.glob("e_*.pkl.gz"))
    node_idx = config.get("inspect_node", 10)
    
    # 1. PARALLEL DATA EXTRACTION
    chunk_args = [
        (nf, ef, node_idx, m_node, inertia, K_se, K_bt)
        for nf, ef in zip(node_files, edge_files)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(_process_single_chunk, chunk_args))

    # Flatten results from all processes
    pos_all = list(chain.from_iterable(r[0] for r in results))
    vel_all = list(chain.from_iterable(r[1] for r in results))
    mom_all = list(chain.from_iterable(r[2] for r in results))
    curv_all = list(chain.from_iterable(r[3] for r in results))
    
    quat_all = list(chain.from_iterable(r[4] for r in results))
    ang_vel_all = list(chain.from_iterable(r[5] for r in results))
    force_all = list(chain.from_iterable(r[6] for r in results))
    strain_all = list(chain.from_iterable(r[7] for r in results))
    
    t_ke_all = list(chain.from_iterable(r[8] for r in results))
    r_ke_all = list(chain.from_iterable(r[9] for r in results))
    bt_pe_all = list(chain.from_iterable(r[10] for r in results))
    ss_pe_all = list(chain.from_iterable(r[11] for r in results))

    pos_all = np.array(pos_all, dtype=np.float32)
    vel_all = np.array(vel_all, dtype=np.float32)
    mom_all = np.array(mom_all, dtype=np.float32)
    curv_all = np.array(curv_all, dtype=np.float32)
    
    quat_all = np.array(quat_all, dtype=np.float32)
    ang_vel_all = np.array(ang_vel_all, dtype=np.float32)
    force_all = np.array(force_all, dtype=np.float32)
    strain_all = np.array(strain_all, dtype=np.float32)
    
    # Apply PE Offset: Subtract first frame PE from all frames
    ss_pe_all = np.array(ss_pe_all, dtype=np.float32)
    ss_pe_all -= ss_pe_all[0]
    
    full_dir = sim_path / "full_sim"
    free_dir = sim_path / "free_vib"
    full_dir.mkdir(exist_ok=True)
    free_dir.mkdir(exist_ok=True)
    short_dir = sim_path / "short_vib"
    short_dir.mkdir(exist_ok=True)
    
    hammer_limit = 5 * config["chunk_size"]

    # 2. PARALLEL PLOTTING
    print(f"[{datetime.now()}] Dispatching plots to multiprocessing pool...")
    dt = config["dt"]
    oversamp = config["oversampling_factor"]
    
    plot_tasks = []

    # --- Full Simulation Tasks ---
    plot_tasks.append((plot_node_pos_vel_moment_fft, (pos_all, vel_all, mom_all, dt, oversamp, 4000, False, True, False, 0.2, str(full_dir), False)))
    plot_tasks.append((plot_structural_dynamics, (curv_all, force_all, strain_all, dt, oversamp, True, str(full_dir))))
    plot_tasks.append((plot_axis_angle_over_time, (quat_all, node_idx, "xyz", True, str(full_dir))))
    plot_tasks.append((plot_euler_unwrapped, (quat_all, node_idx, True, str(full_dir))))
    plot_tasks.append((plot_energies, (t_ke_all, r_ke_all, bt_pe_all, ss_pe_all, dl, False, True, True, str(full_dir))))
    plot_tasks.append((plot_energies, (t_ke_all, r_ke_all, bt_pe_all, ss_pe_all, dl, True, True, True, str(full_dir))))
    plot_tasks.append((plot_energy_partition, (t_ke_all, r_ke_all, bt_pe_all, ss_pe_all, dl, True, str(full_dir))))
    plot_tasks.append((plot_spectrogram, (pos_all, dt, oversamp, True, str(full_dir))))
    plot_tasks.append((plot_3d_orbit, (pos_all, node_idx, True, str(full_dir))))

    # --- Free Vibration Tasks ---
    if len(pos_all) > hammer_limit:
        plot_tasks.append((plot_node_pos_vel_moment_fft, (pos_all[hammer_limit:], vel_all[hammer_limit:], mom_all[hammer_limit:], dt, oversamp, 4000, False, True, False, 0.2, str(free_dir), False)))
        plot_tasks.append((plot_structural_dynamics, (curv_all[hammer_limit:], force_all[hammer_limit:], strain_all[hammer_limit:], dt, oversamp, True, str(free_dir))))
        plot_tasks.append((plot_axis_angle_over_time, (quat_all[hammer_limit:], node_idx, "xyz", True, str(free_dir))))
        plot_tasks.append((plot_euler_unwrapped, (quat_all[hammer_limit:], node_idx, True, str(free_dir))))
        plot_tasks.append((plot_energies, (t_ke_all[hammer_limit:], r_ke_all[hammer_limit:], bt_pe_all[hammer_limit:], ss_pe_all[hammer_limit:], dl, False, True, True, str(free_dir))))
        plot_tasks.append((plot_energies, (t_ke_all[hammer_limit:], r_ke_all[hammer_limit:], bt_pe_all[hammer_limit:], ss_pe_all[hammer_limit:], dl, True, True, True, str(free_dir))))
        plot_tasks.append((plot_energy_partition, (t_ke_all[hammer_limit:], r_ke_all[hammer_limit:], bt_pe_all[hammer_limit:], ss_pe_all[hammer_limit:], dl, True, str(free_dir))))
        plot_tasks.append((plot_spectrogram, (pos_all[hammer_limit:], dt, oversamp, True, str(free_dir))))
        plot_tasks.append((plot_3d_orbit, (pos_all[hammer_limit:], node_idx, True, str(free_dir))))

    # --- Short Files Tasks ---
    plot_tasks.append((plot_node_pos_vel_moment_fft, (pos_all[:hammer_limit], vel_all[:hammer_limit], mom_all[:hammer_limit], dt, oversamp, 4000, False, True, False, 0.2, str(short_dir), True)))
    plot_tasks.append((plot_structural_dynamics, (curv_all[:hammer_limit], force_all[:hammer_limit], strain_all[:hammer_limit], dt, oversamp, True, str(short_dir))))
    plot_tasks.append((plot_axis_angle_over_time, (quat_all[:hammer_limit], node_idx, "xyz", True, str(short_dir))))
    plot_tasks.append((plot_euler_unwrapped, (quat_all[:hammer_limit], node_idx, True, str(short_dir))))
    plot_tasks.append((plot_energies, (t_ke_all[:hammer_limit], r_ke_all[:hammer_limit], bt_pe_all[:hammer_limit], ss_pe_all[:hammer_limit], dl, False, True, True, str(short_dir))))
    plot_tasks.append((plot_energies, (t_ke_all[:hammer_limit], r_ke_all[:hammer_limit], bt_pe_all[:hammer_limit], ss_pe_all[:hammer_limit], dl, True, True, True, str(short_dir))))
    plot_tasks.append((plot_energy_partition, (t_ke_all[:hammer_limit], r_ke_all[:hammer_limit], bt_pe_all[:hammer_limit], ss_pe_all[:hammer_limit], dl, True, str(short_dir))))
    plot_tasks.append((plot_spectrogram, (pos_all[:hammer_limit], dt, oversamp, True, str(short_dir))))
    plot_tasks.append((plot_phase_space, (pos_all[:hammer_limit], vel_all[:hammer_limit], True, str(short_dir))))
    plot_tasks.append((plot_3d_orbit, (pos_all[:hammer_limit], node_idx, True, str(short_dir))))

    generated_files = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(func, *args) for func, args in plot_tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result: generated_files.extend(result)
            except Exception as e:
                print(f"Plotting process failed: {e}")

    # 3. EMAIL DISPATCH
    if email_config.get('sender_email') and email_config.get('sender_password'):
        print(f"[{datetime.now()}] Sending email...")
        msg = EmailMessage()
        msg['Subject'] = f"Piano Sim Run {run_id:03d} Completed"
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['receiver_email']
        msg.set_content(f"Simulation {run_id:03d} finished processing.\nPlots are attached.")

        for file_path in generated_files:
            path = Path(file_path)
            if path.exists():
                with open(path, 'rb') as f:
                    msg.add_attachment(f.read(), maintype='image', subtype='png', filename=f"{path.parent.name}_{path.name}")

        try:
            with smtplib.SMTP_SSL(email_config['smtp_server'], email_config['smtp_port']) as smtp:
                smtp.login(email_config['sender_email'], email_config['sender_password'])
                smtp.send_message(msg)
            print(f"[{datetime.now()}] Email sent successfully!")
        except Exception as e:
            print(f"[{datetime.now()}] Failed to send email: {e}")
