import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.signal import find_peaks, butter, sosfiltfilt, spectrogram
from scipy.spatial.transform import Rotation as R
import smtplib
from email.message import EmailMessage
import os
import pickle
import gzip
import concurrent.futures

# ---------------------------------------------------------
# Helper Functions 
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
    """Worker function to read a single chunk of data and extract timeseries for ALL requested nodes."""
    nf, ef, inspect_nodes, m_node, inertia, K_se, K_bt = args
    
    node_data = {idx: {'pos': [], 'vel': [], 'mom': [], 'curv': [], 
                       'quat': [], 'ang_vel': [], 'force': [], 'strain': []} 
                 for idx in inspect_nodes}
                 
    t_ke_chunk, r_ke_chunk, bt_pe_chunk, ss_pe_chunk = [], [], [], []

    with gzip.open(nf, "rb") as f: n_chunk = pickle.load(f)
    with gzip.open(ef, "rb") as f: e_chunk = pickle.load(f)
        
    for i in range(len(n_chunk)):
        n_frame = n_chunk[i]
        e_frame = e_chunk[i]
        
        for idx in inspect_nodes:
            # FIXED INDEXING: Rust exports [pos, vel, curv, moment]
            p, v, c, m = n_frame[idx]
            node_data[idx]['pos'].append(p)
            node_data[idx]['vel'].append(v)
            node_data[idx]['curv'].append(c)
            node_data[idx]['mom'].append(m)
            
            # Rust: (orientation, [angular_velocity, internal_force, strain])
            quat, edge_data = e_frame[idx]
            w, f_int, s = edge_data
            node_data[idx]['quat'].append(quat)
            node_data[idx]['ang_vel'].append(w)
            node_data[idx]['force'].append(f_int)
            node_data[idx]['strain'].append(s)
        
        # --- GLOBAL ENERGY CALCULATION ---
        t_sum, bt_sum = 0, 0
        for j, node in enumerate(n_frame):
            weight = 0.5 if (j == 0 or j == len(n_frame)-1) else 1.0
            t_sum += E_T(node, m_node) * weight
            bt_sum += E_PB(node[2], K_bt) * weight # FIX: node[2] is curvature
            
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

    return (node_data, t_ke_chunk, r_ke_chunk, bt_pe_chunk, ss_pe_chunk)


# ---------------------------------------------------------
# Plotting Functions 
# ---------------------------------------------------------
def plot_energy(time, trans_ke, rot_ke, bend_pe, shear_pe, save_output=False, prefix="unknown"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    eps = 1e-12
    trans_ke = np.maximum(np.array(trans_ke), eps)
    rot_ke = np.maximum(np.array(rot_ke), eps)
    bend_pe = np.maximum(np.array(bend_pe), eps)
    shear_pe = np.maximum(np.array(shear_pe), eps)
    
    total = trans_ke + rot_ke + bend_pe + shear_pe
    
    ax.plot(time, trans_ke, label='Kinetic (Translational)', alpha=0.8)
    ax.plot(time, rot_ke, label='Kinetic (Rotational)', alpha=0.8)
    ax.plot(time, bend_pe, label='Potential (Bending)', alpha=0.8)
    ax.plot(time, shear_pe, label='Potential (Shear/Stretch)', alpha=0.8)
    ax.plot(time, total, label='Total Energy', color='black', linewidth=2, linestyle='--')
    
    ax.set_title("Global Energy Components over Time (Log Scale)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [J]")
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    saved_file = []
    if save_output:
        filename = os.path.join(prefix, "energy_global_log.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        saved_file = [filename]
    plt.close(fig)
    return saved_file

def plot_spectrogram_and_envelope(pos, dt, oversampling_factor, node_idx, save_output=False, prefix="unknown"):
    T = len(pos)
    time = np.arange(T) * dt * oversampling_factor
    disp_y = pos[:, 1]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Envelope Fit (Log Space Linearization)
    axes[0].plot(time, disp_y, color='tab:green', alpha=0.7, label='Displacement Y')
    y_abs = np.abs(disp_y)
    peaks, _ = find_peaks(y_abs, distance=max(1, len(disp_y)//100))
    if len(peaks) > 3:
        t_peaks = time[peaks]
        y_peaks = y_abs[peaks]
        valid = y_peaks > 1e-10
        if np.sum(valid) > 3:
            slope, intercept = np.polyfit(t_peaks[valid], np.log(y_peaks[valid]), 1)
            A = np.exp(intercept)
            gamma = -slope
            axes[0].plot(time, A * np.exp(-gamma * time), 'k--', label=f'Decay Fit: $\gamma$={gamma:.2f}')
    
    axes[0].set_ylabel("Displacement [m]")
    axes[0].set_title(f"Node {node_idx} Kinematics & Decay Envelope")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Spectrogram
    f, t_spec, Sxx = spectrogram(disp_y, fs=1.0/(dt * oversampling_factor), nperseg=256)
    cax = axes[1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylim(0, 5000)
    fig.colorbar(cax, ax=axes[1], label='Power [dB]')
    
    saved_file = []
    if save_output:
        filename = os.path.join(prefix, "kinematics_spectrogram.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        saved_file = [filename]
    plt.close(fig)
    return saved_file

def plot_structural_dynamics(curvature, internal_force, strain, dt, oversampling_factor, save_output=False, prefix="unknown"):
    T = len(curvature)
    time = np.arange(T) * dt * oversampling_factor
    datasets = [("Curvature", curvature, "tab:purple"), ("Internal Force", internal_force, "tab:red"), ("Strain", strain, "tab:brown")]
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
            y_min, y_max = np.percentile(comp_data, [1, 99])
            padding = (y_max - y_min) * 0.1
            if y_max - y_min > 1e-10: ax.set_ylim(y_min - padding, y_max + padding)

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
                                 fundamental_weight=0.2, prefix="unknown", plot_velocity=False, node_idx=""):
    T = len(pos)
    time = np.arange(T) * dt * oversampling_factor
    freqs = np.fft.rfftfreq(T, dt * oversampling_factor)
    components = [("x", pos[:, 0], vel[:, 0], mom[:, 0]), ("y", pos[:, 1], vel[:, 1], mom[:, 1]), ("z", pos[:, 2], vel[:, 2], mom[:, 2])]

    def fft_mag(signal):
        sig = signal - np.mean(signal)
        if windowing: sig = sig * np.hanning(len(sig))
        return np.abs(np.fft.rfft(sig))

    saved_files = []
    for label, p_data, v_data, m_data in components:
        n_plots = 3 + int(moments)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.8 * n_plots), sharex=False)

        ax = axes[0]
        p_max = np.max(np.abs(p_data))
        ax.plot(time, p_data / p_max if p_max > 0 else p_data, color="tab:blue")
        ax.set_title(f"{label} — displacement (Node {node_idx})")
        ax.grid(True, alpha=0.3)

        plot_idx = 1
        if moments:
            axm = axes[plot_idx]
            m_max = np.max(np.abs(m_data))
            axm.plot(time, m_data / m_max if m_max > 0 else m_data, color="tab:green")
            axm.set_title(f"{label} — moment")
            axm.grid(True, alpha=0.3)
            plot_idx += 1

        bins = fft_mag(p_data)
        axes[plot_idx].plot(freqs, bins, color="tab:blue")
        axes[plot_idx].set_title(f"{label} — displacement FFT")
        axes[plot_idx].set_xlim(0, cutoff)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        ax_fit = axes[plot_idx]
        mag_norm = bins / np.max(bins) if np.max(bins) > 0 else bins
        peak_indices, _ = find_peaks(mag_norm, height=0.08, distance=5)
        peak_freqs, peak_mags = freqs[peak_indices], mag_norm[peak_indices]

        if len(peak_freqs) >= 2:
            significant = np.where(peak_mags > fundamental_weight)[0]
            f1_est = peak_freqs[significant[0]] if len(significant) > 0 else peak_freqs[np.argmax(peak_mags)]

            measured_partials, harmonic_indices, consecutive_misses = [], [], 0
            for n in range(1, 20):
                if consecutive_misses > 2: break
                target = n * f1_est
                idx_closest = np.argmin(np.abs(peak_freqs - target))
                if np.abs(peak_freqs[idx_closest] - target) < (0.45 * f1_est):
                    measured_partials.append(peak_freqs[idx_closest])
                    harmonic_indices.append(n)
                    consecutive_misses = 0
                else: consecutive_misses += 1

            ns, fns = np.array(harmonic_indices), np.array(measured_partials)
            f1_fit, B_fit = f1_est, 0.0
            
            # --- ROBUST LINEARIZED INHARMONICITY FIT ---
            if len(ns) >= 4:
                x_vals = ns**2
                y_vals = (fns / ns)**2
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                
                if intercept > 0:
                    f1_fit = np.sqrt(intercept)
                    B_fit = slope / intercept
            
            ax_fit.plot(freqs, mag_norm, color='lightgray')
            ax_fit.plot(peak_freqs, peak_mags, 'x', color='tab:blue')
            fitted_freqs = ns * f1_fit * np.sqrt(1 + B_fit * ns**2)
            ax_fit.scatter(fitted_freqs, np.interp(fitted_freqs, freqs, mag_norm), color='tab:red', zorder=5, label=f'Fit (B={B_fit:.2e})')
            ax_fit.set_xlim(0, cutoff)
            ax_fit.legend(loc="upper right")
            ax_fit.set_title(f"Robust Inharmonicity Fit")

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
        q /= (np.linalg.norm(q) + 1e-12)
        rot = R.from_quat(q)
        rotvec = rot.as_rotvec()
        angle = np.linalg.norm(rotvec)
        axis = rotvec / angle if angle > 1e-8 else np.array([0.0, 0.0, 0.0])
        axes.append(axis)
        angles.append(angle)
        
    t = np.arange(len(quats))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(t, np.degrees(angles))
    ax1.set_title(f"Axis–Angle orientation over time (Node {node_index})")
    ax1.grid(True)
    
    axes = np.array(axes)
    if 'x' in components: ax2.plot(t, axes[:, 0], label="Axis X")
    if 'y' in components: ax2.plot(t, axes[:, 1], label="Axis Y")
    if 'z' in components: ax2.plot(t, axes[:, 2], label="Axis Z")
    ax2.legend()
    ax2.grid(True)
    
    saved_file = []
    if save_output:
        filename = os.path.join(prefix, "angles.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        saved_file = [filename]
    plt.close(fig)
    return saved_file

def plot_phase_space(pos, vel, save_output=False, prefix="unknown"):
    saved_files = []
    for label, p_data, v_data in [("x", pos[:, 0], vel[:, 0]), ("y", pos[:, 1], vel[:, 1]), ("z", pos[:, 2], vel[:, 2])]:
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(p_data, v_data, c=np.linspace(0, 1, len(p_data)), cmap='viridis', s=1, alpha=0.5)
        fig.colorbar(scatter, ax=ax).set_label('Normalized Time')
        ax.set_title(f"Phase Space Portrait ({label.upper()})")
        ax.grid(True, alpha=0.3)
        if save_output:
            filename = os.path.join(prefix, f"{label}_phase_space.png")
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            saved_files.append(filename)
        plt.close(fig)
    return saved_files

def plot_3d_orbit(pos, node_index, save_output=False, prefix="unknown"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=np.linspace(0, 1, len(pos)), cmap='plasma', s=2, alpha=0.6)
    ax.set_title(f"3D Node Trajectory - Node {node_index}")
    fig.colorbar(p, ax=ax, shrink=0.5, pad=0.1).set_label('Normalized Time')
    
    saved_file = []
    if save_output:
        filename = os.path.join(prefix, "3d_trajectory_orbit.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        saved_file = [filename]
    plt.close(fig)
    return saved_file

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def process_and_email(sim_path, run_id, config, email_config, m_node, dl, inertia, K_se, K_bt):
    sim_path = Path(sim_path)
    node_files = sorted(sim_path.glob("nodes_chunk_*.pkl.gz"))
    edge_files = sorted(sim_path.glob("edges_chunk_*.pkl.gz"))
    
    # Read nodes from array or fallback to single node
    inspect_nodes = config.get("inspect_nodes", [config.get("inspect_node", 25)])
    if not isinstance(inspect_nodes, list): inspect_nodes = [inspect_nodes]
        
    print(f"[{datetime.now()}] Processing {len(node_files)} chunks for nodes: {inspect_nodes}...")
    
    args_list = [(nf, ef, inspect_nodes, m_node, inertia, K_se, K_bt) for nf, ef in zip(node_files, edge_files)]
    t_ke_all, r_ke_all, bt_pe_all, ss_pe_all = [], [], [], []
    all_node_data = {idx: {'pos': [], 'vel': [], 'mom': [], 'curv': [], 'quat': [], 'ang_vel': [], 'force': [], 'strain': []} for idx in inspect_nodes}
                 
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_single_chunk, args_list))
        
    for res in results:
        node_data, t_ke, r_ke, bt_pe, ss_pe = res
        for idx in inspect_nodes:
            for key in all_node_data[idx]:
                all_node_data[idx][key].extend(node_data[idx][key])
            
        t_ke_all.extend(t_ke)
        r_ke_all.extend(r_ke)
        bt_pe_all.extend(bt_pe)
        ss_pe_all.extend(ss_pe)

    # Offset static strain energy
    ss_pe_all = np.array(ss_pe_all, dtype=np.float32)
    ss_pe_all -= ss_pe_all[0]
    
    dt = config["dt"]
    oversamp = config["oversampling_factor"]
    
    # 2. PARALLEL PLOTTING
    print(f"[{datetime.now()}] Dispatching plots to multiprocessing pool...")
    plot_tasks = []
    
    # Global Plot
    plot_tasks.append((plot_energy, (np.arange(len(t_ke_all)) * dt * oversamp, t_ke_all, r_ke_all, bt_pe_all, ss_pe_all, True, str(sim_path))))
    
    # Node Specific Plots
    for idx in inspect_nodes:
        node_dir = sim_path / f"node_{idx}"
        node_dir.mkdir(exist_ok=True)
        
        pos = np.array(all_node_data[idx]['pos'], dtype=np.float32)
        vel = np.array(all_node_data[idx]['vel'], dtype=np.float32)
        mom = np.array(all_node_data[idx]['mom'], dtype=np.float32)
        curv = np.array(all_node_data[idx]['curv'], dtype=np.float32)
        force = np.array(all_node_data[idx]['force'], dtype=np.float32)
        strain = np.array(all_node_data[idx]['strain'], dtype=np.float32)
        quat = np.array(all_node_data[idx]['quat'], dtype=np.float32)
        
        plot_tasks.append((plot_node_pos_vel_moment_fft, (pos, vel, mom, dt, oversamp, 4000, False, True, False, 0.2, str(node_dir), False, idx)))
        plot_tasks.append((plot_structural_dynamics, (curv, force, strain, dt, oversamp, True, str(node_dir))))
        plot_tasks.append((plot_axis_angle_over_time, (quat, idx, "xyz", True, str(node_dir))))
        plot_tasks.append((plot_phase_space, (pos, vel, True, str(node_dir))))
        plot_tasks.append((plot_3d_orbit, (pos, idx, True, str(node_dir))))
        plot_tasks.append((plot_spectrogram_and_envelope, (pos, dt, oversamp, idx, True, str(node_dir))))

    generated_files = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(func, *args) for func, args in plot_tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result: generated_files.extend(result)
            except Exception as e:
                print(f"Plotting failed: {e}")

    # 3. EMAIL DISPATCH
    if email_config.get('sender_email') and email_config.get('sender_password'):
        print(f"[{datetime.now()}] Sending email...")
        msg = EmailMessage()
        msg['Subject'] = f"Piano Sim Run {run_id:03d} Completed"
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['receiver_email']
        msg.set_content(f"Simulation {run_id:03d} finished processing.\nPlots attached.")

        for file_path in generated_files:
            path = Path(file_path)
            if path.exists():
                with open(path, 'rb') as f:
                    msg.add_attachment(f.read(), maintype='image', subtype='png', filename=f"{path.parent.name}_{path.name}")
        try:
            with smtplib.SMTP_SSL(email_config['smtp_server'], email_config['smtp_port']) as smtp:
                smtp.login(email_config['sender_email'], email_config['sender_password'])
                smtp.send_message(msg)
            print(f"[{datetime.now()}] Email sent successfully.")
        except Exception as e:
            print(f"[{datetime.now()}] Failed to send email: {e}")
