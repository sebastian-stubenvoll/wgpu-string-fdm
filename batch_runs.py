import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from py_wgpu_fdm import Simulation
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pickle



def generate_straight_rod(node_count, tuned_length, tension_force, core_radius, 
                          winding_radius, E_core, G_core, rho_core, E_winding, 
                          G_winding, rho_winding, packing_factor, mu, twists):
    """Generates the initial configuration for the discrete Cosserat rod."""
    core_area = np.pi * core_radius**2
    A = core_area
    EA = E_core * A
    dilatation = tension_force / EA
    rest_length = tuned_length / (1 + dilatation)
    dl = rest_length / float(node_count - 1)
    
    core_mass = float(rho_core * core_area * dl)
    winding_area = np.pi * ((core_radius + 2.0 * winding_radius)**2 - core_radius**2)
    winding_mass = float(rho_winding * winding_area * dl * packing_factor)
    mass = core_mass + winding_mass

    I_core = rho_core * core_radius**4 / 4.0 
    I_winding = rho_winding * ((core_radius + 2.0 * winding_radius)**4 - core_radius**4) * packing_factor / 4.0 
    I1 = I_core + I_winding
    I3 = I1 * 2.0
    inertia = (np.pi * dl * np.array([I1, I1, I3], dtype=np.float32))

    alpha = 4.0 / 3.0
    I_c = np.pi * core_radius**4 / 4.0
    J = np.pi * core_radius**4 / 2.0
    I_w = np.pi * ((core_radius + 2.0 * winding_radius)**4 - core_radius**4) * packing_factor / 4.0 
    EI = E_core * I_c + (E_winding * I_w * mu)
    
    K_se = np.array([G_core * A * alpha, G_core * A * alpha, E_core * A], dtype=np.float32)
    K_bt = np.array([EI, EI, G_core * J], dtype=np.float32)

    x = np.linspace(0.0, tuned_length, node_count)
    reference_positions = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1).astype(np.float32)

    # Twist logic
    def twist_quaternion(angle):
        half = 0.5 * angle
        return np.array([np.sin(half), 0.0, 0.0, np.cos(half)])
    
    def quat_mul(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, 
                         w1*z2 + x1*y2 - y1*x2 + z1*w2, w1*w2 - x1*x2 - y1*y2 - z1*z2])

    angle = np.pi / 2
    q_base = np.array([0.0, np.sin(0.5 * angle), 0.0, np.cos(0.5 * angle)])
    total_edges = node_count - 1
    total_angle = 2.0 * np.pi * twists
    orientations = [quat_mul(twist_quaternion(total_angle * (i / total_edges)), q_base) for i in range(total_edges)]
    orientations = [q / np.linalg.norm(q) for q in orientations]

    ref_vecs = np.array([reference_positions[i+1] - reference_positions[i] for i in range(node_count - 1)], dtype=np.float32)
    edges = [(o, v, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)) for o, v in zip(orientations, ref_vecs)]
    edges.append((edges[-1][0], np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)))
    nodes = [(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0])) for _ in reference_positions]
    
    return reference_positions, nodes, edges, dl, mass, inertia, K_se, K_bt


def make_weights(node_count, node_index=None, width_count=4):
  
    data = np.zeros((node_count, 4), dtype=np.float32)
    x = np.arange(node_count, dtype=np.float32)
    
    sigma = width_count / 2.355
    if sigma == 0:
        sigma = 1e-5
        
    gauss_vals = np.exp(-((x - node_index)**2) / (2 * sigma**2))
    
    mask = np.abs(x - node_index) <= width_count
    data[:, 2] = gauss_vals * mask
    
    return data


def plot_node_pos_vel_moment_fft(
    simdir,
    nodes_history,
    dt,
    oversampling_factor,
    cutoff=20_000,
    node_index=None,
    moments=False,
    save_output=False,
    windowing=False,
    fundamental_weight=0.2,
    prefix="unknown",
):
    T = len(nodes_history)
    N = len(nodes_history[0])

    if node_index is None:
        node_index = N // 2

    time = np.arange(T) * dt * oversampling_factor
    freqs = np.fft.rfftfreq(T, dt * oversampling_factor)

    pos = np.zeros((T, 3), dtype=np.float32)
    vel = np.zeros((T, 3), dtype=np.float32)
    mom = np.zeros((T, 3), dtype=np.float32)

    for t in range(T):
        p, v, m, _ = nodes_history[t][node_index]
        pos[t] = p
        vel[t] = v
        mom[t] = m

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
        
        # Use a relative height threshold to avoid fitting noise floor
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
                
                # Check tolerance
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
            ax_fit.text(0.5, 0.5, "Not enough peaks for fit", 
                        ha='center', va='center', transform=ax_fit.transAxes)
            ax_fit.set_title(f"{label} — Inharmonicity Fit")

        ax_fit.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_output:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = simdir / f"{timestamp}_{prefix}-{label}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")


def make_force_curve(dt, duration):
    scale = duration / 4.0
    t = np.arange(0, duration, dt)
    
    mu1 = 0.50 * scale
    amp1 = 15.0
    sig1_left = 0.1 * scale
    sig1_right = 0.75 * scale
    
    mu2 = 3 * scale
    amp2 = 8.5
    sig2 = 0.2 * scale
    
    force = np.zeros_like(t)
    
    mask_left = t <= mu1
    force[mask_left] += amp1 * np.exp(-0.5 * ((t[mask_left] - mu1) / sig1_left)**2)
    force[~mask_left] += amp1 * np.exp(-0.5 * ((t[~mask_left] - mu1) / sig1_right)**2)
    
    force += amp2 * np.exp(-0.5 * ((t - mu2) / sig2)**2)
    
    return t, force


def run_batch(parameter_list, base_dir="simulation_results"):



    node_count = 50
    linear_dampening = 0.000001
    angular_dampening = 0.000001
    dampening = [linear_dampening, angular_dampening]
    duration = 1.0 # seconds
    sample_rate = 50_000 # Hz
    chunk_size = 512
    dt = 2e-7 # seconds
    oversampling_factor = int(1.0 / (dt * sample_rate))
    weights = make_weights(node_count, 10, 8)
    dispatches = 1


    for i, params in enumerate(parameter_list):
        sim_dir = Path(base_dir) / f"run_{i:03d}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting {sim_dir}")
        
        try:
            ref_pos, nodes, edges, dl, mass, inertia, K_se, K_bt = generate_straight_rod(node_count, **params)
            sim = Simulation(
                nodes=nodes,
                edges=edges,
                hammer_weights=weights,
                oversampling_factor=oversampling_factor,
                chunk_size=chunk_size,
                dt=dt,
                dx=dl,
                mass=mass,
                stiffness_se = K_se,
                stiffness_bt = K_bt,
                inertia=inertia,
                clamp_offset=0,
                dampening=dampening
            )
 
            
            sim.initialize()

            _, force = make_force_curve(dt, 0.004)

            ds_force = force[::10]
            for f in ds_force:
                sim.hammer(10, f)

            # 1. Run and save frames to disk individually
            for dispatch in range(dispatches):
                sim.compute()
                n, e = sim.save()

                with open(sim_dir / f"n_{dispatch:04d}.pkl", "wb") as f:
                    pickle.dump(n, f)
                with open(sim_dir / f"e_{dispatch:04d}.pkl", "wb") as f:
                    pickle.dump(e, f)
            
            print("Consolidating data...")

            all_n = []
            for d in range(dispatches):
                with open(sim_dir / f"n_{d:04d}.pkl", "rb") as f:
                    data = pickle.load(f)
                    all_n.append(data)
            with open(sim_dir / "all_nodes.pkl", "wb") as f:
                pickle.dump(all_n, f)
            
            # Cleanup individual frame files
            for f in sim_dir.glob("n_*.pkl"): f.unlink()
            for f in sim_dir.glob("e_*.pkl"): f.unlink()

        except Exception as e:
            print(f'Simulation {i} failed: {e}')



if __name__ == "__main__":
    batch = [
        {
            "tuned_length" : 1.22, # unit: m
            "tension_force" : 600, # unit: N
            "core_radius" : 6.0e-4, # unit: m
            "winding_radius" : 4.0e-4, # unit: m
            "E_core" : 2.07e11, # unit: Pa
            "G_core" : 8.0e10, # unit: Pa
            "rho_core" : 7.85e3, # unit: kg/m^3
            "E_winding" : 1.1e11, # unit: Pa
            "G_winding" : 4.1e10, # unit: Pa
            "rho_winding" : 8.96e3, # unit: kg/m^3
            "packing_factor" : 0.8, # unit: 1
            "mu" : 0.15, # unit: 1
            "twists" : 0.0, # unit: 1
        }
    ]
    run_batch(batch)
