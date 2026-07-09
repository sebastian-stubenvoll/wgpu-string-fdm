import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

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
        highest_peak = peak_freqs.max() if len(peak_freqs) > 0 else 0

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
            ax_fft.set_xlim(0, cutoff) 
            if len(fns) > 0: ax_fit.set_xlim(0, max(fns) * 1.2)
            ax_fit.set_title(f"[{title_prefix}] {label}-axis FFT Inharmonicity Fit (B = {B_fit:.5f})")
        else:
            ax_fit.plot(freqs, mag_norm, color='lightgray', label='Spectrum')
            ax_fit.set_xlim(0, cutoff)
            ax_fft.set_xlim(0, cutoff)
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

        fit_results = dict()
        if len(peak_freqs) >= 2:
            fit_results[label] = {
                "fundamental": float(f1_fit),
                "inharmonicity": float(B_fit),
            }
        else:
            fit_results[label] = {
                "fundamental": 0.0,
                "inharmonicity": 0.0,
            }

    stats_path = filename_prefix + "_fft_stats.json"
    with open(stats_path, "w") as f:
        json.dump(fit_results, f, indent=2)

    generated_files.append(stats_path)
    return generated_files
