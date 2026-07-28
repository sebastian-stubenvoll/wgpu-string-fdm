import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from plots.inharmonicity import run_variant, fft_mag, stiff_string_model


def _dominant_axis(pos):
    idx = 1 if np.var(pos[:, 1]) >= np.var(pos[:, 2]) else 2
    return idx, ("Y" if idx == 1 else "Z")


def _partials_from_fit(time, sig, dt, oversampling_factor):
    T = len(time)
    freqs = np.fft.rfftfreq(T, dt * oversampling_factor)
    bins = fft_mag(sig)
    mag_norm = bins / np.max(bins) if np.max(bins) > 0 else bins
    res = run_variant(freqs, mag_norm, parabolic=True, median_f1=True, iterative=True)
    return res['f1'], res['B'], res['ns']


def plot_partial_decay(time, pos, dt, oversampling_factor, title_prefix="",
                       filename="", max_partials=12):
    fs = 1.0 / (dt * oversampling_factor)
    ax_i, ax_name = _dominant_axis(pos)
    sig = pos[:, ax_i] - np.mean(pos[:, ax_i])     
    f1, B, ns = _partials_from_fit(time, sig, dt, oversampling_factor)

    nperseg = int(min(4096, max(64, len(sig) // 32)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if len(ns) == 0 or f1 <= 0 or len(sig) < nperseg or nperseg < 16:
        ax1.text(0.5, 0.5, "insufficient data for decay analysis",
                 ha='center', va='center', transform=ax1.transAxes)
        ax2.axis('off')
        fig.suptitle(f"[{title_prefix}] Partial Decay Analysis ({ax_name}-axis)")
        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return [filename]

    partial_freqs = stiff_string_model(ns, f1, B)
    f_s, t_s, S = spectrogram(sig, fs=fs, nperseg=nperseg,
                              noverlap=int(nperseg * 0.75), scaling='spectrum',
                              mode='magnitude')

    cmap = plt.cm.viridis
    n_show = min(max_partials, len(ns))
    t60s, ns_used = [], []
    for k in range(n_show):
        n, pf = ns[k], partial_freqs[k]
        bi = int(np.argmin(np.abs(f_s - pf)))
        env = S[bi]
        peak = np.max(env)
        if peak <= 0:
            continue
        env_db = 20.0 * np.log10(env / peak + 1e-12)
        color = cmap(k / max(1, n_show - 1))
        ax1.plot(t_s + time[0], env_db, color=color, lw=0.8, label=f"n={n}")

        pk = int(np.argmax(env))
        seg_t, seg_db = t_s[pk:], env_db[pk:]
        valid = seg_db > -60
        if valid.sum() >= 5:
            slope = np.polyfit(seg_t[valid], seg_db[valid], 1)[0]
            if slope < 0:
                t60s.append(-60.0 / slope)
                ns_used.append(n)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Level (dB re partial peak)")
    ax1.set_ylim(-80, 3)
    ax1.set_title("Partial decay envelopes")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=2, loc='upper right')

    if t60s:
        ax2.plot(ns_used, t60s, 'o-', color='tab:purple')
    ax2.set_xlabel("Partial number n")
    ax2.set_ylabel("T60 (s)")
    ax2.set_title("Decay time per partial")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"[{title_prefix}] Partial Decay Analysis (y-axis)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return [filename]


def plot_spectral_decay(time, pos, dt, oversampling_factor, title_prefix="", filename=""):
    """Spectral centroid (brightness) over time + Schroeder energy-decay curve."""
    fs = 1.0 / (dt * oversampling_factor)
    ax_i, ax_name = _dominant_axis(pos)
    sig = pos[:, ax_i] - np.mean(pos[:, ax_i])
    time = np.asarray(time)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    nperseg = int(min(4096, max(64, len(sig) // 32)))
    if len(sig) >= nperseg and nperseg >= 16:
        f_s, t_s, S = spectrogram(sig, fs=fs, nperseg=nperseg,
                                  noverlap=int(nperseg * 0.75), scaling='spectrum',
                                  mode='magnitude')
        power = S ** 2
        denom = power.sum(axis=0)
        denom[denom == 0] = 1e-20
        centroid = (f_s[:, None] * power).sum(axis=0) / denom
        ax1.plot(t_s + time[0], centroid, color='tab:orange')
    else:
        ax1.text(0.5, 0.5, "signal too short", ha='center', va='center',
                 transform=ax1.transAxes)
    ax1.set_ylabel("Spectral centroid (Hz)")
    ax1.set_xlabel("Time (s)")
    ax1.set_title("Brightness over time")
    ax1.grid(True, alpha=0.3)

    energy = sig.astype(np.float64) ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10.0 * np.log10(edc / np.max(edc) + 1e-20) if np.max(edc) > 0 else np.zeros_like(edc)
    ax2.plot(time, edc_db, color='tab:blue')
    ax2.set_ylim(-80, 2)
    ax2.set_ylabel("Energy decay (dB)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Schroeder Energy Decay Curve")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"[{title_prefix}] Spectral Centroid & Energy Decay ({ax_name}-axis)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return [filename]
