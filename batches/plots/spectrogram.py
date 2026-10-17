import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def plot_node_spectrogram(pos, fs, title_prefix="", filename="", nperseg=4096, overlap=0.75,
                          energy_fraction=0.99, min_fmax=1000.0):
    """Time-frequency view of a node's transverse displacement (Y and Z):
    partial decay, beating between polarisations, phantom partials."""
    pos = np.asarray(pos, dtype=np.float64)
    nperseg = int(min(nperseg, pos.shape[0]))
    noverlap = int(overlap * nperseg)

    specs = []
    for comp in (1, 2):
        f, t, sxx = spectrogram(pos[:, comp] - pos[:, comp].mean(), fs=fs, window='hann',
                                nperseg=nperseg, noverlap=noverlap, scaling='density')
        specs.append(sxx)

    # Frequency limit: where `energy_fraction` of the (Y+Z) energy lies.
    power = (specs[0] + specs[1]).sum(axis=1)
    cum = np.cumsum(power)
    f_max = f[min(np.searchsorted(cum, energy_fraction * cum[-1]), f.size - 1)] if cum[-1] > 0 else f[-1]
    f_max = min(f[-1], max(min_fmax, 1.1 * f_max))
    fmask = f <= f_max

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, layout='constrained')
    for ax, sxx, label in zip(axes, specs, ("Y", "Z")):
        # Each panel relative to its own peak: the polarisation the hammer
        # doesn't drive can sit ~60 dB lower and would otherwise be black.
        peak = sxx.max() or 1.0
        db = 10.0 * np.log10(np.maximum(sxx[fmask] / peak, 1e-10))
        pcm = ax.pcolormesh(t, f[fmask], db, cmap='magma', vmin=-80, vmax=0, shading='nearest')
        fig.colorbar(pcm, ax=ax, label="PSD (dB re. this panel's peak)")
        ax.set_ylim(0, f_max)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{label} displacement")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"[{title_prefix}] Spectrogram ({nperseg / fs * 1e3:.0f} ms window, "
                 f"{fs / nperseg:.1f} Hz resolution)")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return [filename]
