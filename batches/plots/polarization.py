import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _colored_line(ax, x, y, t, cmap='viridis', lw=0.6):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=lw)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def _decimate(*arrays, max_points=6000):
    n = len(arrays[0])
    step = max(1, n // max_points)
    return [a[::step] for a in arrays]


def plot_polarization(time, pos, title_prefix="", filename="", attack_window=0.03):
    y = pos[:, 1] - np.mean(pos[:, 1])
    z = pos[:, 2] - np.mean(pos[:, 2])
    time = np.asarray(time)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    yd, zd, td = _decimate(y, z, time)
    lc = _colored_line(ax1, yd, zd, td)
    ax1.set_title("Full trajectory")
    ax1.set_xlabel("Y displacement")
    ax1.set_ylabel("Z displacement")
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.grid(True, alpha=0.3)
    fig.colorbar(lc, ax=ax1, label="Time (s)")

    mask = time <= (time[0] + attack_window)
    if mask.sum() > 2:
        lc2 = _colored_line(ax2, y[mask], z[mask], time[mask], cmap='plasma', lw=1.0)
        fig.colorbar(lc2, ax=ax2, label="Time (s)")
        ax2.set_title(f"Attack transient (first {attack_window*1000:.0f} ms)")
    else:
        ax2.text(0.5, 0.5, "window too short", ha='center', va='center',
                 transform=ax2.transAxes)
    ax2.set_xlabel("Y displacement")
    ax2.set_ylabel("Z displacement")
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"[{title_prefix}] Transverse Polarization (Y vs Z)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return [filename]
