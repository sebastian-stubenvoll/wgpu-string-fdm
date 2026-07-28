import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def _pick_comp(field, comp):
    if comp is not None:
        return comp
    return 0 if np.max(np.abs(field[:, :, 0])) >= np.max(np.abs(field[:, :, 1])) else 1


def plot_spacetime_heatmap(time, field, dl, title_prefix="", filename="", comp=None,
                           max_t=800, max_x=500):
    comp = _pick_comp(field, comp)
    disp = field[:, :, comp]
    T, N = disp.shape
    ts = max(1, T // max_t)
    xs = max(1, N // max_x)
    disp_d = disp[::ts, ::xs]
    t_d = np.asarray(time)[::ts][:disp_d.shape[0]]
    x_d = np.arange(N)[::xs][:disp_d.shape[1]] * dl

    vmax = float(np.max(np.abs(disp_d))) or 1.0
    axis = "Y" if comp == 0 else "Z"

    fig, ax = plt.subplots(figsize=(11, 6))
    pcm = ax.pcolormesh(x_d, t_d, disp_d, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        shading='auto')
    fig.colorbar(pcm, ax=ax, label=f"{axis} displacement")
    ax.set_xlabel("Position along string (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"[{title_prefix}] Space-Time Transient ({axis})")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return [filename]


def plot_waterfall_3d(time, field, dl, title_prefix="", filename="", comp=None,
                      n_lines=70, max_x=300):
    comp = _pick_comp(field, comp)
    disp = field[:, :, comp]
    T, N = disp.shape
    ts = max(1, T // n_lines)
    xs = max(1, N // max_x)
    x = np.arange(N)[::xs] * dl
    times = np.asarray(time)
    axis = "Y" if comp == 0 else "Z"

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.viridis
    idxs = list(range(0, T, ts))
    for j, i in enumerate(idxs):
        ax.plot(x, np.full_like(x, times[i]), disp[i, ::xs],
                color=cmap(j / max(1, len(idxs) - 1)), lw=0.6)

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Time (s)")
    ax.set_zlabel(f"{axis} disp")
    ax.set_title(f"[{title_prefix}] 3D Waterfall Transient ({axis})")
    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return [filename]
