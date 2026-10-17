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


def plot_node_transient(time, pos, vel, onset=None, title_prefix="", filename=""):
    """One node's motion right after the strike: transverse (Y, Z) and
    longitudinal (X) displacement and velocity, each on its own panel since
    the longitudinal motion is orders of magnitude smaller."""
    from plots.energy import PALETTE, INK_MUTED, _style_axis

    t = np.asarray(time, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    c_y, c_z, c_x = PALETTE[0], PALETTE[2], PALETTE[1]

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    panels = (
        (axes[0], pos, "Transverse displacement (m)", True),
        (axes[1], pos, "Longitudinal displacement (m)", False),
        (axes[2], vel, "Transverse velocity (m/s)", True),
        (axes[3], vel, "Longitudinal velocity (m/s)", False),
    )
    for ax, arr, ylabel, transverse in panels:
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.6)
        if transverse:
            ax.plot(t, arr[:, 1], color=c_y, linewidth=1.0, label="Y")
            ax.plot(t, arr[:, 2], color=c_z, linewidth=1.0, label="Z")
            ax.legend(loc='upper right', frameon=False, fontsize=9)
        else:
            ax.plot(t, arr[:, 0], color=c_x, linewidth=1.0)
        if onset is not None:
            ax.axvline(onset, color=INK_MUTED, linestyle=':', linewidth=1.0)
        ax.set_ylabel(ylabel)
        _style_axis(ax)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"[{title_prefix}] Node Motion after the Strike")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return [filename]
