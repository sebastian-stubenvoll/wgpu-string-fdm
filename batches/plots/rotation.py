import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import spectrogram


def _infer_fs(time):
    time = np.asarray(time)
    if len(time) > 1:
        step = np.median(np.diff(time))
        if step > 0:
            return 1.0 / step
    return 1.0


def plot_angular_velocity(time, omega, title_prefix="", filename="", fs=None):
    omega = np.asarray(omega, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if fs is None:
        fs = _infer_fs(time)

    omega_mag = np.linalg.norm(omega, axis=1)

    fig, axes = plt.subplots(5, 1, figsize=(11, 12))

    comp = [("$\\omega_x$ (bend)", "tab:red"),
            ("$\\omega_y$ (bend)", "tab:green"),
            ("$\\omega_z$ (twist)", "tab:blue")]
    for i, (lbl, color) in enumerate(comp):
        ax = axes[i]
        ax.plot(time, omega[:, i], color=color, linewidth=0.7)
        ax.set_ylabel(f"{lbl}\n(rad/s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f"[{title_prefix}] Angular Velocity Components")
    axes[2].set_xlabel("Time (s)")

    axm = axes[3]
    axm.plot(time, omega_mag, color="black", linewidth=0.8)
    axm.set_ylabel("|$\\omega$| (rad/s)")
    axm.set_xlabel("Time (s)")
    axm.set_title("Angular Speed Magnitude")
    axm.grid(True, alpha=0.3)

    axs = axes[4]
    sig = omega_mag - np.mean(omega_mag)
    nperseg = int(min(2048, max(64, len(sig) // 16)))
    if len(sig) >= nperseg and nperseg >= 16:
        f, t_spec, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg,
                                     noverlap=nperseg // 2, scaling="spectrum")
        Sxx_db = 10.0 * np.log10(Sxx + 1e-20)
        pcm = axs.pcolormesh(t_spec + time[0], f, Sxx_db, shading="gouraud", cmap="magma")
        axs.set_ylim(0, min(fs / 2, 20000))
        fig.colorbar(pcm, ax=axs, label="Power (dB)")
        axs.set_ylabel("Frequency (Hz)")
        axs.set_xlabel("Time (s)")
        axs.set_title("Angular Velocity Spectrogram (|$\\omega$|)")
    else:
        axs.text(0.5, 0.5, "Signal too short for spectrogram",
                 ha="center", va="center", transform=axs.transAxes)
        axs.set_axis_off()

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_axis_angle_over_time(time, quats, title_prefix="", filename=""):
    quats = np.asarray(quats, dtype=np.float64)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q_norm = quats / norms

    rotvec = R.from_quat(q_norm).as_rotvec()
    angles = np.linalg.norm(rotvec, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        axes_arr = np.where(angles[:, None] > 1e-8, rotvec / angles[:, None], 0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(time, np.degrees(angles), label="Rotation Angle", color='tab:purple')
    ax1.set_ylabel("Angle (degrees)")
    ax1.set_title(f"[{title_prefix}] Axis-Angle Orientation Over Time")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2.plot(time, axes_arr[:, 0], label="Axis X", color="tab:red")
    ax2.plot(time, axes_arr[:, 1], label="Axis Y", color="tab:green")
    ax2.plot(time, axes_arr[:, 2], label="Axis Z", color="tab:blue")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Axis Component")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]
