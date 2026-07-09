import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def plot_angular_velocity(time, omega, title_prefix="", filename=""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    omega_mag = np.linalg.norm(omega, axis=1)
    
    ax.plot(time, np.abs(omega[:, 0]), label="|Omega X|", alpha=0.6)
    ax.plot(time, np.abs(omega[:, 1]), label="|Omega Y|", alpha=0.6)
    ax.plot(time, np.abs(omega[:, 2]), label="|Omega Z|", alpha=0.6)
    ax.plot(time, omega_mag, label="Magnitude", color='grey', linewidth=1.5)
    
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"[{title_prefix}] Absolute Angular Velocity Over Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]

def plot_axis_angle_over_time(time, quats, title_prefix="", filename=""):
    axes_arr, angles = [], []
    for q in quats:
        q_norm = q / np.linalg.norm(q)
        rotvec = R.from_quat(q_norm).as_rotvec()
        angle = np.linalg.norm(rotvec)
        axis = rotvec / angle if angle > 1e-8 else np.array([0.0, 0.0, 0.0])
        axes_arr.append(axis)
        angles.append(angle)

    axes_arr, angles = np.array(axes_arr), np.array(angles)

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
