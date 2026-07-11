import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, sosfiltfilt, butter

def plot_energies(time, trans_ke, rot_ke, bend_pe, shear_pe, print_totals=False, 
                  normalized=False, mode_transfer=False, title_prefix="", filename=""):

    # Remove the potential energy offset by applying a highpass filter
    sos = butter(4, 0.01, btype='high', output='sos')
    # trans_ke = sosfiltfilt(sos, trans_ke)
    # rot_ke = sosfiltfilt(sos, rot_ke)
    bend_pe = sosfiltfilt(sos, bend_pe)
    shear_pe = sosfiltfilt(sos, shear_pe)
    
    total_kin = trans_ke + rot_ke
    total_pot = bend_pe + shear_pe
    total_energy = total_kin + total_pot
    
    fig = plt.figure(figsize=(12, 6))
    
    if normalized:
        with np.errstate(divide="ignore", invalid="ignore"):
            trans = np.divide(trans_ke, total_energy,
                              out=np.zeros_like(trans_ke),
                              where=np.abs(total_energy) > 1e-20)
            rot = np.divide(rot_ke, total_energy,
                            out=np.zeros_like(rot_ke),
                            where=np.abs(total_energy) > 1e-20)
            bend = np.divide(bend_pe, total_energy,
                             out=np.zeros_like(bend_pe),
                             where=np.abs(total_energy) > 1e-20)
            shear = np.divide(shear_pe, total_energy,
                              out=np.zeros_like(shear_pe),
                              where=np.abs(total_energy) > 1e-20)

        title = f"[{title_prefix}] Energy Distribution"

        plt.stackplot(
            time,
            trans,
            rot,
            bend,
            shear,
            labels=[
                "Translational KE",
                "Rotational KE",
                "Bend/Twist PE",
                "Shear/Stretch PE",
            ],
            colors=["tab:blue", "tab:orange", "tab:green", "tab:red"],
        )

        plt.ylim(0, 1)
        plt.ylabel("Fraction of Total Energy")
    elif mode_transfer:
        with np.errstate(divide='ignore', invalid='ignore'):
            trans_ke = trans_ke / total_energy
            rot_ke = rot_ke / total_energy
            bend_pe = bend_pe / total_energy
            shear_pe = shear_pe / total_energy
            total_kin = total_kin / total_energy
            total_pot = total_pot / total_energy
        title = f"[{title_prefix}] Energy Mode Transfer (Fraction of Total)"
        plt.plot(time, trans_ke, label="Translational KE", alpha=0.4, linestyle=':')
        plt.plot(time, rot_ke, label="Rotational KE", alpha=0.4, linestyle=':')
        plt.plot(time, bend_pe, label="Bend/Twist PE", alpha=0.4, linestyle=':')
        plt.plot(time, shear_pe, label="Shear/Stretch PE", alpha=0.4, linestyle=':')
        if print_totals:
            plt.plot(time, total_kin, label="TOTAL Kinetic", linewidth=2, color='blue')
            plt.plot(time, total_pot, label="TOTAL Potential", linewidth=2, color='orange')
        plt.ylabel("Fraction of Total Energy")
        plt.ylim(-0.1, 1.1)
    else:
        title = f"[{title_prefix}] Cosserat Rod Energy Breakdown (Offset Removed)"
        plt.plot(time, trans_ke, label="Translational KE", alpha=0.4, linestyle=':')
        plt.plot(time, rot_ke, label="Rotational KE", alpha=0.4, linestyle=':')
        plt.plot(time, bend_pe, label="Bend/Twist PE", alpha=0.4, linestyle=':')
        plt.plot(time, shear_pe, label="Shear/Stretch PE", alpha=0.4, linestyle=':')
        
        if print_totals:
            plt.plot(time, total_kin, label="TOTAL Kinetic", linewidth=2, color='blue')
            plt.plot(time, total_pot, label="TOTAL Potential", linewidth=2, color='orange')
            plt.plot(time, total_energy, label="TOTAL SYSTEM ENERGY", color='gray', linestyle='--', linewidth=1.5)
        plt.ylabel("Energy (Joules)")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]
