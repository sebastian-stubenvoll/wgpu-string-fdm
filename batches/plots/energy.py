import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt, butter


def _dynamic_above_baseline(x):
    return x - np.min(x)


def plot_energies(time, trans_ke, rot_ke, bend_pe, shear_pe, print_totals=False,
                  normalized=False, mode_transfer=False, title_prefix="", filename=""):

    trans_ke = np.asarray(trans_ke, dtype=np.float64)
    rot_ke = np.asarray(rot_ke, dtype=np.float64)
    bend_pe = np.asarray(bend_pe, dtype=np.float64)
    shear_pe = np.asarray(shear_pe, dtype=np.float64)

    fig = plt.figure(figsize=(12, 6))

    if normalized or mode_transfer:
        bend_dyn = _dynamic_above_baseline(bend_pe)
        shear_dyn = _dynamic_above_baseline(shear_pe)
        kin = trans_ke + rot_ke
        pot = bend_dyn + shear_dyn
        total = kin + pot

        peak = np.max(total) if total.size else 0.0
        active = total > (1e-3 * peak) if peak > 0 else np.zeros_like(total, dtype=bool)

        def fraction(x, fill):
            out = np.full_like(x, fill)
            np.divide(x, total, out=out, where=active)
            return out

        active_idx = np.nonzero(active)[0]
        if active_idx.size:
            t0, t1 = time[active_idx[0]], time[active_idx[-1]]
            span = max(t1 - t0, 1e-9)
            view = (max(float(time[0]), t0 - 0.02 * span),
                    min(float(time[-1]), t1 + 0.02 * span))
        else:
            view = (float(time[0]), float(time[-1]))

        if normalized:
            trans = fraction(trans_ke, 0.0)
            rot = fraction(rot_ke, 0.0)
            bend = fraction(bend_dyn, 0.0)
            shear = fraction(shear_dyn, 0.0)

            plt.stackplot(
                time, trans, rot, bend, shear,
                labels=["Translational KE", "Rotational KE",
                        "Bend/Twist PE", "Shear/Stretch PE"],
                colors=["tab:blue", "tab:orange", "tab:green", "tab:red"],
            )
            plt.ylim(0, 1)
            plt.xlim(*view)
            plt.ylabel("Fraction of Dynamic Energy")
            title = f"[{title_prefix}] Energy Distribution (dynamic; ringing window)"
        else:  
            trans = fraction(trans_ke, np.nan)
            rot = fraction(rot_ke, np.nan)
            bend = fraction(bend_dyn, np.nan)
            shear = fraction(shear_dyn, np.nan)

            plt.plot(time, trans, label="Translational KE", alpha=0.5, linestyle=':')
            plt.plot(time, rot, label="Rotational KE", alpha=0.5, linestyle=':')
            plt.plot(time, bend, label="Bend/Twist PE", alpha=0.5, linestyle=':')
            plt.plot(time, shear, label="Shear/Stretch PE", alpha=0.5, linestyle=':')
            if print_totals:
                plt.plot(time, fraction(kin, np.nan), label="TOTAL Kinetic",
                         linewidth=2, color='blue')
                plt.plot(time, fraction(pot, np.nan), label="TOTAL Potential",
                         linewidth=2, color='orange')
            plt.ylim(-0.05, 1.05)
            plt.xlim(*view)
            plt.ylabel("Fraction of Dynamic Energy")
            title = f"[{title_prefix}] Energy Mode Transfer (dynamic fraction)"
    else:
        sos = butter(4, 0.01, btype='high', output='sos')
        bend_ac = sosfiltfilt(sos, bend_pe)
        shear_ac = sosfiltfilt(sos, shear_pe)

        total_kin = trans_ke + rot_ke
        total_pot = bend_ac + shear_ac
        total_energy = total_kin + total_pot

        title = f"[{title_prefix}] Cosserat Rod Energy Breakdown (Offset Removed)"
        plt.plot(time, sosfiltfilt(sos, trans_ke), label="Translational KE", alpha=0.4, linestyle=':')
        plt.plot(time, sosfiltfilt(sos, rot_ke), label="Rotational KE", alpha=0.4, linestyle=':')
        plt.plot(time, bend_ac, label="Bend/Twist PE", alpha=0.4, linestyle=':')
        plt.plot(time, shear_ac, label="Shear/Stretch PE", alpha=0.4, linestyle=':')

        if print_totals:
            plt.plot(time, total_kin, label="TOTAL Kinetic", linewidth=2, color='blue')
            plt.plot(time, total_pot, label="TOTAL Potential", linewidth=2, color='orange')
            plt.plot(time, total_energy, label="TOTAL SYSTEM ENERGY",
                     color='gray', linestyle='--', linewidth=1.5)
        plt.ylabel("Energy (Joules)")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]
