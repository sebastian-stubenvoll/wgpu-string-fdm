import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def _dynamic_above_baseline(x):
    return x - np.min(x)


def _upper_envelope(y):
    """Upper envelope of an oscillating signal, interpolated through its peaks.

    Falls back to the raw signal when there are too few peaks to interpolate
    (e.g. a smooth, monotonically decaying total).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size < 3:
        return y
    peaks, _ = find_peaks(y)
    if peaks.size < 2:
        return y
    idx = np.concatenate(([0], peaks, [y.size - 1]))
    return np.interp(np.arange(y.size), idx, y[idx])


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
        # Kinetic energy is zero at rest (no static offset) so it is shown as-is.
        # The potentials carry a large static tension baseline; we remove it by
        # subtracting the quiescent value (a constant), which keeps every curve
        # >= 0 and makes TOTAL SYSTEM ENERGY a clean, decaying mechanical energy
        # equal to the sum of the plotted components.
        bend_dyn = _dynamic_above_baseline(bend_pe)
        shear_dyn = _dynamic_above_baseline(shear_pe)

        total_kin = trans_ke + rot_ke
        total_pot = bend_dyn + shear_dyn
        total_energy = total_kin + total_pot

        title = f"[{title_prefix}] Cosserat Rod Energy Breakdown (Offset Removed)"
        plt.plot(time, trans_ke, label="Translational KE", alpha=0.4, linestyle=':')
        plt.plot(time, rot_ke, label="Rotational KE", alpha=0.4, linestyle=':')
        plt.plot(time, bend_dyn, label="Bend/Twist PE", alpha=0.4, linestyle=':')
        plt.plot(time, shear_dyn, label="Shear/Stretch PE", alpha=0.4, linestyle=':')

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


def plot_energy_envelope(time, trans_ke, rot_ke, bend_pe, shear_pe,
                         title_prefix="", filename=""):
    """Decay overview for the long phases: the total energies as thin lines,
    with peak-interpolated envelopes for the oscillating kinetic/potential."""
    trans_ke = np.asarray(trans_ke, dtype=np.float64)
    rot_ke = np.asarray(rot_ke, dtype=np.float64)
    bend_pe = np.asarray(bend_pe, dtype=np.float64)
    shear_pe = np.asarray(shear_pe, dtype=np.float64)

    bend_dyn = _dynamic_above_baseline(bend_pe)
    shear_dyn = _dynamic_above_baseline(shear_pe)

    total_kin = trans_ke + rot_ke
    total_pot = bend_dyn + shear_dyn
    total_energy = total_kin + total_pot

    fig = plt.figure(figsize=(12, 6))

    # Faint raw totals for context, envelopes on top as the readable lines.
    plt.plot(time, total_kin, color='blue', alpha=0.15, linewidth=0.5)
    plt.plot(time, total_pot, color='orange', alpha=0.15, linewidth=0.5)

    plt.plot(time, _upper_envelope(total_kin), color='blue', linewidth=1.0,
             label="Kinetic (envelope)")
    plt.plot(time, _upper_envelope(total_pot), color='orange', linewidth=1.0,
             label="Potential (envelope)")
    plt.plot(time, total_energy, color='gray', linestyle='--', linewidth=1.0,
             label="TOTAL SYSTEM ENERGY")

    plt.title(f"[{title_prefix}] Energy Decay Envelopes (Offset Removed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (Joules)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]


def plot_energy_components(time, trans_ke, rot_ke, bend_pe, shear_pe,
                           trans_ke_long, rot_ke_tor, bend_pe_twist, shear_pe_ext,
                           title_prefix="", filename=""):
    """Fine-grained breakdown: each of the four energy families split into its
    transverse and axial parts (eight lines total). The passed *_long/_tor/
    *_twist/_ext arrays are the axial components; the transverse part is the
    remainder of the corresponding total (all quadratic forms are diagonal, so
    transverse + axial == total exactly)."""
    trans_ke = np.asarray(trans_ke, dtype=np.float64)
    rot_ke = np.asarray(rot_ke, dtype=np.float64)
    bend_pe = np.asarray(bend_pe, dtype=np.float64)
    shear_pe = np.asarray(shear_pe, dtype=np.float64)
    trans_ke_long = np.asarray(trans_ke_long, dtype=np.float64)
    rot_ke_tor = np.asarray(rot_ke_tor, dtype=np.float64)
    bend_pe_twist = np.asarray(bend_pe_twist, dtype=np.float64)
    shear_pe_ext = np.asarray(shear_pe_ext, dtype=np.float64)

    # Kinetic parts are zero at rest, so shown as-is. Potential parts carry a
    # static tension baseline (dominantly in extension), removed per component.
    transversal_ke = trans_ke - trans_ke_long
    longitudinal_ke = trans_ke_long
    tilting_rot = rot_ke - rot_ke_tor
    torsional_rot = rot_ke_tor
    bending = _dynamic_above_baseline(bend_pe - bend_pe_twist)
    twisting = _dynamic_above_baseline(bend_pe_twist)
    transversal_shear = _dynamic_above_baseline(shear_pe - shear_pe_ext)
    extension = _dynamic_above_baseline(shear_pe_ext)

    fig = plt.figure(figsize=(12, 6))

    # Colour by family, linestyle by transverse (solid) vs axial (dashed).
    plt.plot(time, transversal_ke, color='tab:blue', linewidth=1.0,
             label="Transversal kinetic")
    plt.plot(time, longitudinal_ke, color='tab:blue', linewidth=1.0,
             linestyle='--', label="Longitudinal kinetic")
    plt.plot(time, tilting_rot, color='tab:orange', linewidth=1.0,
             label="Tilting rotation")
    plt.plot(time, torsional_rot, color='tab:orange', linewidth=1.0,
             linestyle='--', label="Torsional rotation")
    plt.plot(time, bending, color='tab:green', linewidth=1.0,
             label="Bending")
    plt.plot(time, twisting, color='tab:green', linewidth=1.0,
             linestyle='--', label="Twisting")
    plt.plot(time, transversal_shear, color='tab:red', linewidth=1.0,
             label="Transversal shear")
    plt.plot(time, extension, color='tab:red', linewidth=1.0,
             linestyle='--', label="Extension")

    plt.title(f"[{title_prefix}] Energy Component Breakdown (Offset Removed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (Joules)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [filename]
