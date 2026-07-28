import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def stiff_string_model(n, f1, B):
    return n * f1 * np.sqrt(1.0 + B * (np.asarray(n) ** 2))


def fft_mag(signal):
    sig = signal - np.mean(signal)
    window = np.hanning(len(sig))
    return np.abs(np.fft.rfft(sig * window))


def detect_peaks(freqs, mag_norm, height=0.02, distance=5, parabolic=False):
    idx, _ = find_peaks(mag_norm, height=height, distance=distance)
    pf = freqs[idx].astype(float)
    pm = mag_norm[idx].astype(float)
    if parabolic and len(freqs) > 2:
        df = freqs[1] - freqs[0]
        for j, i in enumerate(idx):
            if 0 < i < len(mag_norm) - 1:
                a = np.log(mag_norm[i - 1] + 1e-20)
                b = np.log(mag_norm[i] + 1e-20)
                c = np.log(mag_norm[i + 1] + 1e-20)
                denom = a - 2.0 * b + c
                if abs(denom) > 1e-12:
                    p = np.clip(0.5 * (a - c) / denom, -0.5, 0.5)
                    pf[j] = freqs[i] + p * df
    return pf, pm


def estimate_f1_first_peak(pf, pm, fundamental_weight=0.2):
    """Original heuristic: first peak louder than `fundamental_weight`."""
    if len(pf) == 0:
        return 0.0
    sig = np.where(pm > fundamental_weight)[0]
    return float(pf[sig[0]]) if len(sig) else float(pf[np.argmax(pm)])


def estimate_f1_median_spacing(pf, pm=None):
    """Robust f1 from the median spacing between consecutive partials. Survives a
    weak/missing fundamental and spurious low peaks."""
    if len(pf) < 2:
        return estimate_f1_first_peak(pf, np.ones_like(pf) if pm is None else pm)
    diffs = np.diff(np.sort(pf))
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else float(np.sort(pf)[0])


def associate_harmonics(pf, f1, B=0.0, max_n=25, rel_window=0.45):
    """Match measured peaks to partial numbers using the stiff-string prediction
    (so inharmonic high partials stay inside the window)."""
    ns, fns = [], []
    misses = 0
    if len(pf) == 0 or f1 <= 0:
        return np.array([]), np.array([])
    for n in range(1, max_n + 1):
        if misses > 2:
            break
        target = stiff_string_model(n, f1, B)
        j = int(np.argmin(np.abs(pf - target)))
        if abs(pf[j] - target) < rel_window * f1:
            ns.append(n)
            fns.append(pf[j])
            misses = 0
        else:
            misses += 1
    return np.array(ns), np.array(fns)


def fit_stiff(ns, fns, f1_est):
    f1, B = f1_est, 0.0
    if len(ns) >= 4:
        try:
            popt, _ = curve_fit(stiff_string_model, ns, fns, p0=[f1_est, 1e-4], maxfev=10000)
            f1, B = float(popt[0]), float(popt[1])
        except (RuntimeError, ValueError):
            pass
    return f1, B


def residual_metrics(ns, fns, f1, B):
    if len(ns) < 2:
        return float('nan'), float('nan')
    pred = stiff_string_model(ns, f1, B)
    resid = fns - pred
    rms = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((fns - np.mean(fns)) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return rms, r2


def run_variant(freqs, mag_norm, parabolic=False, median_f1=False, iterative=False,
                fundamental_weight=0.2):
    """Run one fit configuration and return its results/metrics."""
    pf, pm = detect_peaks(freqs, mag_norm, parabolic=parabolic)
    empty = dict(f1=0.0, B=0.0, ns=np.array([]), fns=np.array([]),
                 pf=pf, pm=pm, rms=float('nan'), r2=float('nan'))
    if len(pf) < 2:
        return empty

    f1_est = (estimate_f1_median_spacing(pf, pm) if median_f1
              else estimate_f1_first_peak(pf, pm, fundamental_weight))
    ns, fns = associate_harmonics(pf, f1_est, B=0.0)
    f1, B = fit_stiff(ns, fns, f1_est)

    if iterative:
        for _ in range(3):
            ns, fns = associate_harmonics(pf, f1, B=B)
            new_f1, new_B = fit_stiff(ns, fns, f1)
            converged = abs(new_B - B) < 1e-6 and abs(new_f1 - f1) < 1e-3
            f1, B = new_f1, new_B
            if converged:
                break

    rms, r2 = residual_metrics(ns, fns, f1, B)
    return dict(f1=f1, B=B, ns=ns, fns=fns, pf=pf, pm=pm, rms=rms, r2=r2)


# Cumulative variants: baseline, then each improvement stacked on top.
VARIANTS = [
    ("Baseline (current)", dict(parabolic=False, median_f1=False, iterative=False)),
    ("+ Parabolic interp",  dict(parabolic=True,  median_f1=False, iterative=False)),
    ("+ Median-spacing f1", dict(parabolic=True,  median_f1=True,  iterative=False)),
    ("+ Iterative refit",   dict(parabolic=True,  median_f1=True,  iterative=True)),
]


def plot_inharmonicity_comparison(time, pos, dt, oversampling_factor, cutoff=20000,
                                  title_prefix="", filename_prefix=""):
    """One figure per axis, stacking the baseline fit and the three cumulative
    improvements so the effect of each is directly comparable."""
    T = len(time)
    freqs = np.fft.rfftfreq(T, dt * oversampling_factor)
    components = [("x", pos[:, 0]), ("y", pos[:, 1]), ("z", pos[:, 2])]

    generated = []
    all_stats = {}
    for label, data in components:
        bins = fft_mag(data)
        mag_norm = bins / np.max(bins) if np.max(bins) > 0 else bins
        results = [(name, run_variant(freqs, mag_norm, **kw)) for name, kw in VARIANTS]

        fig, axes = plt.subplots(len(results), 1, figsize=(10, 3.0 * len(results)))
        if len(results) == 1:
            axes = [axes]

        comp_stats = {}
        for ax, (name, res) in zip(axes, results):
            ax.plot(freqs, mag_norm, color='lightgray', label='Spectrum')
            if len(res['pf']):
                ax.plot(res['pf'], res['pm'], 'x', color='tab:blue', ms=5, label='Peaks')

            ns, fns = res['ns'], res['fns']
            if len(ns):
                fitted = stiff_string_model(ns, res['f1'], res['B'])
                ax.scatter(fitted, np.interp(fitted, freqs, mag_norm),
                           color='tab:red', zorder=5, label='Fitted partials')
                for f_ideal in fitted:
                    ax.axvline(f_ideal, color='green', linestyle=':', alpha=0.4)
                xmax = min(cutoff, float(np.max(fns)) * 1.15)
            else:
                xmax = min(cutoff, 2000.0)

            ax.set_xlim(0, xmax)
            ax.set_ylabel("Norm. mag")
            r2s = f"{res['r2']:.4f}" if np.isfinite(res['r2']) else "n/a"
            ax.set_title(f"{name}:  f1={res['f1']:.2f} Hz   B={res['B']:.3e}   "
                         f"R²={r2s}   N={len(ns)}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

            comp_stats[name] = {
                "fundamental": res['f1'], "inharmonicity": res['B'],
                "r2": res['r2'], "rms": res['rms'], "n_partials": int(len(ns)),
            }

        axes[-1].set_xlabel("Frequency (Hz)")
        fig.suptitle(f"[{title_prefix}] {label}-axis Inharmonicity Fit Comparison")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        path = f"{filename_prefix}_inharm_{label}.png"
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        generated.append(path)
        all_stats[label] = comp_stats

    stats_path = f"{filename_prefix}_inharm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    generated.append(stats_path)
    return generated


def plot_inharmonicity_deviation(time, pos, dt, oversampling_factor, cutoff=20000,
                                 title_prefix="", filename=""):
    """f_n / (n*f1) vs n with the sqrt(1+B n^2) curve overlaid (best/iterative fit).
    Makes partial 'sharpening' and the fit quality visually obvious."""
    T = len(time)
    freqs = np.fft.rfftfreq(T, dt * oversampling_factor)
    components = [("x", pos[:, 0]), ("y", pos[:, 1]), ("z", pos[:, 2])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (label, data) in zip(axes, components):
        bins = fft_mag(data)
        mag_norm = bins / np.max(bins) if np.max(bins) > 0 else bins
        res = run_variant(freqs, mag_norm, parabolic=True, median_f1=True, iterative=True)
        ns, fns, f1, B = res['ns'], res['fns'], res['f1'], res['B']

        if len(ns) >= 2 and f1 > 0:
            measured_ratio = fns / (ns * f1)
            nn = np.linspace(1, float(np.max(ns)), 200)
            ax.plot(nn, np.sqrt(1.0 + B * nn ** 2), color='tab:red',
                    label=f"√(1+Bn²), B={B:.2e}")
            ax.plot(ns, measured_ratio, 'o', color='tab:blue', label='Measured')
            ax.axhline(1.0, color='gray', ls=':', alpha=0.6)
            ax.set_title(f"{label}-axis (f1={f1:.1f} Hz)")
            ax.legend(loc='upper left', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'insufficient partials', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f"{label}-axis")

        ax.set_xlabel("Partial number n")
        ax.set_ylabel("f_n / (n·f1)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"[{title_prefix}] Inharmonicity Deviation (partial sharpening)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return [filename]
