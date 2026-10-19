"""Collect a Schelleng sweep (run_batch.py schelleng) into regime maps.

python schelleng.py <runs directory> [output directory]

Reads every run's parameters.json and plots/bow_summary.json, groups the runs by friction model and
torsion setting, and draws one map per group: the regime (plots/bowing.regime) at each bow position
and normal force. Also writes schelleng_summary.json with the grids and, per group, the log-log
slope of the upper Helmholtz boundary (for each bow position, the geometric mean of the largest
Helmholtz force and the next force on the grid; Schelleng's maximum-force law has slope -1).
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots.energy import PALETTE, INK, INK_MUTED, GRID_KW
import bowing

REGIME_STYLE = {
    "Helmholtz": dict(marker="o", color=PALETTE[0]),
    "multiple slip": dict(marker="^", color=PALETTE[1]),
    "irregular": dict(marker="x", color=PALETTE[3]),
    "continuous slip": dict(marker="s", color=INK_MUTED, fillstyle="none"),
}


def regime_class(name):
    return "multiple slip" if name.startswith("multiple slip") else name


def collect(base):
    rows = []
    for p in sorted(Path(base).glob("*/parameters.json")):
        summary = p.parent / "plots" / "bow_summary.json"
        P = json.load(open(p))
        if not P.get("bow_params") or not summary.exists():
            continue
        S = json.load(open(summary))
        b = P["bow_params"]
        rows.append({"friction": b["friction"], "torsion": bool(b.get("torsion", True)), "beta": b["beta"],
                     "force": b["normal_force"], "velocity": b["velocity"], "regime": S["regime"],
                     "playing_frequency": S["playing_frequency"], "run": p.parent.name,
                     "Z0": np.sqrt(P["rod_params"]["tension_force"] * P["rod_derived"]["m_node"]
                                   * (P["config"]["node_count"] - 1) / P["rod_params"]["tuned_length"])})
    return rows


def upper_boundary_slope(rows):
    betas = sorted({r["beta"] for r in rows})
    pts = []
    for b in betas:
        col = sorted((r for r in rows if r["beta"] == b), key=lambda r: r["force"])
        helm = [i for i, r in enumerate(col) if r["regime"] == "Helmholtz"]
        if helm and helm[-1] + 1 < len(col):
            pts.append((b, np.sqrt(col[helm[-1]]["force"] * col[helm[-1] + 1]["force"])))
    if len(pts) < 2:
        return None, pts
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0]), pts


def plot_map(rows, title, filename):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.grid(True, which="both", **GRID_KW)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    seen = set()
    for r in rows:
        c = regime_class(r["regime"])
        ax.plot(r["beta"], r["force"], linestyle="none", markersize=8, **REGIME_STYLE[c],
                label=None if c in seen else c)
        seen.add(c)
    mu = bowing.FRICTION_CURVE
    dmu = mu[1] + mu[3]                                   # mu_s - mu_d of the friction curve
    b = np.geomspace(min(r["beta"] for r in rows) * 0.8, max(r["beta"] for r in rows) * 1.25, 50)
    Fmax = 2.0 * rows[0]["Z0"] * rows[0]["velocity"] / (dmu * b)
    ax.plot(b, Fmax, color=INK_MUTED, lw=0.9, ls=(0, (5, 2.5)), label="Schelleng maximum")
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    for axis, values in ((ax.xaxis, sorted({r["beta"] for r in rows})), (ax.yaxis, sorted({r["force"] for r in rows}))):
        axis.set_major_locator(FixedLocator(values))
        axis.set_major_formatter(FixedFormatter([f"{v:g}" for v in values]))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Bow position β", color=INK)
    ax.set_ylabel("Normal force (N)", color=INK)
    ax.set_title(title, loc="left", color=INK)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return filename


def main(base, out=None):
    out = Path(out or base)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(base)
    summary = {}
    for friction in ("curve", "thermal"):
        for torsion in (True, False):
            grp = [r for r in rows if r["friction"] == friction and r["torsion"] == torsion]
            if not grp:
                continue
            key = f"{friction}_{'torsion' if torsion else 'no_torsion'}"
            slope, pts = upper_boundary_slope(grp)
            title = ("Friction curve" if friction == "curve" else "Thermal model") + (
                ", torsion" if torsion else ", torsion inhibited")
            plot_map(grp, title, out / f"schelleng_{key}.png")
            summary[key] = {"runs": len(grp), "upper_boundary_slope": slope, "upper_boundary": pts,
                            "grid": [{k: r[k] for k in ("beta", "force", "regime", "playing_frequency", "run")}
                                     for r in grp]}
    with open(out / "schelleng_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for k, v in summary.items():
        print(f"{k}: {v['runs']} runs, upper Helmholtz boundary slope {v['upper_boundary_slope']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
