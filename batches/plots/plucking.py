"""Plucked-string figures (plectrum contact, see plucking.py and the shader's pick_contact_body).

Inputs come from the pick records (simulation_run p_*.pkl.gz, plucking.pick_arrays): per frame the
contact normal and friction force of the frame's last solver step, and the works summed over the
frame; plus the pick node's displacement and the bridge-end edge force.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots.energy import PALETTE, INK, INK_MUTED
from plots.bowing import _axes, _save, plot_time_series  # noqa: F401  (re-exported for post_process)


def plot_contact_forces(time, normal_force, friction_force, release_time, title, filename):
    """Normal and friction force of the pick on the string while in contact (ms)."""
    fig, ax = _axes()
    ax.plot(time * 1e3, normal_force, color=PALETTE[0], lw=1.0, label="normal force")
    ax.plot(time * 1e3, friction_force, color=PALETTE[1], lw=1.0, label="friction force")
    ax.axvline(release_time * 1e3, color=INK_MUTED, lw=0.8, ls=":")
    ax.annotate(" release", (release_time * 1e3, ax.get_ylim()[1]), va="top", ha="left", color=INK_MUTED)
    ax.set_xlim(time[0] * 1e3, time[-1] * 1e3)
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, ax, title, "Time (ms)", "Force (N)", filename)


def plot_displacement(time, y, z, release_time, title, filename):
    """Pick-node displacement in the plucking direction (y) and normal to the instrument top (z)."""
    fig, ax = _axes()
    ax.plot(time * 1e3, y * 1e3, color=PALETTE[0], lw=1.0, label="y (plucking direction)")
    ax.plot(time * 1e3, z * 1e3, color=PALETTE[1], lw=1.0, label="z (normal to the top)")
    ax.axvline(release_time * 1e3, color=INK_MUTED, lw=0.8, ls=":")
    ax.set_xlim(time[0] * 1e3, time[-1] * 1e3)
    ax.legend(frameon=False, loc="upper right")
    return _save(fig, ax, title, "Time (ms)", "Displacement (mm)", filename)


def plot_orbit(y, z, title, filename):
    """Trajectory of the pick node in the cross-section plane (both polarisations), mm."""
    fig, ax = _axes(figsize=(6, 6))
    ax.plot(y * 1e3, z * 1e3, color=PALETTE[0], lw=0.6)
    ax.plot(y[0] * 1e3, z[0] * 1e3, "o", color=PALETTE[1], ms=4)
    ax.set_aspect("equal", adjustable="datalim")
    return _save(fig, ax, title, "y (mm)", "z (mm)", filename)
