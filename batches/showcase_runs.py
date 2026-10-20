"""Showcase runs of the three excitations with the most complete model of each, emailed when done.

    python showcase_runs.py [--seconds 1.0] [--only piano guitar cello] [--out /simulation_data]
                            [--no-email]

Scenarios (all: variational force law, two-part displacements, torsion in every contact):
  piano   run-002 string (wound, 3 rest twists), hysteretic felt hammer at 6 m/s striking at 1/8 of
          the length from the far end, transverse damping of Bensa et al. (C2) and constant-Q
          torsional damping (twist-rate term)
  guitar  plain steel G3 (D'Addario PL017) plucked by a rigid plectrum with exact Coulomb friction
  cello   cello D (Jargar Classic) bowed with Woodhouse's thermal friction model, 9-node (10.7 mm)
          contact; the bow is lifted at 75 % of the run so the decay is audible

Each run is analysed in memory per dispatch (no chunk files). Written to OUT/<timestamp>_showcase/:
  <scenario>_waveforms.png      bridge force (transverse, zoom, longitudinal) and the excitation
                                vertex's displacement
  <scenario>_energy.png         energy per channel above the rest state (log scale)
  <scenario>_spectrogram.png    transverse bridge force
  <scenario>_bridge_*.wav       bridge force y, z (transverse) and x (longitudinal), 48 kHz, 16 bit
  <scenario>_excitation_displacement_*.wav
                                the same for the displacement of the struck / plucked / bowed vertex
  <scenario>_series.npz         the per-frame series behind the figures (and contact records)
  <scenario>/parameters.json    run parameters (SimulationRun.save_metadata)
  summary.json                  per-scenario numbers, wall times and errors
All figures, sound files and summary.json are emailed together at the end (EMAIL_CONFIG / .env).
"""
import argparse
import copy
import json
import smtplib
import tempfile
import time
import traceback
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt, spectrogram

from config import EMAIL_CONFIG
from simulation_run import SimulationRun
from run_batch import OUTPUT_DIR, RUN_002_CONFIG, RUN_002_ROD, RUN_002_HAMMER, bow_job, pluck_job
from utils import (fundamental_frequency, node_displacement, split_float32, generate_straight_rod, resolve_rod_params,
                   rest_frames, rest_relative_orientations, reference_vectors, reference_offsets)
from plots.energy_flow import _qmul, _qinv, rotate_inv, rotation_deviation_inv
from plots import bowing as bow_plots
import plucking

FRAME_RATE = 50_000          # every scenario writes one frame per 20 us
WAV_RATE = 48_000
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------------------------
def dispatches_for(config, seconds):
    frame_time = config["dt"] * config["oversampling_factor"]
    return max(1, int(np.ceil(seconds / (frame_time * config["chunk_size"]) - 1e-9)))


def piano_job(seconds):
    config = copy.deepcopy(RUN_002_CONFIG)
    config["damping"] = {**config["damping"], "torsion_q_fit_modes": [1, 10]}
    # Strike at 1/8 of the speaking length from the far (agraffe) end; the bridge force is read at
    # edge 0. Pianos strike at 1/7 to 1/9 of the length over most of the compass (Fletcher &
    # Rossing, The Physics of Musical Instruments, ch. 12). run-002 struck node 25 (1/41).
    N = config["node_count"]
    config["hammer_node"] = (N - 1) - int(round((N - 1) / 8))
    config["dispatches"] = dispatches_for(config, seconds)
    return {"name": "piano", "title": "Hammered piano string (run 002 string, wound, 3 twists, struck at 1/8)",
            "run_id": 900, "config": config, "rod": copy.deepcopy(RUN_002_ROD),
            "hammer": copy.deepcopy(RUN_002_HAMMER), "hammer_off_after": 5}


def guitar_job(seconds):
    job = pluck_job(901, torsion=True)
    job["config"]["dispatches"] = dispatches_for(job["config"], seconds)
    return {**job, "name": "guitar", "title": "Plucked guitar G3 (plain steel, plectrum)"}


def cello_job(seconds):
    job = bow_job(902, friction="thermal", torsion=True, width_nodes=9, stop_time=0.75 * seconds)
    job["config"]["dispatches"] = dispatches_for(job["config"], seconds)
    return {**job, "name": "cello", "title": "Bowed cello D (thermal friction, 10.7 mm contact)"}


SCENARIOS = {"piano": piano_job, "guitar": guitar_job, "cello": cello_job}
ENERGY_LABELS = {"kin_long": "kinetic, longitudinal", "kin_trans": "kinetic, transverse",
                 "rot_tilt": "rotational, tilt", "rot_torsion": "rotational, torsion",
                 "shear_stretch": "shear/stretch", "bend_twist": "bend/twist"}


# ---------------------------------------------------------------------------------------------
# Discrete energy of the variational law, from the stored float32 state (float64 evaluation)
# ---------------------------------------------------------------------------------------------
class Energy:
    """Energies as the shader defines them: translational (longitudinal / transverse), rotational
    (tilt / torsion), shear-stretch W_s = sum 1/2 eta^2 sigma^T S sigma dl and bend-twist
    W_b = sum 1/2 eta_D^4 kappa^T B kappa dl over the Voronoi regions."""

    def __init__(self, rod, config):
        N, law = config["node_count"], config["force_law"]
        rod = resolve_rod_params(rod, N, law)
        nodes, _, dl, mass, inertia, K_se, K_bt = generate_straight_rod(N, **rod, force_law=law)
        f32 = lambda x: np.asarray(x, np.float32).astype(np.float64)
        self.dl, self.m = float(np.float32(dl)), float(np.float32(mass))
        self.I, self.K, self.B = f32(inertia), f32(K_se), f32(K_bt)
        twists = rod.get("twists", 0.0)
        frames = rest_frames(N, twists)
        ref = reference_vectors(N, rod["tuned_length"])
        self.R0 = f32(frames)
        self.off = f32(reference_offsets(ref, frames, dl))
        self.rel = f32(rest_relative_orientations(N, twists))
        hi, lo = split_float32([n[0] for n in nodes])
        u0 = hi.astype(np.float64) + lo
        p0 = np.tile([0.0, 0.0, 0.0, 1.0], (N - 1, 1))
        self.rest = self.channels(u0[None], np.zeros((1, N, 3)), p0[None], np.zeros((1, N - 1, 3)))

    def channels(self, u, v, p, omega):
        dl, K, B, I = self.dl, self.K, self.B, self.I
        p = p / np.linalg.norm(p, axis=-1, keepdims=True)
        du = u[:, 1:] - u[:, :-1]
        w = self.off + np.einsum('fnk,nkj->fnj', du, self.R0)
        sigma = rotate_inv(p, w) / dl + rotation_deviation_inv(p)
        length = np.sqrt(w[..., 0] ** 2 + w[..., 1] ** 2 + (dl + w[..., 2]) ** 2)
        s = (2.0 * dl * w[..., 2] + np.sum(w * w, axis=-1)) / (dl * (length + dl))
        eta = 1.0 - 0.3 * s
        rel = _qmul(_qmul(_qinv(p[:, :-1]), self.rel), p[:, 1:])
        rel = np.where(rel[..., 3:] < 0.0, -rel, rel)
        kappa = 2.0 / dl * rel[..., :3]
        eta_v = 1.0 - 0.3 * 0.5 * (s[:, :-1] + s[:, 1:])
        e2 = eta ** 2
        return {
            "kin_long": 0.5 * self.m * np.sum(v[..., 0] ** 2, axis=1),
            "kin_trans": 0.5 * self.m * np.sum(v[..., 1] ** 2 + v[..., 2] ** 2, axis=1),
            "rot_tilt": 0.5 * np.sum(e2 * (I[0] * omega[..., 0] ** 2 + I[1] * omega[..., 1] ** 2), axis=1),
            "rot_torsion": 0.5 * np.sum(e2 * I[2] * omega[..., 2] ** 2, axis=1),
            "shear_stretch": 0.5 * dl * np.sum(e2 * np.sum(K * sigma * sigma, axis=-1), axis=1),
            "bend_twist": 0.5 * dl * np.sum(eta_v ** 4 * np.sum(B * kappa * kappa, axis=-1), axis=1),
        }


# ---------------------------------------------------------------------------------------------
# One scenario
# ---------------------------------------------------------------------------------------------
def run_scenario(job, out_dir):
    name = job["name"]
    t0 = time.perf_counter()
    r = SimulationRun(run_id=job["run_id"], rod_params=copy.deepcopy(job["rod"]),
                      hammer_params=copy.deepcopy(job["hammer"]), config=job["config"], email_config={},
                      base_dir=tempfile.mkdtemp(dir=out_dir, prefix=f"{name}_"),
                      bow_params=job.get("bow"), pick_params=job.get("pick"))
    r.build()
    r.save_metadata()
    config = r.config
    E = Energy(job["rod"], config)
    if name == "piano":
        excite = int(config["hammer_node"])
    elif name == "guitar":
        excite = int(r.pick_derived["pick_node"])
    else:
        nodes = r.bow_derived["contact_nodes"]
        excite = int(nodes[len(nodes) // 2])
    r.sim.enable_hammer(name == "piano")
    D = config["dispatches"]
    bridge, disp, energy, surface, bow = [], [], {k: [] for k in E.rest}, [], []
    for d in range(D):
        r.sim.compute()
        n_raw, e_raw = r.sim.save()
        if name == "cello":
            bow.extend(r.sim.save_bow())
        if name == "guitar":
            surface.extend(r.sim.save_surface())
        nodes = np.asarray(n_raw, np.float32)
        N = nodes.shape[1]
        quats = np.asarray([[e[0] for e in fr] for fr in e_raw], np.float64)[:, :N - 1]
        omega = np.asarray([[e[1][0] for e in fr] for fr in e_raw], np.float64)[:, :N - 1]
        bridge.append(np.asarray([fr[0][1][1] for fr in e_raw], np.float64))
        u = node_displacement(nodes)
        disp.append(u[:, excite])
        ch = E.channels(u, nodes[:, :, 1].astype(np.float64), quats, omega)
        for k in energy:
            energy[k].append(ch[k] - E.rest[k][0])
        del n_raw, e_raw, nodes, quats, omega, u
        if name == "piano" and d == job.get("hammer_off_after", 5):
            r.sim.enable_hammer(False)
        if d % 10 == 0 or d == D - 1:
            el = time.perf_counter() - t0
            log(f"{name}: dispatch {d + 1}/{D}, {el / 60:.1f} min, ETA {el / (d + 1) * (D - d - 1) / 60:.1f} min")
    series = {"bridge": np.concatenate(bridge), "displacement": np.concatenate(disp),
              "excitation_vertex": excite, "frame_dt": config["dt"] * config["oversampling_factor"],
              "f_nominal": fundamental_frequency(r.tension, r.rod_derived["m_node"], config["node_count"],
                                                 job["rod"]["tuned_length"]),
              **{f"energy_{k}": np.concatenate(v) for k, v in energy.items()}}
    if bow:
        series.update({f"bow_{k}": v for k, v in bow_plots.bow_arrays(bow).items()})
    if surface:
        series.update({f"pick_{k}": v for k, v in plucking.pick_arrays(surface).items()})
    wall = time.perf_counter() - t0
    del r
    return series, wall


# ---------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------
def fundamental(x, fs, f_nominal):
    """Fundamental from the strongest spectral peak within 8 % of the flexible-string value
    sqrt(T / rho_L) / (2 L) (stiffness raises it by well under 8 %); 16x zero-padded Hann."""
    x = x - x.mean()
    n = len(x)
    S = np.abs(np.fft.rfft(x * np.hanning(n), 16 * n))
    f = np.fft.rfftfreq(16 * n, 1 / fs)
    band = (f > 0.92 * f_nominal) & (f < 1.08 * f_nominal)
    return float(f[band][np.argmax(S[band])])


def write_wavs(name, S, fs, out_dir):
    """Bridge force and excitation-vertex displacement, per component (change from the first
    frame), high-passed at 20 Hz, resampled to 48 kHz and each normalised to -1 dBFS."""
    files = []
    sos = butter(2, 20.0, btype="highpass", fs=fs, output="sos")
    for signal, prefix in ((S["bridge"], "bridge"), (S["displacement"], "excitation_displacement")):
        for comp, label in ((1, "transverse_y"), (2, "transverse_z"), (0, "longitudinal_x")):
            x = signal[:, comp] - signal[0, comp]
            x = sosfiltfilt(sos, x)
            x = resample_poly(x, WAV_RATE // 2000, fs // 2000)          # 50 kHz -> 48 kHz
            fade = min(len(x), int(0.01 * WAV_RATE))
            x[-fade:] *= np.linspace(1.0, 0.0, fade)
            peak = np.max(np.abs(x))
            if peak == 0.0:
                continue
            pcm = np.int16(np.round(x / peak * 0.89 * 32767))             # -1 dBFS
            path = out_dir / f"{name}_{prefix}_{label}.wav"
            wavfile.write(path, WAV_RATE, pcm)
            files.append(path)
    return files


def plot_waveforms(name, title, S, f1, out_dir):
    t = (np.arange(len(S["bridge"])) + 1) * S["frame_dt"]
    B, U = S["bridge"], S["displacement"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(t, B[:, 1], color=BLUE, lw=0.6, label="y")
    ax.plot(t, B[:, 2], color=ORANGE, lw=0.6, label="z")
    ax.set_title("(a) Transverse bridge force", loc="left"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (N)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax = axes[0, 1]
    t_mid = 0.5 * t[-1]
    w = (t > t_mid) & (t < t_mid + 4.0 / f1)
    ax.plot((t[w] - t[w][0]) * 1e3, B[w, 1], color=BLUE, lw=1.2, label="y")
    ax.plot((t[w] - t[w][0]) * 1e3, B[w, 2], color=ORANGE, lw=1.2, label="z")
    ax.set_title(f"(b) Four periods from t = {t_mid:.2f} s (f₁ ≈ {f1:.1f} Hz)", loc="left")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Force (N)")
    ax = axes[1, 0]
    ax.plot(t, B[:, 0] - B[0, 0], color=VIOLET, lw=0.6)
    ax.set_title("(c) Longitudinal bridge force (change from rest)", loc="left")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (N)")
    ax = axes[1, 1]
    ax.plot(t, U[:, 1] * 1e3, color=BLUE, lw=0.6, label="y")
    ax.plot(t, U[:, 2] * 1e3, color=ORANGE, lw=0.6, label="z")
    ax.set_title(f"(d) Displacement of the excitation vertex {S['excitation_vertex']}", loc="left")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Displacement (mm)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(title, fontweight="bold", x=0.01, ha="left")
    path = out_dir / f"{name}_waveforms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_energy(name, title, S, out_dir):
    t = (np.arange(len(S["bridge"])) + 1) * S["frame_dt"]
    keys = [k for k in S if k.startswith("energy_")]
    cols = [BLUE, AQUA, MAGENTA, VIOLET, YELLOW, ORANGE]
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    total = sum(S[k] for k in keys)
    for k, c in zip(keys, cols):
        y = S[k]
        ax.semilogy(t, np.where(y > 0, y, np.nan), color=c, lw=1.0, label=ENERGY_LABELS[k[len("energy_"):]])
    ax.semilogy(t, np.where(total > 0, total, np.nan), color=INK, lw=1.6, label="total")
    peak = np.nanmax(total)
    if peak > 0:
        ax.set_ylim(peak * 1e-10, peak * 3)          # ten decades below the peak
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy above rest (J)")
    ax.set_title(f"{title}: energy by channel", loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower center")
    path = out_dir / f"{name}_energy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spectrogram(name, title, S, out_dir):
    fs = 1.0 / S["frame_dt"]
    B = S["bridge"]
    comp = 1 if np.std(B[:, 1]) >= np.std(B[:, 2]) else 2
    nper = int(min(4096, max(64, len(B) // 4)))
    f, ts, P = spectrogram(B[:, comp] - B[:, comp].mean(), fs=fs, nperseg=nper, noverlap=nper * 7 // 8,
                           scaling="density")
    db = 10 * np.log10(np.maximum(P, 1e-30))
    hi = np.percentile(db, 99.5)
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    im = ax.pcolormesh(ts, f / 1e3, db, shading="nearest", cmap="magma", vmin=hi - 100, vmax=hi, rasterized=True)
    ax.set_ylim(0, 8); ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (kHz)"); ax.grid(False)
    ax.set_title(f"{title}: bridge force {'yz'[comp - 1]} (Hann {nper / fs * 1e3:.0f} ms)", loc="left")
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("PSD (dB re 1 N²/Hz)"); cb.outline.set_visible(False)
    path = out_dir / f"{name}_spectrogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def send(files, summary, out_dir):
    cfg = EMAIL_CONFIG
    if not (cfg.get("sender_email") and cfg.get("sender_password") and cfg.get("receiver_email")):
        log("email not configured - skipping")
        return
    msg = EmailMessage()
    ok = [k for k, v in summary["scenarios"].items() if "error" not in v]
    msg["Subject"] = f"String showcase finished: {', '.join(ok) or 'no scenario'} ({len(files)} files)"
    msg["From"], msg["To"] = cfg["sender_email"], cfg["receiver_email"]
    msg.set_content(f"Output directory: {out_dir}\n\n" + json.dumps(summary, indent=2))
    total = 0
    for path in sorted(files, key=lambda p: (p.suffix != ".json", p.suffix != ".png", p.name)):
        data = path.read_bytes()
        if total + len(data) > 15e6:        # base64 adds ~37 %; stay under 25 MB mail limits
            log(f"attachment limit reached, not attached: {path.name}")
            continue
        total += len(data)
        sub = {".png": ("image", "png"), ".wav": ("audio", "wav"), ".json": ("application", "json")}[path.suffix]
        msg.add_attachment(data, maintype=sub[0], subtype=sub[1], filename=path.name)
    try:
        with smtplib.SMTP_SSL(cfg["smtp_server"], cfg["smtp_port"]) as smtp:
            smtp.login(cfg["sender_email"], cfg["sender_password"])
            smtp.send_message(msg)
        log(f"email sent ({total / 1e6:.1f} MB)")
    except Exception as e:
        log(f"email failed: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=1.0, help="simulated time per scenario")
    p.add_argument("--only", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    p.add_argument("--out", default=OUTPUT_DIR, help="base output directory (default /simulation_data)")
    p.add_argument("--no-email", action="store_true")
    a = p.parse_args()
    out_dir = Path(a.out) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_showcase"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"output: {out_dir}")
    summary = {"seconds": a.seconds, "scenarios": {}}
    files = []
    for name in a.only:
        job = SCENARIOS[name](a.seconds)
        log(f"{name}: {job['title']}, {job['config']['dispatches']} dispatches")
        try:
            S, wall = run_scenario(job, out_dir)
            np.savez(out_dir / f"{name}_series.npz", **S)
            fs = int(round(1.0 / S["frame_dt"]))
            transverse = S["bridge"][:, 1] if np.std(S["bridge"][:, 1]) >= np.std(S["bridge"][:, 2]) else S["bridge"][:, 2]
            half = len(transverse) // 2
            f1 = fundamental(transverse[half:], fs, float(S["f_nominal"]))
            output_errors = []
            for make in (lambda: [plot_waveforms(name, job["title"], S, f1, out_dir)],
                         lambda: [plot_energy(name, job["title"], S, out_dir)],
                         lambda: [plot_spectrogram(name, job["title"], S, out_dir)],
                         lambda: write_wavs(name, S, fs, out_dir)):
                try:
                    files += make()
                except Exception as e:
                    traceback.print_exc()
                    output_errors.append(f"{type(e).__name__}: {e}")
            keys = [k for k in S if k.startswith("energy_")]
            summary["scenarios"][name] = {
                "title": job["title"], "wall_minutes": wall / 60, "frames": int(len(S["bridge"])),
                "excitation_vertex": int(S["excitation_vertex"]),
                "fundamental_Hz_second_half": f1, "flexible_string_f1_Hz": float(S["f_nominal"]),
                "energy_above_rest_end_J": {ENERGY_LABELS[k[7:]]: float(S[k][-1]) for k in keys},
                "energy_above_rest_max_J": float(np.max(sum(S[k] for k in keys))),
                **({"output_errors": output_errors} if output_errors else {})}
            log(f"{name}: done in {wall / 60:.1f} min, f1 {f1:.2f} Hz")
        except Exception as e:
            traceback.print_exc()
            summary["scenarios"][name] = {"title": job["title"], "error": f"{type(e).__name__}: {e}"}
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    files.append(out_dir / "summary.json")
    if not a.no_email:
        send(files, summary, out_dir)
    log("all done")


if __name__ == "__main__":
    main()
