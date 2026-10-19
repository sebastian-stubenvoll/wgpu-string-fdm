"""Time a 1.5 s piano hammer strike (the run-002 string of run_batch.py) on this machine.

    python time_hammer.py [--seconds 1.5] [--json PATH] [--full OUT_DIR]

Default: the simulation only (compute + readback of every frame, as a production run pays it), no
chunk files, no post-processing. --full runs the normal SimulationRun pipeline (chunks written to
OUT_DIR, full post-processing with its usual plot email) and times compute and post-processing.
When the email variables are set (.env, see config.EMAIL_CONFIG), the timing report is emailed too.
"""
import argparse
import copy
import json
import math
import shutil
import smtplib
import tempfile
import time
from email.message import EmailMessage

import numpy as np

import simulation_run
from simulation_run import SimulationRun
from run_batch import RUN_002_CONFIG, RUN_002_ROD, RUN_002_HAMMER
from config import EMAIL_CONFIG


def job(seconds):
    config = copy.deepcopy(RUN_002_CONFIG)
    frame_time = config["dt"] * config["oversampling_factor"]
    dispatch_time = frame_time * config["chunk_size"]
    config["dispatches"] = math.ceil(seconds / dispatch_time - 1e-9)
    return config, dispatch_time


def stats(x):
    x = np.asarray(x)
    return {"total": float(x.sum()), "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}


def time_simulation(config, dispatch_time):
    base = tempfile.mkdtemp(prefix="time_hammer_")
    try:
        t0 = time.perf_counter()
        run = SimulationRun(run_id=2, rod_params=copy.deepcopy(RUN_002_ROD), hammer_params=copy.deepcopy(RUN_002_HAMMER),
                            config=config, email_config={}, base_dir=base)
        run.build()
        t_build = time.perf_counter() - t0
        sim, dispatches, hammer_node = run.sim, config["dispatches"], config["hammer_node"]
        sim.enable_hammer(True)
        t_compute, t_save, peak_z = [], [], 0.0
        t_loop = time.perf_counter()
        for d in range(dispatches):
            a = time.perf_counter()
            sim.compute()
            b = time.perf_counter()
            n_raw, e_raw = sim.save()
            nodes, edges = list(n_raw), list(e_raw)
            c = time.perf_counter()
            t_compute.append(b - a)
            t_save.append(c - b)
            peak_z = max(peak_z, max(abs(fr[hammer_node][0][2]) for fr in nodes))
            del nodes, edges, n_raw, e_raw
            if d == 5:
                sim.enable_hammer(False)          # as SimulationRun.run
            if d % 10 == 0 or d == dispatches - 1:
                done = time.perf_counter() - t_loop
                eta = done / (d + 1) * (dispatches - d - 1)
                print(f"dispatch {d + 1}/{dispatches}: {t_compute[-1]:.2f} s compute + {t_save[-1]:.2f} s readback, "
                      f"elapsed {done / 60:.1f} min, ETA {eta / 60:.1f} min", flush=True)
        t_total = time.perf_counter() - t0
    finally:
        shutil.rmtree(base, ignore_errors=True)

    steps = dispatches * config["chunk_size"] * config["oversampling_factor"]
    loop = sum(t_compute) + sum(t_save)
    return {
        "mode": "simulation only",
        "node_count": config["node_count"], "dt": config["dt"], "oversampling_factor": config["oversampling_factor"],
        "dispatches": dispatches, "simulated_seconds": dispatches * dispatch_time, "solver_steps": steps,
        "build_seconds": t_build, "wall_seconds": t_total,
        "first_dispatch_seconds": t_compute[0] + t_save[0],
        "compute_per_dispatch": stats(t_compute), "readback_per_dispatch": stats(t_save),
        "steps_per_second": steps / loop,
        "wall_seconds_per_simulated_second": loop / (dispatches * dispatch_time),
        "hammer_node_peak_abs_uz_m": peak_z,
    }


def time_full(config, dispatch_time, out_dir):
    post = {}
    original = simulation_run.process_and_plot

    def timed_post(*args, **kwargs):
        post["start"] = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            post["end"] = time.perf_counter()

    simulation_run.process_and_plot = timed_post
    try:
        t0 = time.perf_counter()
        SimulationRun(run_id=2, rod_params=copy.deepcopy(RUN_002_ROD), hammer_params=copy.deepcopy(RUN_002_HAMMER),
                      config=config, email_config=EMAIL_CONFIG, base_dir=out_dir).run()
        t1 = time.perf_counter()
    finally:
        simulation_run.process_and_plot = original
    return {
        "mode": "full pipeline", "out_dir": out_dir, "dispatches": config["dispatches"],
        "simulated_seconds": config["dispatches"] * dispatch_time, "wall_seconds": t1 - t0,
        "build_compute_write_seconds": post["start"] - t0, "post_processing_seconds": post["end"] - post["start"],
    }


def email_report(report):
    """Send the report with the same settings and transport as post_process.send_email."""
    cfg = EMAIL_CONFIG
    if not (cfg.get("sender_email") and cfg.get("sender_password") and cfg.get("receiver_email")):
        print("Email not configured - skipping the timing report email.")
        return
    msg = EmailMessage()
    msg["Subject"] = (f"Hammer timing ({report['mode']}): {report['simulated_seconds']:.3f} s simulated "
                      f"in {report['wall_seconds'] / 60:.1f} min")
    msg["From"] = cfg["sender_email"]
    msg["To"] = cfg["receiver_email"]
    msg.set_content(json.dumps(report, indent=2))
    try:
        with smtplib.SMTP_SSL(cfg["smtp_server"], cfg["smtp_port"]) as smtp:
            smtp.login(cfg["sender_email"], cfg["sender_password"])
            smtp.send_message(msg)
        print("Timing report emailed.")
    except Exception as e:
        print(f"Failed to send the timing report email: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=1.5, help="simulated time (rounded up to whole dispatches)")
    p.add_argument("--json", help="also write the report to this file")
    p.add_argument("--full", metavar="OUT_DIR", help="run the full pipeline (chunks + post-processing) into OUT_DIR")
    a = p.parse_args()
    config, dispatch_time = job(a.seconds)
    report = time_full(config, dispatch_time, a.full) if a.full else time_simulation(config, dispatch_time)
    print(json.dumps(report, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2)
    email_report(report)


if __name__ == "__main__":
    main()
