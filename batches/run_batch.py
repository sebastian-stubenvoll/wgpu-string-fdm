import os
import sys
import copy
from itertools import product
from simulation_run import SimulationRun
from config import (SIM_CONFIG, ROD_PARAMS, HAMMER_PARAMS, EMAIL_CONFIG, BOW_SIM_CONFIG, BOW_PARAMS,
                    CELLO_D_JARGAR, GUITAR_SIM_CONFIG, GUITAR_G_PLAIN)
from plucking import PICK_PARAMS

OUTPUT_DIR = "/simulation_data"

# Re-run of run 002 from the 2026-07-29 batch
# (cosserat-string-sim-data/2026-07-29_20-08-13_run_002/parameters.json),
# to compare against it after the reference-spacing fix in utils.py.
RUN_002_DT = 1.25e-7
RUN_002_CONFIG = {
    **SIM_CONFIG,
    "dt": RUN_002_DT,
    "oversampling_factor": int(1.0 / (RUN_002_DT * 50_000)),  # 160, as in that run
    "hammer_width": 25,
}
RUN_002_ROD = {
    **ROD_PARAMS,
    "tuned_length": 0.95,
    "tension_force": 800.0,
    "core_radius": 4.925e-4,
    "winding_radius": 5.08e-4,
    "mu": 0.15,
    "twists": 3.0,
}
RUN_002_HAMMER = {
    **HAMMER_PARAMS,
    "hammer_velocity": 6.0,
    "hammer_offset_y": 0.0,
}


def bow_job(run_id, config=None, **bow):
    return {"run_id": run_id, "config": copy.deepcopy(config or BOW_SIM_CONFIG),
            "rod": copy.deepcopy(CELLO_D_JARGAR), "hammer": copy.deepcopy(HAMMER_PARAMS),
            "bow": {**BOW_PARAMS, **bow}}


def bow_showcase_jobs():
    """Bowed cello D at beta 0.1, 0.4 N, 0.05 m/s (full post-processing): both friction models with
    torsion on and inhibited (point contact), and a 9-node (10.7 mm) contact with torsion on."""
    jobs, run_id = [], 100
    for friction, torsion, width in [("curve", True, 1), ("curve", False, 1), ("thermal", True, 1),
                                     ("thermal", False, 1), ("curve", True, 9), ("thermal", True, 9)]:
        jobs.append(bow_job(run_id, friction=friction, torsion=torsion, width_nodes=width))
        run_id += 1
    return jobs


# Schelleng diagram: bow position and normal force grids (log-spaced), bow speed 0.05 m/s.
SCHELLENG_BETAS = [0.03, 0.045, 0.067, 0.1, 0.15, 0.2]
SCHELLENG_FORCES = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]


def schelleng_jobs(frictions=("curve", "thermal"), torsions=(True, False)):
    """One 0.41 s run per (friction, torsion, beta, force), point contact, compact output and the
    bow analysis only (bow_summary.json per run; collect with schelleng.py)."""
    config = {**BOW_SIM_CONFIG, "save_state": False, "post_process": "bow_only"}
    jobs, run_id = [], 1000
    for friction, torsion, beta, force in product(frictions, torsions, SCHELLENG_BETAS, SCHELLENG_FORCES):
        job = bow_job(run_id, config=config, friction=friction, torsion=torsion, beta=beta,
                      normal_force=force)
        job["email"] = False          # 168 runs: no email per run
        jobs.append(job)
        run_id += 1
    return jobs


def pluck_job(run_id, config=None, **pick):
    return {"run_id": run_id, "config": copy.deepcopy(config or GUITAR_SIM_CONFIG),
            "rod": copy.deepcopy(GUITAR_G_PLAIN), "hammer": copy.deepcopy(HAMMER_PARAMS),
            "pick": {**PICK_PARAMS, **pick}}


def pluck_showcase_jobs():
    """Plectrum-plucked plain steel G3 (1 s, full post-processing), with torsion in the contact on
    and inhibited."""
    return [pluck_job(200, torsion=True), pluck_job(201, torsion=False)]


JOB_SETS = {
    "run002": lambda: [{"run_id": 2, "config": RUN_002_CONFIG,
                        "rod": copy.deepcopy(RUN_002_ROD), "hammer": copy.deepcopy(RUN_002_HAMMER)}],
    "bow_showcase": bow_showcase_jobs,
    "schelleng": schelleng_jobs,
    "pluck_showcase": pluck_showcase_jobs,
}


def main():
    """python run_batch.py [run002 | bow_showcase | schelleng | pluck_showcase]  (default run002)"""
    batch_jobs = JOB_SETS[sys.argv[1] if len(sys.argv) > 1 else "run002"]()

    for params in batch_jobs:
        sim = SimulationRun(
            run_id=params["run_id"],
            rod_params=params["rod"],
            hammer_params=params["hammer"],
            config=params["config"],
            email_config=EMAIL_CONFIG if params.get("email", True) else {},
            base_dir=OUTPUT_DIR,
            bow_params=params.get("bow"),
            pick_params=params.get("pick"),
        )
        sim.run()

if __name__ == "__main__":
    main()
