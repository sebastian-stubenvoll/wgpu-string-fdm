import os
import copy
from itertools import product
from simulation_run import SimulationRun
from config import SIM_CONFIG, ROD_PARAMS, HAMMER_PARAMS, EMAIL_CONFIG

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


def main():
    batch_jobs = [
        {"run_id": 2, "config": RUN_002_CONFIG,
         "rod": copy.deepcopy(RUN_002_ROD), "hammer": copy.deepcopy(RUN_002_HAMMER)},
    ]

    for params in batch_jobs:
        sim = SimulationRun(
            run_id=params["run_id"],
            rod_params=params["rod"],
            hammer_params=params["hammer"],
            config=params["config"],
            email_config=EMAIL_CONFIG,
            base_dir=OUTPUT_DIR
        )
        sim.run()

if __name__ == "__main__":
    main()
