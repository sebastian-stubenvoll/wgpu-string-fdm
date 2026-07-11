import os
from itertools import product
from simulation_run import SimulationRun
from config import SIM_CONFIG, ROD_PARAMS, HAMMER_PARAMS, EMAIL_CONFIG

OUTPUT_DIR = "/simulation_data" 

def main():
    batch_jobs = list()

    mu_vars = [ 0.15, 0.2, 0.25 ]
    hammer_velocities = [ 6.0 ]
    twists = [ 0.0, 3.0 ]

    for mu, hv, tw in product(mu_vars, hammer_velocities, twists):
        base = {"rod": ROD_PARAMS, "hammer": HAMMER_PARAMS}
        base["rod"]["mu"] = mu
        base["hammer"]["hammer_velocity"] = hv
        base["hammer"]["hammer_offset_y"] = 7.0e-4
        base["rod"]["twists"] = tw
        batch_jobs.append(base)

    for i, params in enumerate(batch_jobs):
        sim = SimulationRun(
            run_id=i,
            rod_params=params["rod"],
            hammer_params=params["hammer"],
            config=SIM_CONFIG,
            email_config=EMAIL_CONFIG,
            base_dir=OUTPUT_DIR 
        )
        sim.run()

if __name__ == "__main__":
    main()
