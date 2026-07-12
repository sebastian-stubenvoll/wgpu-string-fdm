import os
import copy
from itertools import product
from simulation_run import SimulationRun
from config import SIM_CONFIG, ROD_PARAMS, HAMMER_PARAMS, EMAIL_CONFIG

OUTPUT_DIR = "/simulation_data" 

def main():
    batch_jobs = list()

    mu_vars = [ 0.15, 0.3 ]
    hammer_velocities = [ 2.5, 6.0 ]
    twists = [ 0.0, 3.0 ]
    offset = [ 0.0, 7.0e-3]

    for mu, hv, tw, o in product(mu_vars, hammer_velocities, twists, offset):
        base = {"rod": copy.deepcopy(ROD_PARAMS), "hammer": copy.deepcopy(HAMMER_PARAMS)}
        base["rod"]["mu"] = mu
        base["hammer"]["hammer_velocity"] = hv
        base["hammer"]["hammer_offset_y"] = o
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
