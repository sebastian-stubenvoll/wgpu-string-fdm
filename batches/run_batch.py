import os
from simulation_run import SimulationRun
from config import SIM_CONFIG, ROD_PARAMS, HAMMER_PARAMS, EMAIL_CONFIG

OUTPUT_DIR = "/simulation_data" 

def main():
    batch_jobs = [
        {"rod": ROD_PARAMS, "hammer": HAMMER_PARAMS},
        {"rod": ROD_PARAMS, "hammer": HAMMER_PARAMS},
        {"rod": ROD_PARAMS, "hammer": HAMMER_PARAMS},
        {"rod": ROD_PARAMS, "hammer": HAMMER_PARAMS}
    ]

    mu_vars = [ 0.1, 0.15, 0.2, 0.25 ]

    for i, mu_v in enumerate(mu_vars):
        batch_jobs[i]["rod"]["mu"] = mu_v

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
