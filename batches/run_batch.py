import os
from simulation_run import SimulationRun
from config import SIM_CONFIG, ROD_PARAMS, HAMMER_PARAMS, EMAIL_CONFIG

# Change this to whatever absolute or relative path you prefer
OUTPUT_DIR = "simulation_results" 

def main():
    batch_jobs = [
        {"rod": ROD_PARAMS, "hammer": HAMMER_PARAMS}
    ]

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
