from simulation_run import SimulationRun
from config import SIM_CONFIG, ROD_PARAMS, HAMMER_PARAMS

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
        )
        sim.run()

if __name__ == "__main__":
    main()
