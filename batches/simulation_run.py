import json
import threading
import queue
from pathlib import Path
from datetime import datetime
import numpy as np

from py_wgpu_fdm import Simulation
from utils import generate_straight_rod, make_weights

class SimulationRun:
    def __init__(self, run_id, rod_params, hammer_params, config, base_dir="simulation_results"):
        self.run_id = run_id
        self.rod_params = rod_params
        self.hammer_params = hammer_params
        self.config = config

        self.sim_dir = Path(base_dir) / f"run_{run_id:03d}"
        self.sim_dir.mkdir(parents=True, exist_ok=True)

        self.write_queue = queue.Queue(maxsize=128)
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)

    def _writer_loop(self):
        while True:
            item = self.write_queue.get()
            if item is None:
                self.write_queue.task_done()
                break

            dispatch, nodes, edges = item
            np.save(self.sim_dir / f"n_{dispatch:06d}.npy", np.array(nodes, dtype=object))
            np.save(self.sim_dir / f"e_{dispatch:06d}.npy", np.array(edges, dtype=object))
            self.write_queue.task_done()

    def save_metadata(self):
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "simulation": self.config,
            "rod": self.rod_params,
            "hammer": self.hammer_params,
        }
        with open(self.sim_dir / "parameters.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def build(self):
        ref_pos, nodes, edges, dl, mass, inertia, K_se, K_bt = generate_straight_rod(
            self.config["node_count"], **self.rod_params
        )

        weights = make_weights(
            self.config["node_count"], self.config["hammer_node"], self.config["hammer_width"]
        )

        self.sim = Simulation(
            nodes=nodes,
            edges=edges,
            hammer_weights=weights,
            
            # --- The new physical hammer model parameters ---
            **self.hammer_params, 
            
            oversampling_factor=self.config["oversampling_factor"],
            chunk_size=self.config["chunk_size"],
            dt=self.config["dt"],
            dx=dl,
            mass=mass,
            stiffness_se=K_se,
            stiffness_bt=K_bt,
            inertia=inertia,
            clamp_offset=0,
            dampening=self.config["dampening"],
        )
        self.sim.initialize()

    def run(self):
        print(f"[{datetime.now()}] Starting run {self.run_id:03d}")
        self.save_metadata()
        self.build()
        self.writer_thread.start()

        dispatches = self.config["dispatches"]

        # --- Main simulation phase (Hammer interaction is handled natively) ---
        for dispatch in range(dispatches):
            self.sim.compute()
            n, e = self.sim.save()
            self.write_queue.put((dispatch, n, e))

            if dispatch % 10 == 0:
                print(f"run {self.run_id:03d} {dispatch}/{dispatches}")

        self.write_queue.join()
        self.write_queue.put(None)
        self.writer_thread.join()
        
        print(f"Run {self.run_id:03d} complete.")
