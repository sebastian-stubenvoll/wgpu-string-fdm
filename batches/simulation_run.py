import threading
import queue
import json
import pickle
import gzip
from pathlib import Path
from datetime import datetime
import numpy as np

from py_wgpu_fdm import Simulation
from utils import generate_straight_rod, make_weights
from post_process import process_and_email

class SimulationRun:
    def __init__(self, run_id, rod_params, hammer_params, config, email_config, base_dir="simulation_results"):
        self.run_id = run_id
        self.rod_params = rod_params
        self.hammer_params = hammer_params
        self.config = config
        self.email_config = email_config

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.sim_dir = Path(base_dir) / f"{timestamp_str}_run_{run_id:03d}"
        self.sim_dir.mkdir(parents=True, exist_ok=True)

        # Strict backpressure: only 2 chunks allowed in RAM at a time.
        self.write_queue = queue.Queue(maxsize=2)
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)

        self.rod_derived = {}

    def _writer_loop(self):
        while True:
            item = self.write_queue.get()
            if item is None:
                self.write_queue.task_done()
                break

            dispatch, nodes, edges = item
            try:
                # GZIP Compression with compresslevel=4 for fast write speeds
                n_file = self.sim_dir / f"n_{dispatch:06d}.pkl.gz"
                e_file = self.sim_dir / f"e_{dispatch:06d}.pkl.gz"

                with gzip.open(n_file, "wb", compresslevel=4) as f:
                    pickle.dump(nodes, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                with gzip.open(e_file, "wb", compresslevel=4) as f:
                    pickle.dump(edges, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Explicitly severe references so Python immediately reclaims the RAM
                del nodes
                del edges
                del item

            except Exception as e:
                print(f"Write error on dispatch {dispatch}: {e}")

            self.write_queue.task_done()

    def save_metadata(self):
        """Saves all run parameters to a parameters.json file"""
        metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "rod_params": self.rod_params,
            "hammer_params": self.hammer_params,
            "rod_derived": self.rod_derived
        }
        
        # Helper function to convert numpy arrays into standard JSON lists
        def np_encoder(obj):
            if isinstance(obj, np.generic): 
                return obj.item()
            if isinstance(obj, np.ndarray): 
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(self.sim_dir / "parameters.json", "w") as f:
            json.dump(metadata, f, default=np_encoder, indent=4)

    def build(self):
        nodes, edges, dl, mass, inertia, K_se, K_bt = generate_straight_rod(
            self.config["node_count"], **self.rod_params
        )
        
        self.rod_derived = {
            "m_node": mass,
            "inertia": inertia,
            "K_se": K_se,
            "K_bt": K_bt,
            "dl": dl
        }

        weights = make_weights(
            self.config["node_count"], self.config["hammer_node"], self.config["hammer_width"]
        )

        self.sim = Simulation(
            nodes=nodes,
            edges=edges,
            hammer_weights=weights,
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
        print(f"[{datetime.now()}] Starting run {self.run_id:03d} in {self.sim_dir.name}")
        self.build()
        
        # --- SAVE METADATA TO DISK ---
        self.save_metadata() 
        
        self.writer_thread.start()

        dispatches = self.config["dispatches"]

        self.sim.enable_hammer(True)
        for dispatch in range(dispatches):
            self.sim.compute()
            n_raw, e_raw = self.sim.save()

            # --- CRITICAL ---
            # Deep-copy into pure Python lists NOW so the background 
            # thread doesn't read GPU memory during the next compute()
            n_safe = list(n_raw)
            e_safe = list(e_raw)

            self.write_queue.put((dispatch, n_safe, e_safe))

            if dispatch == 1:
                print(f"[{datetime.now()}] Run {self.run_id:03d}: Disabling hammer.")
                self.sim.enable_hammer(False)

            if dispatch % 10 == 0 or dispatch == dispatches - 1:
                print(f"run {self.run_id:03d} {dispatch + 1}/{dispatches}")

        self.write_queue.join()
        self.write_queue.put(None)
        self.writer_thread.join()
        
        print(f"Run {self.run_id:03d} compute complete. Launching post-processing...")
        
        plot_thread = threading.Thread(
            target=process_and_email,
            args=(
                self.sim_dir, self.run_id, self.config, self.email_config, 
                self.rod_derived["m_node"], self.rod_derived["inertia"], 
                self.rod_derived["K_se"], self.rod_derived["K_bt"], self.rod_derived["dl"]
            ),
            daemon=False
        )
        plot_thread.start()
