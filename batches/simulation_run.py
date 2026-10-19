import sys
import threading
import queue
import json
import pickle
import gzip
from pathlib import Path
from datetime import datetime
import numpy as np

from py_wgpu_fdm import Simulation
from utils import (generate_straight_rod, make_weights, damping_coefficients,
                   legacy_damping_coefficients, resolve_rod_params, twist_rate_damping)
from post_process import process_and_plot
import bowing
import plucking

# hammer_params entries that are not Simulation() arguments (set_hammer_friction)
HAMMER_CONTACT_KEYS = ("hammer_crown_radius", "hammer_felt_friction")

class SimulationRun:
    def __init__(self, run_id, rod_params, hammer_params, config, email_config, base_dir="simulation_results",
                 bow_params=None, pick_params=None):
        self.run_id = run_id
        self.rod_params = rod_params
        self.hammer_params = hammer_params
        # Bowed scenario (see bowing.configure_bow); the hammer stays disabled.
        self.bow_params = bow_params
        self.bow_derived = {}
        # Plucked scenario (see plucking.configure_pick); the hammer stays disabled.
        self.pick_params = pick_params
        self.pick_derived = {}
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

            dispatch, nodes, edges, bow_frames, surface_frames = item
            try:
                if not self.config.get("save_state", True):
                    # compact output (sweeps): bow records, the first edge's force (bridge force) and
                    # the contact nodes' displacement and velocity
                    j = (self.bow_derived["contact_nodes"] if self.bow_params is not None
                         else [self.pick_derived["pick_node"]])
                    np.savez_compressed(self.sim_dir / f"s_{dispatch:06d}.npz",
                                        bridge_force=np.asarray([fr[0][1][1] for fr in edges], np.float32),
                                        contact_displacement=np.asarray([[fr[i][0] for i in j] for fr in nodes], np.float32),
                                        contact_velocity=np.asarray([[fr[i][1] for i in j] for fr in nodes], np.float32))
                    nodes = edges = None
                if bow_frames is not None:
                    with gzip.open(self.sim_dir / f"b_{dispatch:06d}.pkl.gz", "wb", compresslevel=4) as f:
                        pickle.dump(bow_frames, f, protocol=pickle.HIGHEST_PROTOCOL)
                if surface_frames is not None:
                    with gzip.open(self.sim_dir / f"p_{dispatch:06d}.pkl.gz", "wb", compresslevel=4) as f:
                        pickle.dump(surface_frames, f, protocol=pickle.HIGHEST_PROTOCOL)
                # GZIP Compression with compresslevel=4 for fast write speeds
                n_file = self.sim_dir / f"n_{dispatch:06d}.pkl.gz"
                e_file = self.sim_dir / f"e_{dispatch:06d}.pkl.gz"

                if nodes is not None:
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
            "rod_derived": self.rod_derived,
            "bow_params": self.bow_params,
            "bow_derived": self.bow_derived,
            "pick_params": self.pick_params,
            "pick_derived": self.pick_derived,
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
        # Runs from before the variational force law carry no "force_law" entry.
        self.config.setdefault("force_law", "legacy")
        # rod_params may give the string's linear density instead of the winding packing factor.
        rod = resolve_rod_params(self.rod_params, self.config["node_count"], self.config["force_law"])
        nodes, edges, dl, mass, inertia, K_se, K_bt = generate_straight_rod(
            self.config["node_count"], **rod, force_law=self.config["force_law"]
        )
        
        self.rod_derived = {
            "m_node": mass,
            "inertia": inertia,
            "K_se": K_se,
            "K_bt": K_bt,
            "dl": dl,
            # Marks runs whose edges share one bit-identical reference spacing
            # (utils.reference_vectors); older runs used the float32 position
            # differences, which post-processing reproduces when this is absent.
            "uniform_reference_spacing": True,
            # Edge orientations are stored as the deviation p from the rest
            # orientation (q = q0 * p); post-processing rebuilds q from it.
            "orientation_representation": "relative_to_rest",
            "packing_factor": rod["packing_factor"],
            # rod_params may give the pitch instead of the tension (plain strings)
            "tension_force": rod["tension_force"],
        }
        self.tension = rod["tension_force"]

        weights = make_weights(
            self.config["node_count"], self.config["hammer_node"], self.config["hammer_width"]
        )

        if "damping" in self.config:
            c_lin, c_ang = damping_coefficients(self.config["damping"], mass, inertia, K_bt=K_bt,
                                                dl=dl, length=self.rod_params["tuned_length"],
                                                tension=self.tension,
                                                node_count=self.config["node_count"])
            d3 = twist_rate_damping(self.config["damping"], inertia, K_bt, dl,
                                    self.rod_params["tuned_length"])
        else:
            c_lin, c_ang = legacy_damping_coefficients(self.config["dampening"])
            d3 = 0.0
        # Recorded in parameters.json (config) for post-processing.
        self.config["damping_coefficients"] = {"linear": c_lin, "angular": list(c_ang),
                                               "twist_rate": d3}

        # Rust prints the selected GPU adapter on its own stdout; flush ours
        # first so the two streams don't interleave when output is redirected.
        sys.stdout.flush()
        self.sim = Simulation(
            nodes=nodes,
            edges=edges,
            hammer_weights=weights,
            **{k: v for k, v in self.hammer_params.items() if k not in HAMMER_CONTACT_KEYS},
            oversampling_factor=self.config["oversampling_factor"],
            chunk_size=self.config["chunk_size"],
            dt=self.config["dt"],
            dx=dl,
            mass=mass,
            stiffness_se=K_se,
            stiffness_bt=K_bt,
            inertia=inertia,
            clamp_offset=0,
            linear_damping=c_lin,
            angular_damping=c_ang,
            force_law=1 if self.config["force_law"] == "variational" else 0,
        )
        if d3 != 0.0:
            self.sim.set_twist_rate_damping(d3)
        self.sim.initialize()
        # self.sim.create_references()
        if self.bow_params is not None:
            run_steps = self.config["dispatches"] * self.config["chunk_size"] * self.config["oversampling_factor"]
            radius = self.rod_params["core_radius"] + 2.0 * self.rod_params["winding_radius"]
            self.bow_derived = bowing.configure_bow(self.sim, self.bow_params, self.config["node_count"],
                                                    self.config["dt"], run_steps, radius)
            self.bow_derived["string_radius"] = radius
        radius = self.rod_params["core_radius"] + 2.0 * self.rod_params["winding_radius"]
        if self.pick_params is not None:
            self.pick_derived = plucking.configure_pick(self.sim, self.pick_params, self.config["node_count"],
                                                        self.config["dt"], radius)
        crown = float(self.hammer_params.get("hammer_crown_radius", 0.0))
        felt_mu = float(self.hammer_params.get("hammer_felt_friction", 0.0))
        self.hammer_contact = crown > 0.0 or felt_mu > 0.0
        if self.hammer_contact:
            # felt friction on every node the hammer weights touch
            hammer_nodes = [int(i) for i in np.nonzero(weights > 0.0)[0]]
            self.sim.set_hammer_friction(hammer_nodes if felt_mu > 0.0 else [], True, radius, crown,
                                         felt_mu, felt_mu)
            self.rod_derived["hammer_friction_nodes"] = hammer_nodes if felt_mu > 0.0 else []

    def run(self):
        print(f"[{datetime.now()}] Starting run {self.run_id:03d} in {self.sim_dir.name}")
        self.build()
        
        # --- SAVE METADATA TO DISK ---
        self.save_metadata() 
        
        self.writer_thread.start()

        dispatches = self.config["dispatches"]

        bowed = self.bow_params is not None
        plucked = self.pick_params is not None
        self.sim.enable_hammer(not (bowed or plucked))
        for dispatch in range(dispatches):
            self.sim.compute()
            n_raw, e_raw = self.sim.save()
            # per frame: (normal force, bow velocity, step, per contact (sticking, [slip velocity, force,
            # impulse, bow work, friction work, normal work, temperature, heat power]))
            b_raw = self.sim.save_bow() if bowed else None
            # per frame: (pick record, hammer record), see Simulation.save_surface
            p_raw = self.sim.save_surface() if (plucked or self.hammer_contact) else None

            # --- CRITICAL ---
            # Deep-copy into pure Python lists NOW so the background 
            # thread doesn't read GPU memory during the next compute()
            n_safe = list(n_raw)
            e_safe = list(e_raw)

            self.write_queue.put((dispatch, n_safe, e_safe, b_raw, p_raw))

            if dispatch == 5 and not (bowed or plucked):
                print(f"[{datetime.now()}] Run {self.run_id:03d}: Disabling hammer.")
                self.sim.enable_hammer(False)

            if dispatch % 10 == 0 or dispatch == dispatches - 1:
                print(f"run {self.run_id:03d} {dispatch + 1}/{dispatches}")

        self.write_queue.join()
        self.write_queue.put(None)
        self.writer_thread.join()
        
        print(f"Run {self.run_id:03d} compute complete. Launching post-processing...")
        
        plot_thread = threading.Thread(
            target=process_and_plot,
            args=(
                self.sim_dir, self.run_id, self.config, self.email_config, 
                self.rod_derived["m_node"], self.rod_derived["dl"], self.rod_derived["inertia"], 
                self.rod_derived["K_se"], self.rod_derived["K_bt"], self.config["inspect_nodes"]
            ),
            kwargs={
                "tension": self.tension,
                "length": self.rod_params["tuned_length"],
                "rod_params": self.rod_params,
                "uniform_reference": True,
                "relative_orientations": True,
            },
            daemon=False
        )
        plot_thread.start()
        plot_thread.join()
