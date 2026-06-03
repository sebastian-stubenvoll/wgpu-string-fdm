import json
import copy
import pickle
import threading
import queue

from pathlib import Path
from datetime import datetime

import numpy as np
from py_wgpu_fdm import Simulation


class SimulationRun:

    def __init__(
        self,
        run_id,
        rod_params,
        config,
        base_dir="simulation_results",
    ):
        self.run_id = run_id
        self.rod_params = rod_params
        self.config = config

        self.sim_dir = (
            Path(base_dir)
            / f"run_{run_id:03d}"
        )

        self.sim_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.write_queue = queue.Queue(maxsize=4)
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
        )

    @staticmethod
    def generate_straight_rod(
        node_count,
        tuned_length,
        tension_force,
        core_radius,
        winding_radius,
        E_core,
        G_core,
        rho_core,
        E_winding,
        G_winding,
        rho_winding,
        packing_factor,
        mu,
        twists,
    ):
        core_area = np.pi * core_radius**2
        EA = E_core * core_area
        dilatation = tension_force / EA
        rest_length = tuned_length / (1 + dilatation)
        dl = rest_length / float(node_count - 1)

        core_mass = float(rho_core * core_area * dl)
        winding_area = np.pi * ((core_radius + 2.0 * winding_radius)**2 - core_radius**2)
        winding_mass = float(rho_winding * winding_area * dl * packing_factor)
        mass = core_mass + winding_mass

        I_core = rho_core * core_radius**4 / 4.0
        I_winding = rho_winding * ((core_radius + 2.0 * winding_radius)**4 - core_radius**4) * packing_factor / 4.0
        I1 = I_core + I_winding
        I3 = I1 * 2.0
        inertia = (np.pi * dl * np.array([I1, I1, I3], dtype=np.float32))

        alpha = 4.0 / 3.0
        I_c = np.pi * core_radius**4 / 4.0
        J = np.pi * core_radius**4 / 2.0
        I_w = np.pi * ((core_radius + 2.0 * winding_radius)**4 - core_radius**4) * packing_factor / 4.0
        EI = E_core * I_c + (E_winding * I_w * mu)

        K_se = np.array([G_core * core_area * alpha,
                         G_core * core_area * alpha,
                         E_core * core_area], dtype=np.float32)

        K_bt = np.array([EI, EI, G_core * J], dtype=np.float32)

        x = np.linspace(0.0, tuned_length, node_count)
        reference_positions = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1).astype(np.float32)

        # orientation init (unchanged)
        def twist_quaternion(angle):
            half = 0.5 * angle
            return np.array([np.sin(half), 0.0, 0.0, np.cos(half)])

        def quat_mul(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            return np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ])

        angle = np.pi / 2
        q_base = np.array([0.0, np.sin(0.5 * angle), 0.0, np.cos(0.5 * angle)])

        total_edges = node_count - 1
        total_angle = 2.0 * np.pi * twists

        orientations = [
            quat_mul(twist_quaternion(total_angle * (i / total_edges)), q_base)
            for i in range(total_edges)
        ]
        orientations = [q / np.linalg.norm(q) for q in orientations]

        ref_vecs = np.array(
            [reference_positions[i+1] - reference_positions[i] for i in range(node_count - 1)],
            dtype=np.float32
        )

        edges = [(o, v, np.zeros(3), np.zeros(3)) for o, v in zip(orientations, ref_vecs)]
        edges.append((edges[-1][0], np.zeros(3), np.zeros(3), np.zeros(3)))

        nodes = [
            (np.zeros(3), np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
            for _ in reference_positions
        ]

        return reference_positions, nodes, edges, dl, mass, inertia, K_se, K_bt

    @staticmethod
    def make_weights(node_count, node_index=None, width_count=4):
        """
        Gaussian spatial weighting around a node index.
        Used for hammer/contact forcing distribution.
        """

        data = np.zeros((node_count, 4), dtype=np.float32)
        x = np.arange(node_count, dtype=np.float32)

        if node_index is None:
            node_index = node_count // 2

        sigma = width_count / 2.355
        if sigma == 0:
            sigma = 1e-5

        gauss_vals = np.exp(-((x - node_index) ** 2) / (2 * sigma ** 2))

        mask = np.abs(x - node_index) <= width_count

        # channel 2 is your active weight channel (same as before)
        data[:, 2] = gauss_vals * mask

        return data


    @staticmethod
    def make_force_curve(dt, duration):
        """
        Generates a synthetic hammer/excitation force curve:
        - asymmetric Gaussian pulse + secondary excitation
        - used for modal excitation of the rod
        """

        scale = duration / 4.0
        t = np.arange(0, duration, dt)

        mu1 = 0.50 * scale
        amp1 = 15.0
        sig1_left = 0.1 * scale
        sig1_right = 0.75 * scale

        mu2 = 3.0 * scale
        amp2 = 8.5
        sig2 = 0.2 * scale

        force = np.zeros_like(t)

        mask_left = t <= mu1

        force[mask_left] += amp1 * np.exp(
            -0.5 * ((t[mask_left] - mu1) / sig1_left) ** 2
        )

        force[~mask_left] += amp1 * np.exp(
            -0.5 * ((t[~mask_left] - mu1) / sig1_right) ** 2
        )

        force += amp2 * np.exp(-0.5 * ((t - mu2) / sig2) ** 2)

        return t, force

    def _writer_loop(self):

        while True:
            item = self.write_queue.get()
            if item is None:
                self.write_queue.task_done()
                break

            dispatch, nodes, edges = item

            with open(self.sim_dir / f"n_{dispatch:06d}.pkl","wb") as f:
                pickle.dump(nodes, f, protocol=pickle.HIGHEST_PROTOCOL,)
            
            with open(self.sim_dir / f"e_{dispatch:06d}.pkl", "wb") as f:
                pickle.dump(edges, f, protocol=pickle.HIGHEST_PROTOCOL,)

            self.write_queue.task_done()

    def save_metadata(self):
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,

            "simulation": {
                "node_count": self.config["node_count"],
                "dt": self.config["dt"],
                "chunk_size": self.config["chunk_size"],
                "dispatches": self.config["dispatches"],
                # "sample_rate": self.config["sample_rate"],
                "oversampling_factor": self.config["oversampling_factor"],
                "hammer_node": self.config["hammer_node"],
                "hammer_width": self.config["hammer_width"],
                "dampening": self.config["dampening"],
            },

            "rod": self.rod_params,
        }

        with open(self.sim_dir / "parameters.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def build(self):

        (ref_pos, nodes, edges, dl, mass, inertia, K_se, K_bt,) = SimulationRun.generate_straight_rod(
            self.config["node_count"],
            **self.rod_params,
        )

        weights = SimulationRun.make_weights(
            self.config["node_count"],
            self.config["hammer_node"],
            self.config["hammer_width"],
        )

        self.sim = Simulation(
            nodes=nodes,
            edges=edges,
            hammer_weights=weights,
            oversampling_factor=self.config[
                "oversampling_factor"
            ],
            chunk_size=self.config[
                "chunk_size"
            ],
            dt=self.config["dt"],
            dx=dl,
            mass=mass,
            stiffness_se=K_se,
            stiffness_bt=K_bt,
            inertia=inertia,
            clamp_offset=0,
            dampening=self.config[
                "dampening"
            ],
        )

        self.sim.initialize()


    
    def excite(self):

        _, force = SimulationRun.make_force_curve(
            self.config["dt"],
            0.004,
        )

        for f in force[::10]:
            self.sim.hammer(
                self.config["hammer_node"],
                f,
            )



    def run(self):

        print(
            f"[{datetime.now()}] "
            f"Starting run {self.run_id:03d}"
        )

        self.save_metadata()
        self.build()
        self.excite()
        self.writer_thread.start()
        dispatches = self.config[
            "dispatches"
        ]

        for dispatch in range(dispatches):
            self.sim.compute()
            n, e = self.sim.save()

            n = copy.deepcopy(n)
            e = copy.deepcopy(e)

            self.write_queue.put((dispatch, n, e,))

            if dispatch % 10 == 0:
                print(
                    f"run {self.run_id:03d} "
                    f"{dispatch}/{dispatches}"
                )

        self.write_queue.join()
        self.write_queue.put(None)
        self.writer_thread.join()
        print(f"Run {self.run_id:03d} complete.")


config = {
    "node_count": 50,
    "dt": 2e-7,
    "chunk_size": 512,
    "dispatches": 1,
    "hammer_node": 10,
    "hammer_width": 8,
    "oversampling_factor": int(
        1.0 / (2e-7 * 50_000)
    ),
    "dampening": [1e-6, 1e-6],
}

base_params = {
    "tuned_length": 1.22,        # m
    "tension_force": 600,        # N
    "core_radius": 6.0e-4,       # m
    "winding_radius": 4.0e-4,    # m
    "E_core": 2.07e11,           # Pa
    "G_core": 8.0e10,            # Pa
    "rho_core": 7.85e3,          # kg/m^3
    "E_winding": 1.1e11,         # Pa
    "G_winding": 4.1e10,         # Pa
    "rho_winding": 8.96e3,       # kg/m^3
    "packing_factor": 0.8,       # dimensionless
    "mu": 0.15,                  # dimensionless
    "twists": 0.0,               # dimensionless
}

batch = [ base_params ]


for i, params in enumerate(batch):

    SimulationRun(
        run_id=i,
        rod_params=params,
        config=config,
    ).run()
