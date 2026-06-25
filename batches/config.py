SIM_CONFIG = {
    "node_count": 50,
    "dt": 2e-7,
    "chunk_size": 512,
    "dispatches": 10,
    "hammer_node": 10,
    "hammer_width": 8,
    "oversampling_factor": int(1.0 / (2e-7 * 50_000)),
    "dampening": [1e-6, 1e-6],
}

ROD_PARAMS = {
    "tuned_length": 1.1602,      # m
    "tension_force": 915.0,      # N
    "core_radius": 6.0e-4,       # m
    "winding_radius": 5.0e-4,    # m
    "E_core": 2.07e11,           # Pa
    "G_core": 8.0e10,            # Pa
    "rho_core": 7.85e3,          # kg/m^3
    "E_winding": 1.1e11,         # Pa
    "G_winding": 4.1e10,         # Pa
    "rho_winding": 8.96e3,       # kg/m^3
    "packing_factor": 0.8,       # 1
    "mu": 0.15,                  # 1
    "twists": 0.0,               # 1
}

HAMMER_PARAMS = {
    "hammer_displacement": -0.05,
    "hammer_velocity": 5.0,
    "hammer_mass": 0.01,
    "hammer_stiffness": 500_000_000.0,
    "hammer_exponent": 2.3,
    "hammer_hysteresis_factor": 0.1,
    "hammer_relaxation_time": 0.000014,
}
