import os
from dotenv import load_dotenv

load_dotenv()

SIM_CONFIG = {
    "node_count": 1024,
    "dt": 4e-7, # 2e-7 is every so slightly unstable
    "chunk_size": 512,
    "dispatches": 200,
    "hammer_node": 25,
    "inspect_nodes": [25],
    "hammer_width": 1,
    "oversampling_factor": int(1.0 / (4e-7 * 50_000)),
    "dampening": [1e-6, 1e-6],
}

ROD_PARAMS = {
    "tuned_length": 1.602,
    "tension_force": 915.0,
    "core_radius": 5.5e-4,
    "winding_radius": 4.5e-4,
    "E_core": 2.07e11,
    "G_core": 8.0e10,
    "rho_core": 7.85e3,
    "E_winding": 1.1e11,
    "G_winding": 4.1e10,
    "rho_winding": 8.96e3,
    "packing_factor": 0.8,
    "mu": 0.15,
    "twists": 0.0,
}

HAMMER_PARAMS = {
    "hammer_displacement": -0.05,
    "hammer_velocity": 5.0,
    "hammer_mass": 0.0102,
    "hammer_stiffness": 500_000_000.0,
    "hammer_exponent": 2.3,
    "hammer_hysteresis_factor": 0.1,
    "hammer_relaxation_time": 0.000014,
}

EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER"),
    "smtp_port": int(os.getenv("SMTP_PORT", 465)) if os.getenv("SMTP_PORT") else 465,
    "sender_email": os.getenv("SENDER_EMAIL"),
    "sender_password": os.getenv("SENDER_PASSWORD"),
    "receiver_email": os.getenv("RECEIVER_EMAIL")
}
