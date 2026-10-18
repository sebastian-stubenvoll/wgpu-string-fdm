import os
from dotenv import load_dotenv

load_dotenv()

dt = 2.5e-7 # 4e-7 is every so slightly unstable

SIM_CONFIG = {
    "node_count": 1024,
    "dt": dt, 
    "chunk_size": 512,
    "dispatches": 150,
    "hammer_node": 25,
    "inspect_nodes": [25, 512, 800],
    "hammer_width": 15,
    "oversampling_factor": int(1.0 / (dt * 50_000)),
    # Damping as physical quantities (see utils.damping_coefficients): transverse mode n decays
    # at b1 + b2 k_n^2, measured for piano note C2 (Bensa et al., JASA 114(2), 2003, Table I);
    # torsion from the quality factor of wound steel bass strings (Wollman, Smith & Wolfe,
    # ISMA 2010: Q ~ 15-20). Runs before 2026-10-18 used the equivalent of
    # {"b1": 1.0454e-2, "b2": 1.0454e-2, "torsion_rate": 9.2713e3} (run-002 rod).
    "damping": {"b1": 0.25, "b2": 7.5e-5, "torsion_q": 17.5},
    # "variational": the internal force is the gradient of the stored energy (includes the
    # stretch derivative of the cross-section scaling eta); "legacy": runs before 2026-10-18.
    "force_law": "variational",
}

ROD_PARAMS = {
    "tuned_length": 1.602,
    "tension_force": 915.0,
    "core_radius": 3.375e-4,
    "winding_radius": 3.25e-4,
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
    "hammer_offset_y": 0.0,
}

EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER"),
    "smtp_port": int(os.getenv("SMTP_PORT", 465)) if os.getenv("SMTP_PORT") else 465,
    "sender_email": os.getenv("SENDER_EMAIL"),
    "sender_password": os.getenv("SENDER_PASSWORD"),
    "receiver_email": os.getenv("RECEIVER_EMAIL")
}
