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

# Bowed-string showcase: Jargar Classic cello D (medium), steel core, chrome-steel wound.
# Sourced: outer diameter 0.98 mm and tension 14.0 kp (Presto Music product listing; Aitchison
# & Mnatzaganian string tension chart), construction (same sources). Definitions: open D3 =
# 146.83 Hz, 4/4 cello vibrating length 0.69 m, so the linear density at tension is
# T / (2 L f1)^2. Assumptions: core diameter 0.35 mm (inside the 0.30-0.40 mm range for
# solid steel cores of bowed strings, patent US 11367420); winding bending contribution mu as
# in run 002. Handbook values for steel. The winding packing factor follows from the linear
# density (utils.resolve_rod_params).
CELLO_D_TENSION = 14.0 * 9.80665
CELLO_D_LENGTH = 0.69
CELLO_D_F1 = 146.83
CELLO_D_JARGAR = {
    "tuned_length": CELLO_D_LENGTH,
    "tension_force": CELLO_D_TENSION,
    "linear_density": CELLO_D_TENSION / (2.0 * CELLO_D_LENGTH * CELLO_D_F1) ** 2,
    "core_radius": 0.35e-3 / 2.0,
    "winding_radius": (0.98e-3 / 2.0 - 0.35e-3 / 2.0) / 2.0,   # outer radius = core + 2 x wire radius
    "E_core": 2.0e11,
    "G_core": 7.9e10,
    "rho_core": 7.85e3,
    "E_winding": 2.0e11,
    "G_winding": 7.9e10,
    "rho_winding": 7.8e3,
    "packing_factor": None,
    "mu": 0.15,
    "twists": 0.0,
    # Measured torsional wave speed of a cello D string at nominal tension (Woodhouse 2003,
    # sec. 2.2, Dominant D; Mansour et al. 2016 use f_t = 758 Hz, i.e. 1046 m/s). The core
    # alone (G J) would give 567 m/s: the winding carries most of the torsional stiffness.
    "torsional_wave_speed": 1060.0,
}

# Transverse: constant Q = 500, the value Woodhouse (Acta Acustica 89, 2003, sec. 2.2) uses
# for a cello D string, matched at modes 1 and 10 (b1 + b2 k^2 cannot be constant-Q).
# Torsional: Q = 34, the average for steel-core cello D strings (Woodhouse & Loach 1999,
# as quoted by Mansour et al., Acta Acustica 102, 2016), constant over the measured modes;
# matched at torsional modes 1 and 10 by viscous + twist-rate damping (a viscous term alone
# gives every torsional mode the same decay rate, i.e. Q rising as n: 340 at mode 10).
CELLO_D_DAMPING = {"transverse_q": 500.0, "q_fit_modes": [1, 10],
                   "torsion_q": 34.0, "torsion_q_fit_modes": [1, 10]}

# Bowed cello D: 512 nodes (dl = 1.34 mm) and dt = 2e-7 s, the largest stable step for this string
# (the shear-rotation cutoff near 1.2 MHz limits it to about 2.7e-7 s); output at 50 kHz.
BOW_DT = 2e-7
BOW_SIM_CONFIG = {
    "node_count": 512,
    "dt": BOW_DT,
    "chunk_size": 512,
    "dispatches": 40,                       # 40 x 512 frames at 50 kHz = 0.41 s
    "hammer_node": 256,                     # unused (hammer disabled), kept for the post-processing
    "inspect_nodes": [51, 256, 400],
    "hammer_width": 4,
    "oversampling_factor": int(round(1.0 / (BOW_DT * 50_000))),
    "damping": CELLO_D_DAMPING,
    "force_law": "variational",
}

# Plucked guitar string: plain steel G3 .017" (0.4318 mm), D'Addario PL017, at a 25.5" (0.6477 m)
# scale. A solid steel wire needs no construction assumption: radius from the gauge, handbook steel
# values (as the cello core), GJ = G pi r^4 / 2 and the inertias of a solid circular section. The
# tension follows from the pitch (G3 = 196.00 Hz) with the rod's own mass (utils.resolve_rod_params):
# about 74 N (D'Addario's tension chart: 16.6 lb = 73.8 N for PL017 at G3, 25.5").
GUITAR_LENGTH = 25.5 * 0.0254
GUITAR_G_F1 = 196.00
GUITAR_G_PLAIN = {
    "tuned_length": GUITAR_LENGTH,
    "tension_force": None,
    "fundamental_frequency": GUITAR_G_F1,
    "core_radius": 0.017 * 0.0254 / 2.0,
    "winding_radius": 0.0,
    "E_core": 2.0e11,
    "G_core": 7.9e10,
    "rho_core": 7.85e3,
    "E_winding": 2.0e11,
    "G_winding": 7.9e10,
    "rho_winding": 7.85e3,
    "packing_factor": 0.0,
    "mu": 0.0,
    "twists": 0.0,
}

# Transverse: the loss model of Valette & Cuesta with the parameters Issanchou, Bilbao, Le Carrou,
# Touze & Doare (JSV 393, 2017, sec. 2.3.1 and 4.2) fitted to 36 measured modes of a D'Addario
# 0.43 mm plain steel electric guitar string (G3): loss angle delta_ve = 4.5e-3, 1/Q_te = 2.03e-4,
# air viscosity 1.8e-5 kg/(m s) and density 1.2 kg/m^3; matched at modes 1 and 10 by b1 + b2 k^2
# (utils.valette_q, damping_coefficients). Torsion (assumption: no measured torsional Q of a plain
# steel string found): constant Q_t = 1 / delta_ve, i.e. the same material loss angle in shear (no
# thermoelastic loss in pure shear, air drag on twisting neglected), matched at torsional modes 1 and
# 10 like the cello D.
GUITAR_DAMPING = {
    "transverse_valette": {"diameter": 0.017 * 0.0254, "delta_ve": 4.5e-3, "q_te_inv": 2.03e-4,
                           "eta_air": 1.8e-5, "rho_air": 1.2},
    "q_fit_modes": [1, 10],
    "torsion_q": 1.0 / 4.5e-3,
    "torsion_q_fit_modes": [1, 10],
}

# Plucked guitar G: 512 nodes (dl = 1.26 mm) and dt = 5e-8 s, below the stability limit 2/omega =
# 5.9e-8 s of the solid wire's shear-rotation cutoff (5.40 MHz); output at 50 kHz.
GUITAR_DT = 5e-8
GUITAR_SIM_CONFIG = {
    "node_count": 512,
    "dt": GUITAR_DT,
    "chunk_size": 512,
    "dispatches": 98,                       # 98 x 512 frames at 50 kHz = 1.0 s
    "hammer_node": 256,                     # unused (hammer disabled), kept for the post-processing
    "inspect_nodes": [102, 256, 384],
    "hammer_width": 4,
    "oversampling_factor": int(round(1.0 / (GUITAR_DT * 50_000))),
    "damping": GUITAR_DAMPING,
    "force_law": "variational",
}

# Bow (bowing.configure_bow): Woodhouse 2003 Table I bowing speed 0.05 m/s; beta and normal force
# inside the Schelleng range of this string (maximum force ~ 2 Z0 v_b / ((mu_s - mu_d) beta) = 0.8 N
# at beta = 0.1); 1 ms attack ramp.
BOW_PARAMS = {
    "friction": "curve",                    # "curve" (Eq. 2) or "thermal" (Eq. 1 + calibrated mu(T))
    "torsion": True,
    "beta": 0.1,
    "width_nodes": 1,                       # point contact; n nodes = width (n - 1) dl
    "normal_force": 0.4,
    "velocity": 0.05,
    "ramp_time": 1e-3,
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
    # Felt contact geometry and friction (simulation_run.build -> set_hammer_friction): crown radius
    # across the string (0: flat crown, the force acts along +z as before) and the felt-on-steel
    # friction coefficient (0: none). Assumed values with the same status as the winding bending
    # contribution "mu" (no measurement found; Izadbakhsh et al. assume 0.2 and 0.6 in a piano action
    # model). With both 0 the strike is bit-identical to the frictionless contact.
    "hammer_crown_radius": 0.0,
    "hammer_felt_friction": 0.0,
}

EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER"),
    "smtp_port": int(os.getenv("SMTP_PORT", 465)) if os.getenv("SMTP_PORT") else 465,
    "sender_email": os.getenv("SENDER_EMAIL"),
    "sender_password": os.getenv("SENDER_PASSWORD"),
    "receiver_email": os.getenv("RECEIVER_EMAIL")
}
