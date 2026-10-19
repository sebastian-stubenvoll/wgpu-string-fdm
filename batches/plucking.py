"""Plectrum pluck: parameters and set-up of the shader's rigid pick contact (see struct Pick and
pick_contact_body in src/shaders/cosserat.wgsl).

Geometry in the cross-section plane (y, z) of the pick node: the string's rest line is the origin,
the pick moves along +y, +z points away from the instrument's top. The pick is a capsule: the segment
from the tip centre P along the unit axis u = (sin theta, cos theta), thickened by half the pick
thickness. theta > 0 leans the pick body towards the stroke direction, so the tip trails: the face
the string presses on is inclined such that the string slides along it towards the tip (it slides
at all only if tan theta > mu, the friction cone) and leaves the pick round the tip.
"""
import numpy as np

# Pick: Dunlop Tortex standard, 0.73 mm, Delrin (acetal homopolymer; Dunlop). Friction acetal on
# steel: dynamic 0.2 (DuPont Delrin 150 data sheet, ASTM D3702, 40 psi, 50 fpm); one coefficient
# for stick and slip (no static value given there). The tip edge is modelled as a half circle of the
# pick thickness. Gesture (choices, not measured): angle, depth of the tip below the string, speed,
# speed ramp, pluck position beta (from the bridge end, node 0).
PICK_PARAMS = {
    "beta": 0.2,
    "thickness": 0.73e-3,
    "face_length": 10e-3,
    "angle_deg": 30.0,
    "depth": 1.0e-3,
    "velocity": 0.5,
    "ramp_time": 1e-3,
    "start_time": 0.0,
    "mu_s": 0.2,
    "mu_d": 0.2,
    "torsion": True,
}


def pick_node(node_count, beta):
    return int(round(beta * (node_count - 1)))


def capsule_distance(centre, tip, axis, length):
    """Distance from a point to the segment tip + s axis, s in [0, length] (float64)."""
    centre, tip, axis = (np.asarray(v, dtype=float) for v in (centre, tip, axis))
    s = np.clip(np.dot(centre - tip, axis), 0.0, length)
    return float(np.linalg.norm(centre - (tip + s * axis)))


def touching_tip_y(string_radius, tip_radius, axis, tip_z, length):
    """Tip-centre y (< 0) at which the pick, approaching along +y, just touches the string at rest:
    capsule distance from the origin equal to R + rho. The distance decreases monotonically while
    the pick approaches, so bisection on [-(R + rho + length), y_min] finds the first touch."""
    reach = string_radius + tip_radius
    dist = lambda y: capsule_distance((0.0, 0.0), (y, tip_z), axis, length) - reach
    lo = -(reach + length) - 1e-3
    # the y at which the segment comes closest to the origin (distance minimal over y)
    ys = np.linspace(lo, 0.0, 20001)
    d = np.array([dist(y) for y in ys])
    hi = float(ys[np.argmin(d)])
    assert d.min() < 0.0, "the pick passes the string without touching it (tip not deep enough)"
    assert dist(lo) > 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dist(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return lo


def configure_pick(sim, pick, node_count, dt, string_radius):
    """Set the pick on sim; returns the derived quantities (recorded in parameters.json)."""
    theta = np.radians(pick["angle_deg"])
    axis = (float(np.sin(theta)), float(np.cos(theta)))
    rho = 0.5 * pick["thickness"]
    tip_z = -float(pick["depth"])
    tip_y0 = touching_tip_y(string_radius, rho, axis, tip_z, pick["face_length"])
    node = pick_node(node_count, pick["beta"])
    start_step = int(round(pick["start_time"] / dt))
    ramp_steps = max(1, int(round(pick["ramp_time"] / dt)))
    sim.set_pick(node, bool(pick["torsion"]), float(string_radius), float(rho), float(pick["face_length"]),
                 [axis[0], axis[1]], [float(tip_y0), tip_z], float(pick["velocity"]), start_step,
                 ramp_steps, float(pick["mu_s"]), float(pick["mu_d"]))
    return {"pick_node": node, "axis": axis, "tip_radius": rho, "tip_y0": tip_y0, "tip_z": tip_z,
            "start_step": start_step, "ramp_steps": ramp_steps, "string_radius": float(string_radius),
            "friction_cone_slides": bool(np.tan(theta) > pick["mu_s"])}


def pick_arrays(frames):
    """Per-frame arrays from Simulation.save_surface() output (pick part)."""
    p = [f[0] for f in frames]
    out = {
        "step": np.array([x[0] for x in p], dtype=np.int64),
        "first_contact_step": np.array([x[1] for x in p], dtype=np.int64),
        "last_contact_step": np.array([x[2] for x in p], dtype=np.int64),
        "max_penetration": np.array([x[3] for x in p], dtype=np.float64),
        "tip_y": np.array([x[4] for x in p], dtype=np.float64),
        "in_contact": np.array([x[5][0] for x in p], dtype=bool),
        "sticking": np.array([x[5][1] for x in p], dtype=bool),
    }
    vals = np.array([x[5][2] for x in p], dtype=np.float64)
    for i, k in enumerate(["slip_velocity", "normal_force", "friction_force", "normal_y", "normal_z",
                           "body_work", "friction_work", "normal_work", "impulse_t"]):
        out[k] = vals[:, i]
    return out
