import numpy as np

def generate_straight_rod(
    node_count, tuned_length, tension_force, core_radius, winding_radius,
    E_core, G_core, rho_core, E_winding, G_winding, rho_winding, packing_factor, mu, twists
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

    def twist_quaternion(angle):
        half = 0.5 * angle
        return np.array([np.sin(half), 0.0, 0.0, np.cos(half)])

    def quat_mul(q1, q2):
        x1, y1, z1, w1 = q1; x2, y2, z2, w2 = q2
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

def make_weights(node_count, node_index=None, width_count=4):
    data = np.zeros((node_count, 4), dtype=np.float32)
    x = np.arange(node_count, dtype=np.float32)
    if node_index is None:
        node_index = node_count // 2

    sigma = width_count / 2.355
    if sigma == 0:
        sigma = 1e-5

    gauss_vals = np.exp(-((x - node_index) ** 2) / (2 * sigma ** 2))
    mask = np.abs(x - node_index) <= width_count
    data[:, 2] = gauss_vals * mask
    return data
