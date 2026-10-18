struct Node {
    displacement: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    velocity: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    curvature: vec3<f32>, // 12 bytes
    bend_stretch: f32, // 4 bytes (offset 44): eta' eta_v^3 kappa^T K_bt kappa of this bending region,
                       // the stretch derivative of its bending/twist energy (variational force law)
   
    reference_orientation: vec4<f32>, // 16 bytes
    // this is the quaterion encoding the rotation needed to transform
    // the one set of adjacent directors to the other adjacent one at rest
        
    internal_moment: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    rest_relative_orientation: vec4<f32>, // 16 bytes
    // q0_i^-1 * q0_{i+1}: rotation from the rest frame of edge i to that of edge i+1.
    // Precomputed in double precision, so it is never rebuilt from two rounded absolute
    // orientations.
}

struct Edge {
// alignment is 16 bytes
    orientation: vec4<f32>, // 16 bytes
    // Deviation p from the rest orientation q0: the edge orientation is q = q0 * p.
    // p stays near the identity, so its vector part keeps full relative precision.

    angular_velocity: vec3<f32>, // 12 bytes
    len_inv: f32, // 4 bytes

    strain: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
        
    reference_strain: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    internal_force: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    reference_vector: vec3<f32>,
    // 4 bytes of implicit padding

    reference_offset: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
    // R0^T * reference_vector - dl * e3 (MF of the rest frame), precomputed in double precision.
    // Adding the rotated displacement difference gives R0^T * r - dl * e3, a small vector.

    rest_frame: mat3x3<f32>, // 48 bytes (3 columns of 16 bytes)
    // Rest directors [d1 d2 d3] in lab coordinates, R0 = R(q0).
}

struct Uniforms {
    node_count: u32,
    edge_count: u32,
    chunk_size: u32,
    mass_inv: f32,

    dt: f32,
    dt_inv: f32,
    dl: f32,
    dl_inv: f32,
    

    stiffness_se: vec3<f32>, // 12 bytes - shearing / extension stiffness matrix
    clamp_offset: u32,
    stiffness_bt: vec3<f32>, // 12 bytes - bending / twisting extension matrix
    // 4 bytes of implicit padding
    inertia: vec3<f32>, // 12 bytes, diag(I1, I2, I3)
    // 4 bytes of implicit padding
    inertia_inv: vec3<f32>, // 12 bytes, 
    // 4 bytes of implicit padding

    // Viscous damping coefficients: torque -angular_damping * omega (per material axis: tilt, tilt,
    // torsion; N m s) and force -linear_damping * v (N s / m). Derived from physical decay rates in
    // batches/utils.damping_coefficients.
    angular_damping: vec3<f32>, // offset 96
    linear_damping: f32,        // offset 108

    // 0: legacy force law (stiffness eta^2 K without its derivative, runs before 2026-10-18)
    // 1: variational (the force is the gradient of the stored energy, see compute_internals)
    force_law: u32,             // offset 112
}

const ETA_SLOPE: f32 = -0.3; // eta(e) = 1 + ETA_SLOPE (e - 1)

struct HammerState {
    displacement: f32,
    velocity: f32,
    mass_inv: f32,
    stiffness: f32,

    exponent: f32,
    hysteresis_factor: f32,
    relaxation_time: f32,
    force_accumulator: atomic<i32>,
}

struct PushConstants { 
    current: u32, // either 0 or 1
    future: u32,  // current == 0 ? 1 : 0
    output_index: u32,
    hammering: u32, // 0 == false

    write_output: u32, // advance_frame: 1 if this dispatch ends the output frame (formerly linear damping)
    _reserved1: f32, // formerly angular damping (now in Uniforms)
    hammer_offset_y: f32,
    steps: u32, // solver steps per output frame (advance_frame only)
}

var <push_constant> c: PushConstants;

@group(0)
@binding(0)
var <uniform> uniforms: Uniforms;

@group(0)
@binding(1)
var<storage, read_write> nodes: array<Node>;

@group(0)
@binding(2)
var<storage, read_write> edges: array<Edge>;

@group(0)
@binding(3)
var<storage, read_write> output_nodes: array<Node>;

@group(0)
@binding(4)
var<storage, read_write> output_edges: array<Edge>;

@group(0)
@binding(5)
var<storage, read> hammer_weights: array<f32>;

@group(0)
@binding(6)
var<storage, read_write> hammer: HammerState;

@group(0)
@binding(7)
var<storage, read_write> integral_memory: array<f32>;

@compute
@workgroup_size(64) 
fn init(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;
    let future = (c.future * uniforms.node_count) + global_id.x;
    if (global_id.x < uniforms.node_count - 1) {
        let r = edges[current].reference_vector + (nodes[current + 1].displacement - nodes[current].displacement);
    }
}

@compute
@workgroup_size(64) 
fn create_references(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;
    let future = (c.future * uniforms.node_count) + global_id.x;
    if (global_id.x < uniforms.node_count - 1) {
        let current_strain_MF = material_strain(edges[current].orientation, edge_offset(current));

        edges[current].reference_strain = current_strain_MF;
        edges[future].reference_strain = current_strain_MF;
    }

    if (global_id.x < uniforms.node_count - 2) {
        let relative_orientation = relative_rotation(current);

        nodes[current].reference_orientation = relative_orientation;
        nodes[future].reference_orientation = relative_orientation;
    }
}



@compute
@workgroup_size(64) 
fn half_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    half_step_body(global_id.x, c.current * uniforms.node_count);
}

fn half_step_body(gid: u32, current_base: u32) {
    let global_id = vec3<u32>(gid, 0u, 0u);
    let current = current_base + gid;

    // Half step position update (drift)
    if (global_id.x > (0 + uniforms.clamp_offset) && global_id.x < (uniforms.node_count - 2 - uniforms.clamp_offset)) {
        var update = integrate_exponential(edges[current].orientation, edges[current].angular_velocity, uniforms.dt * 0.5);
        // Safety: make sure future orientation selects pole closest to current orientation
        if (dot(update, edges[current].orientation) < 0.0) {
            update = -update;
        }
        
        edges[current].orientation = update;
    }

    if (global_id.x > 0 && global_id.x < uniforms.node_count - 1) {
        nodes[current].displacement = nodes[current].displacement + (nodes[current].velocity * uniforms.dt * 0.5);
    }
}

@compute
@workgroup_size(1) 
fn hammer_half_step() {
    hammer_half_step_body();
}

fn hammer_half_step_body() {
    hammer.displacement = hammer.displacement + (hammer.velocity * uniforms.dt * 0.5);
}

@compute
@workgroup_size(64) 
fn compute_internals(@builtin(global_invocation_id) global_id: vec3<u32>) {
    compute_internals_body(global_id.x, c.current * uniforms.node_count, c.future * uniforms.node_count);
}

fn compute_internals_body(gid: u32, current_base: u32, future_base: u32) {
    let global_id = vec3<u32>(gid, 0u, 0u);
    let current = current_base + gid;
    let future = future_base + gid;
    var stretch: f32;

    if (global_id.x < uniforms.node_count - 1) {

        // Same quantities as R(q)^T r / dl - e3 and |r| / dl - 1, evaluated from the small
        // offset w = R0^T r - dl e3 so that nothing close to 1 is subtracted from 1.
        let w = edge_offset(current);
        let current_strain_MF = material_strain(edges[current].orientation, w);
        let len = offset_length(w);
        edges[current].len_inv = 1.0 / len;

        stretch = offset_stretch(w, len); // e - 1
        let eta = 1.0 - 0.3 * stretch;
        let stiffness_se_dyn = uniforms.stiffness_se * eta * eta;

        let sigma_eff = current_strain_MF - edges[current].reference_strain; // (MF)
        edges[future].strain = sigma_eff;
        var n_MF = stiffness_se_dyn * sigma_eff;
        if (uniforms.force_law == 1u) {
            // Gradient of W = 1/2 eta(e)^2 sigma^T K sigma dl and of the rotational kinetic energy
            // T = 1/2 omega^T I eta(e)^2 omega with respect to the edge vector, through e = |r|/dl:
            // eta eta' (sigma^T K sigma - omega^T I omega / dl) along the edge tangent.
            let omega = edges[current].angular_velocity;
            let slope = ETA_SLOPE * eta * (dot(uniforms.stiffness_se * sigma_eff, sigma_eff)
                                           - dot(uniforms.inertia * omega, omega) * uniforms.dl_inv);
            let tangent_unit_MF = (vec3<f32>(0.0, 0.0, 1.0) + current_strain_MF) * (uniforms.dl / len);
            n_MF = n_MF + slope * tangent_unit_MF;
        }
        edges[current].internal_force = to_lab(current, n_MF); // (LF)
    }

    if ( global_id.x < uniforms.node_count - 2) {
        // NB: The paper chooses its indexing in a way that makes this somewhat weird. nodes[i] stores the average
        // curvature whose integral is rotation needed to from edges[i].orientation to edges[i+1].orientation.
        // Therefore this convention causes the i-th region to contain node i+1. This must be accounted for when
        // defining the descrete difference/average operators (i.e. here one must use backwards differences)!
        
        let w_next = edge_offset(current + 1);
        let next_stretch = offset_stretch(w_next, offset_length(w_next));
        // epsilon - 1 with epsilon = (e1 + e2) / 2
        let eta_voronoi = 1.0 - 0.3 * (0.5 * (stretch + next_stretch));
        let stiffness_bt_dyn = uniforms.stiffness_bt * eta_voronoi * eta_voronoi * eta_voronoi * eta_voronoi;

        var relative_orientation = relative_rotation(current);

        // Select closest quaternion pole
        if (dot(relative_orientation, nodes[current].reference_orientation) < 0.0) {
            relative_orientation = -relative_orientation;
        }

        var effective_orientation = qmul(qinv(nodes[current].reference_orientation), relative_orientation); // (MF)
        // if (effective_orientation.w < 0.0) { effective_orientation = -effective_orientation; }

        // This is B applied to kappa (MF); here B is diagonal -> pointwise multiplication
        let kappa = 2.0 * uniforms.dl_inv * effective_orientation.xyz;
        nodes[current].curvature = kappa;
        nodes[current].internal_moment = stiffness_bt_dyn * kappa;
        if (uniforms.force_law == 1u) {
            // d/de of 1/2 eta_v^4 kappa^T K_bt kappa dl for each of the two edges (eta_v uses their mean)
            nodes[current].bend_stretch = ETA_SLOPE * dot(kappa, stiffness_bt_dyn * kappa) / eta_voronoi;
        }
    }
}


@compute
@workgroup_size(64) 
fn compute_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    compute_forces_body(global_id.x, c.current * uniforms.node_count, c.future * uniforms.node_count);
}

fn compute_forces_body(gid: u32, current_base: u32, future_base: u32) {
    let global_id = vec3<u32>(gid, 0u, 0u);
    let current = current_base + gid;
    let future = future_base + gid;

    var ext_force = vec3<f32>(0.0);
    var ext_couple = vec3<f32>(0.0);

    // Interior nodes
    if (global_id.x > 0 && global_id.x < uniforms.node_count - 1) {

        // Hammer collision
        if (c.hammering == 1) {
            let weight = hammer_weights[global_id.x];
            let compression = hammer.displacement - nodes[current].displacement.z;

            if (compression > 0.0) {
                let hereditary_term = (hammer.hysteresis_factor / hammer.relaxation_time) * integral_memory[global_id.x];
                let u_p = pow(compression, hammer.exponent);
                let dZ_dt = u_p - (integral_memory[global_id.x] / hammer.relaxation_time);
                integral_memory[global_id.x] += dZ_dt * uniforms.dt;

                let local_force = max(0.0, hammer.stiffness *  (u_p - hereditary_term));

                let weighted_force = local_force * weight;
                ext_force = vec3<f32>(0.0, 0.0, weighted_force);

                // LF_x is center line, LF_z is striking direction, so LF_y is tangential
                // THEORETICALLY this belongs inside the edge IF-block, but if the string isn't struck
                // at its far end, calculating this here is fine. 
                // In fact, the ghost edge makes these accesses always safe and the ghost torque is never applied.
                // For hammer_offset_y = 0 this becomes 0.
                ext_couple = to_material(current, cross(vec3<f32>(0.0, c.hammer_offset_y, 0.0), ext_force));

                let fixed_point_force = i32(weighted_force * 10000.0);
                atomicAdd(&hammer.force_accumulator, fixed_point_force);
            } else {
                let decay = -(integral_memory[global_id.x] / hammer.relaxation_time);
                integral_memory[global_id.x] = max(0.0, integral_memory[global_id.x] + decay * uniforms.dt);
            }
        }

        
        let damping_force = - uniforms.linear_damping * nodes[current].velocity;
        
        var bend_stretch_force = vec3<f32>(0.0);
        if (uniforms.force_law == 1u) {
            // Edge e carries (c_e + c_{e-1}) along its tangent, c_i = nodes[i].bend_stretch of the
            // bending regions containing it (regions exist for i = 0 .. N-3; others are zero).
            let j = global_id.x;
            var c_here = nodes[current].bend_stretch + nodes[current - 1].bend_stretch;     // edge j
            var c_prev = nodes[current - 1].bend_stretch;                                    // edge j-1
            if (j >= 2u) { c_prev = c_prev + nodes[current - 2].bend_stretch; }
            if (j == uniforms.node_count - 2u) { c_here = nodes[current - 1].bend_stretch; }
            let t_here = (edges[current].reference_vector + nodes[current + 1].displacement - nodes[current].displacement) * edges[current].len_inv;
            let t_prev = (edges[current - 1].reference_vector + nodes[current].displacement - nodes[current - 1].displacement) * edges[current - 1].len_inv;
            bend_stretch_force = c_here * t_here - c_prev * t_prev;
        }

        // Only calculate for interior edges, so naive difference operator is fine here!
        let v_tt = (edges[current].internal_force - edges[current- 1].internal_force + bend_stretch_force + ext_force + damping_force) * uniforms.mass_inv;

        // Full step velocity update (kick)
        nodes[future].velocity = nodes[current].velocity + (v_tt * uniforms.dt);
        // Half step position update (drift)
        nodes[future].displacement = nodes[current].displacement + (nodes[future].velocity * uniforms.dt * 0.5);
    } else {
        nodes[future].velocity = vec3<f32>(0.0);
    }

    
    if (global_id.x > (0 + uniforms.clamp_offset) && global_id.x < (uniforms.node_count - 2 - uniforms.clamp_offset)) {
        
        // tangent_MF = R(q)^T r / dl = e3 + strain_MF; kept split, see the couple term below.
        let w = edge_offset(current);
        let strain_MF = material_strain(edges[current].orientation, w);

        let damping_torque = - uniforms.angular_damping * edges[current].angular_velocity;

        var difference: vec3<f32>;
        var average: vec3<f32>;
        if (global_id.x == 0) {
            difference = nodes[current].internal_moment;
            average = 0.5 * uniforms.dl * cross(nodes[current].curvature, nodes[current].internal_moment);
        } else if (global_id.x == uniforms.node_count - 2) {
            difference = - nodes[current- 1].internal_moment;
            average = 0.5 * uniforms.dl * cross(nodes[current- 1].curvature, nodes[current- 1].internal_moment);
        } else {
            difference = nodes[current].internal_moment - nodes[current- 1].internal_moment;
            average = 0.5 * uniforms.dl * (cross(nodes[current].curvature, nodes[current].internal_moment) + cross(nodes[current- 1].curvature, nodes[current- 1].internal_moment));
        }

        let eta = 1.0 - 0.3 * offset_stretch(w, offset_length(w));
        let inertia_dyn = uniforms.inertia * eta * eta;
        let inertia_inv_dyn = uniforms.inertia_inv * ( 1.0 / (eta * eta));

        let relative_velocity = nodes[current + 1].velocity - nodes[current].velocity;
        let tangent_unit = (edges[current].reference_vector + nodes[current + 1].displacement - nodes[current].displacement) * edges[current].len_inv;
        let e_dot = dot(relative_velocity, tangent_unit) * uniforms.dl_inv;

        let unsteady_dilatation = (0.6 * e_dot / eta) * inertia_dyn * edges[current].angular_velocity;

        // cross(tangent_MF, n) with tangent_MF = e3 + strain_MF, split by linearity
        let n_MF = to_material(current, edges[current].internal_force);
        let shear_couple = cross(vec3<f32>(0.0, 0.0, 1.0), n_MF) + cross(strain_MF, n_MF);

        let phi_tt =( difference
                            + average
                            + (shear_couple * uniforms.dl)
                            + ext_couple
                            + damping_torque
                            + cross((inertia_dyn * edges[current].angular_velocity), edges[current].angular_velocity) 
                            + unsteady_dilatation
                            ) * inertia_inv_dyn;


        // Full step velocity update (kick)
        edges[future].angular_velocity = edges[current].angular_velocity + (phi_tt * uniforms.dt);
        // Half step position update (drift)
        var update = integrate_exponential(edges[current].orientation, edges[future].angular_velocity, uniforms.dt * 0.5);
        // Safety: make sure future orientation selects pole closest to current orientation
        if (dot(update, edges[current].orientation) < 0.0) {
            update = -update;
        }
        
        edges[future].orientation = update;
    }
}

@compute
@workgroup_size(1) 
fn hammer_update() {
    hammer_update_body();
}

fn hammer_update_body() {
    let total_force = f32(atomicLoad(&hammer.force_accumulator)) / 10000.0;
    let hammer_acceleration = - total_force * hammer.mass_inv;
    hammer.velocity = hammer.velocity + (hammer_acceleration * uniforms.dt);
    hammer.displacement= hammer.displacement + (hammer.velocity * uniforms.dt * 0.5);
    atomicStore(&hammer.force_accumulator, 0);
}

@compute
@workgroup_size(64) 
fn save_output(@builtin(global_invocation_id) global_id: vec3<u32>) {
    save_output_body(global_id.x, c.current * uniforms.node_count);
}

fn save_output_body(gid: u32, current_base: u32) {
    let i = (c.output_index * uniforms.node_count) + gid;
    let current = current_base + gid;
    output_nodes[i] = nodes[current];
    output_edges[i] = edges[current];
}

// One output frame in one dispatch: a single workgroup runs all c.steps solver steps of the frame
// and writes the output frame. The phases are exactly the passes of compute_reference (half_step,
// hammer_half_step, compute_internals, compute_forces, hammer_update), separated by storage
// barriers instead of pass boundaries; no phase reads what the same phase writes, so every
// element sees the same values in the same order as with separate passes.
// Two workgroup sizes: 1024 threads (one per node at N = 1024) where the adapter allows it, 256 otherwise.
@compute
@workgroup_size(1024)
fn advance_frame_1024(@builtin(local_invocation_index) t: u32) {
    advance_frame_body(t, 1024u);
}

@compute
@workgroup_size(256)
fn advance_frame_256(@builtin(local_invocation_index) t: u32) {
    advance_frame_body(t, 256u);
}

fn advance_frame_body(t: u32, threads: u32) {
    var current_base = c.current * uniforms.node_count;
    var future_base = c.future * uniforms.node_count;

    for (var step = 0u; step < c.steps; step++) {
        for (var i = t; i < uniforms.node_count; i += threads) {
            half_step_body(i, current_base);
        }
        if (t == 0u) {
            hammer_half_step_body();
        }
        storageBarrier();

        for (var i = t; i < uniforms.node_count; i += threads) {
            compute_internals_body(i, current_base, future_base);
        }
        storageBarrier();

        for (var i = t; i < uniforms.node_count; i += threads) {
            compute_forces_body(i, current_base, future_base);
        }
        storageBarrier();

        if (t == 0u) {
            hammer_update_body();
        }
        storageBarrier();

        let swap = current_base;
        current_base = future_base;
        future_base = swap;
    }

    if (c.write_output == 1u) {
        for (var i = t; i < uniforms.node_count; i += threads) {
            save_output_body(i, current_base);
        }
    }
}

fn qinv(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

// Rotate TO the lab frame
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = vec4<f32>(v, 0.0);

    let q_inv= vec4<f32>(-q.xyz, q.w); // inverse of a unit quaternion
    let rotated = qmul(qmul(q, u), q_inv);

    return rotated.xyz;
}

// Rotate TO the material frame cooridinates (this is inv because Q rotates the (0,0,1) to (d1, d2, d3) in lab coordinates)
fn rotate_inv(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = vec4<f32>(v, 0.0);

    let q_inv= vec4<f32>(-q.xyz, q.w); // inverse of a unit quaternion
    let rotated = qmul(qmul(q_inv, u), q);

    return rotated.xyz;
}

// R(q)^T v for edge i, with q = q0 * p: first into the rest frame, then by the deviation.
fn to_material(i: u32, v: vec3<f32>) -> vec3<f32> {
    // v * M == transpose(M) * v
    return rotate_inv(edges[i].orientation, v * edges[i].rest_frame);
}

// R(q) v for edge i, with q = q0 * p.
fn to_lab(i: u32, v: vec3<f32>) -> vec3<f32> {
    return edges[i].rest_frame * rotate(edges[i].orientation, v);
}

// w = R0^T r - dl e3 for edge i, with r = reference_vector + (u_{i+1} - u_i).
fn edge_offset(i: u32) -> vec3<f32> {
    return edges[i].reference_offset
        + ((nodes[i + 1].displacement - nodes[i].displacement) * edges[i].rest_frame);
}

// R(p)^T e3 - e3 for a unit quaternion p, built from its vector part only.
fn rotation_deviation_inv(p: vec4<f32>) -> vec3<f32> {
    let e3 = vec3<f32>(0.0, 0.0, 1.0);
    return -2.0 * p.w * cross(p.xyz, e3) + 2.0 * cross(p.xyz, cross(p.xyz, e3));
}

// Strain R(q)^T r / dl - e3, written as R(p)^T w / dl + (R(p)^T e3 - e3).
fn material_strain(p: vec4<f32>, w: vec3<f32>) -> vec3<f32> {
    return rotate_inv(p, w) * uniforms.dl_inv + rotation_deviation_inv(p);
}

// |r| = |dl e3 + w|
fn offset_length(w: vec3<f32>) -> f32 {
    return length(vec3<f32>(w.x, w.y, uniforms.dl + w.z));
}

// e - 1 = |r| / dl - 1 = (2 dl w_z + |w|^2) / (dl (|r| + dl))
fn offset_stretch(w: vec3<f32>, len: f32) -> f32 {
    return (2.0 * uniforms.dl * w.z + dot(w, w)) / (uniforms.dl * (len + uniforms.dl));
}

// q_i^-1 * q_{i+1} = p_i^-1 * (q0_i^-1 * q0_{i+1}) * p_{i+1}
fn relative_rotation(i: u32) -> vec4<f32> {
    return qmul(qmul(qinv(edges[i].orientation), nodes[i].rest_relative_orientation),
                edges[i + 1].orientation);
}

fn integrate_exponential(q: vec4<f32>, omega: vec3<f32>, dt: f32) -> vec4<f32> {
    let theta_vec = omega * dt;
    let theta_mag = length(theta_vec);
    
    var dq: vec4<f32>;
    
    if (theta_mag < 0.0001) {
        dq = vec4<f32>(theta_vec * 0.5, 1.0);
    } else {
        let half_theta = theta_mag * 0.5;
        let s = sin(half_theta) / theta_mag; // Pre-divide by mag to normalize theta_vec
        dq = vec4<f32>(theta_vec * s, cos(half_theta));
    }
    
    var next_q = qmul(q, dq);
    
    return normalize(next_q);
}
