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

    displacement_lo: vec3<f32>, // 12 bytes (offset 96)
    // 4 bytes of implicit padding
    // The displacement is the unevaluated sum displacement + displacement_lo of two f32
    // (|lo| <= ulp(displacement) / 2), see drift_node. A single f32 resolves a displacement of a
    // few mm only to ~2e-10 m; the difference of two neighbours, i.e. the shear strain of the
    // edge, would carry that rounding (~2e-7), which pumps the stiff shear-rotation branch.
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

    // Kelvin-Voigt damping of the twist rate (N m^2 s): moment D3 * d(kappa_3)/dt added to the twist
    // moment of each bending region, kappa_dot being the exact time derivative of the region's
    // curvature measure. Torsional decay rate gamma_0 + D3 k^2 / (2 rho I_p) together with the
    // viscous torsion damping; 0 disables it.
    twist_rate_damping: f32,    // offset 116

    // Always 0, unknown to the compiler: see opaque.
    opaque_zero: u32,           // offset 120
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

// Bow contact (see bow_contact_body). The bow presses along -z on the string's top surface and
// moves along +y. Contact k acts on node contacts[k].node with weight contacts[k].weight; its
// friction couple acts on the edge with the same index, whose rotation also gives the surface
// speed. Pairing node j with edge j (half an edge apart) keeps every contact independent, so each
// one is resolved exactly on its own.
const MAX_BOW_CONTACTS: u32 = 32u;

struct BowContact {           // 48 bytes
    node: u32,
    weight: f32,              // share of the normal force (sum 1)
    sticking: u32,            // 1: stuck at the end of the last step
    slip_velocity: f32,       // relative surface velocity v_y - r omega_x - v_b after the last step (m/s)
    force: f32,               // friction force of the last step on the string (lab y, N)
    impulse: f32,             // sum of force dt over the current output frame (N s)
    bow_work: f32,            // sum of force v_b dt over the frame (J): work of the bow on the string
    friction_work: f32,       // sum of force (step-average v_rel) dt over the frame (J): work of the slip
    normal_work: f32,         // sum of -normal (step-average v_z) dt over the frame (J)
    temperature: f32,         // contact temperature above ambient (K, thermal model)
    heat_power: f32,          // frictional heat input of the last step (W): -force (step-average v_rel)
    _pad1: f32,
}

// Woodhouse's thermal friction model (batches/bowing.py has the parameters and a float64 reference).
const BOW_EXACT_TERMS: u32 = 32u;
const BOW_SOE_TERMS: u32 = 32u;
const BOW_RING: u32 = 65536u;     // temperature history per contact (power of two)

struct BowThermal {               // 560 bytes
    c_over_h: f32,                // heat capacity of the layer / dt (W/K)
    conv_coef: f32,               // convection a delta rho_r c_r (W/K per m/s of slip speed)
    lam_string: f32,              // lambda / sqrt(dt) of the string (stationary surface, W/K)
    lam_bow: f32,                 // the same for the bow (moving surface)
    window: u32,                  // M: steps a bow surface point has been in contact
    f_window: f32,                // f(M)
    terms: u32,                   // sum-of-exponentials terms in use
    _pad0: u32,
    cal_k1: f32,                  // steady-sliding calibration T(v) = mu(v) N sqrt(v) / (k1 + k2 sqrt(v))
    cal_k2: f32,
    cal_force: f32,
    _pad1: f32,
    exact: array<f32, 32>,        // f(n) = 2 (sqrt(n + 1) - sqrt(n)), n < BOW_EXACT_TERMS
    decay: array<f32, 32>,        // f(n) ~ sum_k weight_k decay_k^(n - BOW_EXACT_TERMS) beyond
    weight: array<f32, 32>,
    weight_window: array<f32, 32>,  // weight_k decay_k^(M - BOW_EXACT_TERMS)
}

struct BowFrame {             // 1552 bytes: the bow profile and every contact at the end of an output frame
    normal_force: f32,
    velocity: f32,
    step: u32,
    _pad: u32,
    contacts: array<BowContact, MAX_BOW_CONTACTS>,
}

struct Bow {
    model: u32,               // 0: off, 1: friction curve mu(|v_rel|), 2: thermal mu(T)
    torsion: u32,             // 1: surface speed and couple include the edge rotation; 0: inhibited
    contact_count: u32,
    step: u32,                // solver steps since set_bow (bow profile clock)

    radius: f32,              // lever arm from the centre line to the contact (string radius)
    normal_force: f32,        // steady normal force (N)
    velocity: f32,            // steady bow velocity (m/s)
    mu_d: f32,                // friction curve mu(v) = mu_d + mu_a1 exp(-v / mu_v1) + mu_a2 exp(-v / mu_v2)

    mu_a1: f32,
    mu_v1: f32,
    mu_a2: f32,
    mu_v2: f32,

    start_step: u32,          // profile: linear rise over ramp_steps from start_step,
    ramp_steps: u32,          // steady, linear fall over ramp_steps from stop_step
    stop_step: u32,
    _pad: u32,

    contacts: array<BowContact, MAX_BOW_CONTACTS>,
    thermal: BowThermal,
    soe: array<f32, 2048>,                  // per contact: 32 full-history states, 32 window states
    ring: array<f32, 2097152>,              // per contact: BOW_RING temperatures, T_j at j % BOW_RING
    frames: array<BowFrame>,  // chunk_size output frames
}

// Contacts at the string surface other than the bow: the plectrum (pick_contact_body) and the
// friction of the hammer felt (hammer_friction_body). Each acts on node j and, through its friction
// couple, on edge j, like a bow contact. The contact normal n lies in the lab y-z plane and points
// from the body to the string centre; the friction direction is t = e_x x n = (0, -n_z, n_y), and
// the contact point sits at -R n from the centre line, so the friction couple is -R F_t e_x and the
// surface speed along t is v.t - R omega_x. Slip along the string is frictionless (as for the bow).
const MAX_HAMMER_CONTACTS: u32 = 32u;
const MAX_SURFACE_NODES: u32 = 1024u;

struct SurfaceContact {        // 48 bytes
    node: u32,
    sticking: u32,             // 1: stuck at the end of the last step
    in_contact: u32,           // 1: normal force > 0 in the last step
    slip_velocity: f32,        // surface speed along t relative to the body after the last step (m/s)
    normal_force: f32,         // along n, last step (N)
    friction_force: f32,       // along t, last step (N)
    normal_y: f32,             // contact normal n (lab y, z), last step
    normal_z: f32,
    body_work: f32,            // frame sum (J): F . (body surface velocity) dt
    friction_work: f32,        // frame sum (J): F_t (step-average slip velocity) dt
    normal_work: f32,          // frame sum (J): F_n (step-average relative normal velocity) dt
    impulse_t: f32,            // frame sum of F_t dt (N s); the twist impulse about e_x is -R times it
}

// Rigid plectrum. In the cross-section plane (y, z) of node j it is a capsule: the segment from the
// tip centre P along the unit vector u (pick axis) of length `length`, thickened by tip_radius (half
// the pick thickness), so its face and its rounded tip are one distance function. P moves along +y:
// at rest until start_step, then with a speed rising linearly to `velocity` over ramp_steps.
struct Pick {                  // 128 bytes
    enabled: u32,
    torsion: u32,              // 1: surface speed and couple include the edge rotation; 0: inhibited
    node: u32,
    step: u32,                 // solver steps since set_pick (pick clock)

    radius: f32,               // string radius R
    tip_radius: f32,
    length: f32,
    mu_s: f32,

    mu_d: f32,
    axis_y: f32,               // u
    axis_z: f32,
    tip_y0: f32,               // P at the start (lab y, z of the string's rest line = 0)

    tip_z: f32,
    velocity: f32,             // steady pick speed along +y (m/s)
    start_step: u32,
    ramp_steps: u32,

    first_contact_step: u32,   // pick clock of the first and the last step with a normal force
    last_contact_step: u32,    // (0: none yet)
    max_penetration: f32,      // largest -gap at a force evaluation (m)
    tip_y: f32,                // P_y at the last step

    contact: SurfaceContact,
}

// Hammer contact geometry and felt friction. curved == 1: the crown is a circle of radius
// crown_radius across the string, centred c.hammer_offset_y beside it, and the felt force acts
// along the normal of that geometry; 0: flat crown, force along +z (the original contact).
// enabled == 1: Coulomb friction of the felt on the hammer's contact nodes.
struct HammerFriction {        // 1584 bytes
    enabled: u32,
    torsion: u32,
    count: u32,
    curved: u32,

    radius: f32,               // string radius R
    crown_radius: f32,
    mu_s: f32,
    mu_d: f32,

    hammer_work: f32,          // frame sum (J): work of the friction reaction on the hammer
    hammer_displacement: f32,  // hammer state at the end of the frame
    hammer_velocity: f32,
    _pad: f32,

    contacts: array<SurfaceContact, MAX_HAMMER_CONTACTS>,
}

struct SurfaceFrame {          // 1712 bytes: pick and hammer records at the end of an output frame
    pick: Pick,
    hammer: HammerFriction,
}

struct Surface {
    pick: Pick,
    hammer: HammerFriction,
    // per node, written by compute_forces while the felt friction is active: contact normal (y, z),
    // felt force (N, weighted) and 1 when in contact
    hammer_normal: array<vec4<f32>, MAX_SURFACE_NODES>,
    frames: array<SurfaceFrame>,  // chunk_size output frames
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

@group(0)
@binding(8)
var<storage, read_write> bow: Bow;

@group(0)
@binding(9)
var<storage, read_write> surface: Surface;

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
        drift_node(current, current, nodes[current].velocity * uniforms.dt * 0.5);
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
        var moment = stiffness_bt_dyn * kappa;
        if (uniforms.twist_rate_damping != 0.0) {
            // d/dt (q_i^-1 q_{i+1}) = 1/2 (q_rel omega_{i+1} - omega_i q_rel), body angular velocities
            let omega_i = vec4<f32>(edges[current].angular_velocity, 0.0);
            let omega_n = vec4<f32>(edges[current + 1].angular_velocity, 0.0);
            let rate = 0.5 * (qmul(relative_orientation, omega_n) - qmul(omega_i, relative_orientation));
            let kappa_rate = 2.0 * uniforms.dl_inv * qmul(qinv(nodes[current].reference_orientation), rate).xyz;
            moment.z = moment.z + uniforms.twist_rate_damping * kappa_rate.z;
        }
        nodes[current].internal_moment = moment;
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

            // Felt compression along the contact normal. Flat crown: along +z. Curved crown (radius
            // R_c across the string, centred hammer_offset_y beside it): reach - |c_s - c_h| with
            // reach = R_c + R, written so that no two numbers of size reach are subtracted.
            var normal = vec2<f32>(0.0, 1.0);
            var compression = hammer.displacement - nodes[current].displacement.z;
            if (surface.hammer.curved == 1u) {
                let reach = surface.hammer.crown_radius + surface.hammer.radius;
                let dy = nodes[current].displacement.y - c.hammer_offset_y;
                let dz = nodes[current].displacement.z - hammer.displacement;
                let dist = sqrt(dy * dy + (dz + reach) * (dz + reach));
                compression = -(dy * dy + dz * (dz + 2.0 * reach)) / (dist + reach);
                normal = vec2<f32>(dy, dz + reach) / dist;
            }
            if (surface.hammer.enabled == 1u) {
                surface.hammer_normal[global_id.x] = vec4<f32>(0.0);
            }

            if (compression > 0.0) {
                let hereditary_term = (hammer.hysteresis_factor / hammer.relaxation_time) * integral_memory[global_id.x];
                let u_p = pow(compression, hammer.exponent);
                let dZ_dt = u_p - (integral_memory[global_id.x] / hammer.relaxation_time);
                integral_memory[global_id.x] += dZ_dt * uniforms.dt;

                let local_force = max(0.0, hammer.stiffness *  (u_p - hereditary_term));

                let weighted_force = local_force * weight;
                // Felt pressure on a round string acts normal to its surface, through the centre
                // line, so it exerts no couple (an off-centre strike twists the string only through
                // the felt friction, hammer_friction_body).
                var fixed_point_force: i32;
                if (surface.hammer.curved == 1u) {
                    ext_force = vec3<f32>(0.0, weighted_force * normal.x, weighted_force * normal.y);
                    fixed_point_force = i32(weighted_force * normal.y * 10000.0);
                } else {
                    ext_force = vec3<f32>(0.0, 0.0, weighted_force);
                    fixed_point_force = i32(weighted_force * 10000.0);
                }
                if (surface.hammer.enabled == 1u) {
                    surface.hammer_normal[global_id.x] = vec4<f32>(normal, weighted_force, 1.0);
                }

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
            let t_here = (edges[current].reference_vector + node_difference(current)) * edges[current].len_inv;
            let t_prev = (edges[current - 1].reference_vector + node_difference(current - 1)) * edges[current - 1].len_inv;
            bend_stretch_force = c_here * t_here - c_prev * t_prev;
        }

        // Only calculate for interior edges, so naive difference operator is fine here!
        let v_tt = (edges[current].internal_force - edges[current- 1].internal_force + bend_stretch_force + ext_force + damping_force) * uniforms.mass_inv;

        // Full step velocity update (kick)
        nodes[future].velocity = nodes[current].velocity + (v_tt * uniforms.dt);
        // Half step position update (drift)
        drift_node(current, future, nodes[future].velocity * uniforms.dt * 0.5);
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
        let tangent_unit = (edges[current].reference_vector + node_difference(current)) * edges[current].len_inv;
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

// Bow profile in [0, 1] at solver step `step`: linear rise from start_step over ramp_steps,
// linear fall from stop_step over ramp_steps.
fn bow_profile(step: u32) -> f32 {
    let ramp = f32(max(bow.ramp_steps, 1u));
    let up = clamp(f32(i32(step) - i32(bow.start_step)) / ramp, 0.0, 1.0);
    let down = clamp(f32(i32(bow.stop_step) + i32(bow.ramp_steps) - i32(step)) / ramp, 0.0, 1.0);
    return min(up, down);
}

fn bow_mu(v: f32) -> f32 {
    return bow.mu_d + bow.mu_a1 * exp(-v / bow.mu_v1) + bow.mu_a2 * exp(-v / bow.mu_v2);
}

fn bow_mu_slope(v: f32) -> f32 {
    return -(bow.mu_a1 / bow.mu_v1) * exp(-v / bow.mu_v1) - (bow.mu_a2 / bow.mu_v2) * exp(-v / bow.mu_v2);
}

fn bow_ring_temperature(k: u32, j: u32) -> f32 {
    // j is a step index; for j <= 0 (wrapped) the slot has not been written yet and holds 0
    return bow.ring[k * BOW_RING + (j & (BOW_RING - 1u))];
}

fn bow_ring_difference(k: u32, j: u32) -> f32 {
    return bow_ring_temperature(k, j) - bow_ring_temperature(k, j - 1u);
}

// Contact temperature T_m of contact k at step m (>= 1), before this step's friction (Woodhouse 2003
// sec. 2.3): backward difference of his Eq. 1 with the friction power and slip speed of step m - 1,
//   C (T_m - T_(m-1)) / h = P - conv |v| T_m - Q_m,
//   Q_m = lam_s [2 dT_m + F_m] + lam_b [2 dT_m + F_m - W_m + f(M) T_(m-M)],
// with Smith & Woodhouse's kernel in step-response form (sum_n g_n T_(m-n) = sum_n G_n dT_(m-n),
// G_n = lambda h^-1/2 f(n), f(0) = 2): F_m = sum_(n>=1) f(n) dT_(m-n) (exact for n < 32, sum of
// exponentials beyond), W_m = sum_(n>=M) f(n) dT_(m-n) (the same exponentials fed with dT delayed by M)
// ends the bow's history after M steps. The temperature is implicit only through 2 lam dT_m and the
// convection, which keeps the step stable (explicitly, h g_0 / C is about 4 at dt = 2e-7 s).
fn bow_temperature(k: u32, m: u32) -> f32 {
    var F = 0.0;
    for (var n = 1u; n < BOW_EXACT_TERMS; n++) {
        F += bow.thermal.exact[n] * bow_ring_difference(k, m - n);
    }
    let d_full = bow_ring_difference(k, m - BOW_EXACT_TERMS);
    let d_window = bow_ring_difference(k, m - bow.thermal.window);
    var W = 0.0;
    let base = k * 2u * BOW_SOE_TERMS;
    for (var i = 0u; i < bow.thermal.terms; i++) {
        let a = bow.thermal.decay[i] * bow.soe[base + i] + bow.thermal.weight[i] * d_full;
        bow.soe[base + i] = a;
        F += a;
        let b = bow.thermal.decay[i] * bow.soe[base + BOW_SOE_TERMS + i] + bow.thermal.weight_window[i] * d_window;
        bow.soe[base + BOW_SOE_TERMS + i] = b;
        W += b;
    }
    let T_prev = bow.contacts[k].temperature;
    let T_window = bow_ring_temperature(k, m - bow.thermal.window);
    let lam = bow.thermal.lam_string + bow.thermal.lam_bow;
    let rhs = bow.thermal.c_over_h * T_prev + bow.contacts[k].heat_power + 2.0 * lam * T_prev - lam * F
              + bow.thermal.lam_bow * (W - bow.thermal.f_window * T_window);
    let T = rhs / (bow.thermal.c_over_h + bow.thermal.conv_coef * abs(bow.contacts[k].slip_velocity) + 2.0 * lam);
    bow.ring[k * BOW_RING + (m & (BOW_RING - 1u))] = T;
    bow.contacts[k].temperature = T;
    return T;
}

fn bow_calibration_temperature(u: f32) -> f32 {
    // steady sliding at speed u^2 under the Fig. 3 conditions (bowing.calibration_temperature)
    return bow_mu(u * u) * bow.thermal.cal_force * u / (bow.thermal.cal_k1 + bow.thermal.cal_k2 * u);
}

// mu(T): the friction coefficient of the steady sliding speed whose contact temperature is T
// (T(v) increases with v): bisection on u = sqrt(v) until the bracket cannot shrink.
fn bow_mu_thermal(T: f32) -> f32 {
    if (T <= 0.0) {
        return bow_mu(0.0);
    }
    var hi = 1.0;
    for (var i = 0; i < 40; i++) {
        if (bow_calibration_temperature(hi) >= T) { break; }
        hi = 2.0 * hi;
    }
    var lo = 0.0;
    for (var it = 0; it < 64; it++) {
        let mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi) { break; }
        if (bow_calibration_temperature(mid) < T) { lo = mid; } else { hi = mid; }
    }
    return bow_mu(hi * hi);
}

// Friction of contact k for this step, applied after compute_forces (which wrote the friction-free
// kicked velocities v*, omega* to the future buffers). With F the friction force (lab y) the kick
// changes the relative surface velocity v_rel = v_y - r omega_x - v_b by F dt / m_eff, where
// 1 / m_eff = 1 / m + r^2 d^T I_dyn^-1 d and d = R^T e_x (the couple -r F e_x enters the edge's
// kick in material coordinates). So the step is resolved exactly:
//   stick if |F_stick| <= mu(0) N with F_stick = -v_rel* m_eff / dt   (v_rel = 0 after the step),
//   slip otherwise: F = -s mu(x) N with x = |v_rel| after the step, x + c mu(x) = |v_rel*|,
//   c = N dt / m_eff, s = sign(v_rel*).
// g(x) = x + c mu(x) - |v_rel*| is convex (mu'' > 0): if sticking is impossible it has exactly one
// root; if sticking is possible it has none or two. In that case the contact keeps its previous
// state (McIntyre & Woodhouse's hysteresis rule): a stuck contact stays stuck, a contact slipping
// in direction s takes the larger root (the continuation of its slipping branch).
// The normal force -w N e_z is applied here as well; it exerts no couple (it passes through the
// centre line). Then node j and edge j redo their half-step drift as compute_forces does.
fn bow_contact_body(k: u32, current_base: u32, future_base: u32) {
    let j = bow.contacts[k].node;
    let cur = current_base + j;
    let fut = future_base + j;
    let dt = uniforms.dt;
    let profile = bow_profile(bow.step);
    let normal = bow.normal_force * profile * bow.contacts[k].weight;
    let v_b = bow.velocity * profile;

    var r = 0.0;
    if (bow.torsion == 1u) {
        r = bow.radius;
    }
    let d = to_material(cur, vec3<f32>(1.0, 0.0, 0.0));
    let eta = 1.0 - 0.3 * offset_stretch(edge_offset(cur), offset_length(edge_offset(cur)));
    let inertia_inv_dyn = uniforms.inertia_inv * (1.0 / (eta * eta));
    let m_eff_inv = uniforms.mass_inv + r * r * dot(d * d, inertia_inv_dyn);

    let omega_star = edges[fut].angular_velocity;
    let v_rel_star = nodes[fut].velocity.y - r * dot(d, omega_star) - v_b;
    // at the start of the step (for the work, see below)
    let v_rel_start = nodes[cur].velocity.y - r * dot(d, edges[cur].angular_velocity) - v_b;
    let speed = abs(v_rel_star);
    let s = sign(v_rel_star);
    let c = normal * dt * m_eff_inv;
    var mu_static = bow_mu(0.0);
    if (bow.model == 2u) {
        // thermal model: one limiting coefficient for this step, independent of the slip speed
        mu_static = bow_mu_thermal(bow_temperature(k, bow.step));
    }

    var stick = speed <= mu_static * c;
    var x = 0.0; // |v_rel| after the step when slipping
    if (bow.model == 2u) {
        if (!stick) {
            x = speed - mu_static * c;
        }
    } else if (!stick) {
        // unique root in (0, speed): g(0) < 0 < g(speed) = c mu(speed)
        var lo = 0.0;
        var hi = speed;
        for (var it = 0; it < 64; it++) {
            let mid = 0.5 * (lo + hi);
            if (mid <= lo || mid >= hi) { break; }
            if (mid + c * bow_mu(mid) - speed < 0.0) { lo = mid; } else { hi = mid; }
        }
        x = hi;
    } else if (bow.contacts[k].sticking == 0u && s != 0.0 && sign(bow.contacts[k].slip_velocity) == s
               && bow_mu_slope(speed) * c > -1.0 && bow_mu_slope(0.0) * c < -1.0) {
        // Was slipping in this direction: the minimum of g lies where mu'(x) c = -1, in (0, speed).
        var lo = 0.0;
        var hi = speed;
        for (var it = 0; it < 64; it++) {
            let mid = 0.5 * (lo + hi);
            if (mid <= lo || mid >= hi) { break; }
            if (bow_mu_slope(mid) * c < -1.0) { lo = mid; } else { hi = mid; }
        }
        let x_min = hi;
        if (x_min + c * bow_mu(x_min) - speed < 0.0) {
            // two roots: take the larger one, in (x_min, speed)
            lo = x_min;
            hi = speed;
            for (var it = 0; it < 64; it++) {
                let mid = 0.5 * (lo + hi);
                if (mid <= lo || mid >= hi) { break; }
                if (mid + c * bow_mu(mid) - speed < 0.0) { lo = mid; } else { hi = mid; }
            }
            x = hi;
            stick = false;
        }
    }

    var force: f32;
    var v_rel: f32;
    if (stick) {
        force = -v_rel_star / (dt * m_eff_inv);
        v_rel = 0.0;
    } else {
        if (bow.model == 2u) {
            force = -s * mu_static * normal;
        } else {
            force = -s * bow_mu(x) * normal;
        }
        v_rel = s * x;
    }

    // Kick corrections: friction along y and the normal force along -z on node j, the friction
    // couple r_c x F = -r F e_x on edge j.
    var v = nodes[fut].velocity;
    let v_z_start = nodes[cur].velocity.z;
    v.y = v.y + force * uniforms.mass_inv * dt;
    v.z = v.z - normal * uniforms.mass_inv * dt;
    nodes[fut].velocity = v;
    drift_node(cur, fut, v * dt * 0.5);

    if (r != 0.0) {
        let omega = omega_star - (r * force * dt) * (inertia_inv_dyn * d);
        edges[fut].angular_velocity = omega;
        var update = integrate_exponential(edges[cur].orientation, omega, dt * 0.5);
        if (dot(update, edges[cur].orientation) < 0.0) {
            update = -update;
        }
        edges[fut].orientation = update;
    }

    bow.contacts[k].sticking = select(0u, 1u, stick);
    bow.contacts[k].slip_velocity = v_rel;
    bow.contacts[k].force = force;
    bow.contacts[k].impulse += force * dt;
    bow.contacts[k].bow_work += force * v_b * dt;
    // A kick changes the kinetic energy by F dt times the step-average velocity (both half drifts
    // use the start and end velocities), so the work of F on the string over the step is
    // F (v_b + (v_rel_start + v_rel) / 2) dt: bow work plus friction work.
    bow.contacts[k].friction_work += force * 0.5 * (v_rel_start + v_rel) * dt;
    bow.contacts[k].heat_power = -force * 0.5 * (v_rel_start + v_rel);
    bow.contacts[k].normal_work += -normal * 0.5 * (v_z_start + v.z) * dt;
}

// Output frame record of the bow; resets the per-frame sums.
fn bow_save_body(k: u32) {
    let i = c.output_index;
    if (k == 0u) {
        let profile = bow_profile(bow.step);
        bow.frames[i].normal_force = bow.normal_force * profile;
        bow.frames[i].velocity = bow.velocity * profile;
        bow.frames[i].step = bow.step;
    }
    bow.frames[i].contacts[k] = bow.contacts[k];
    bow.contacts[k].impulse = 0.0;
    bow.contacts[k].bow_work = 0.0;
    bow.contacts[k].friction_work = 0.0;
    bow.contacts[k].normal_work = 0.0;
}

struct FrictionStep {
    force: f32,       // along t (N)
    slip: f32,        // surface speed relative to the body along t after the step
    slip_start: f32,  // the same at the start of the step
    stick: bool,
}

// Exact Coulomb friction correction of node j / edge j (cur, fut: their slots) for a contact with
// normal n (lab y, z), normal force normal_force (N) and body surface speed body_t along
// t = (0, -n_z, n_y). Same structure as the bow contact: the surface speed along t after the kick is
// v.t - r omega_x, with 1/m_eff = 1/m + r^2 d^T I_dyn^-1 d; stick if the force that stops the slip is
// at most mu_s normal_force, otherwise slip with mu_d normal_force (mu_d <= mu_s: no sign change).
// Updates the node velocity and the edge spin and orientation; the caller redoes the node drift.
fn surface_friction(cur: u32, fut: u32, n: vec2<f32>, body_t: f32, normal_force: f32,
                    mu_s: f32, mu_d: f32, r: f32) -> FrictionStep {
    let dt = uniforms.dt;
    let t = vec3<f32>(0.0, -n.y, n.x);
    let d = to_material(cur, vec3<f32>(1.0, 0.0, 0.0));
    let eta = 1.0 - 0.3 * offset_stretch(edge_offset(cur), offset_length(edge_offset(cur)));
    let inertia_inv_dyn = uniforms.inertia_inv * (1.0 / (eta * eta));
    let m_eff_inv = uniforms.mass_inv + r * r * dot(d * d, inertia_inv_dyn);

    let omega_star = edges[fut].angular_velocity;
    let v_star = dot(nodes[fut].velocity, t) - r * dot(d, omega_star) - body_t;
    let v_start = dot(nodes[cur].velocity, t) - r * dot(d, edges[cur].angular_velocity) - body_t;

    var out: FrictionStep;
    out.slip_start = v_start;
    if (abs(v_star) <= mu_s * normal_force * dt * m_eff_inv) {
        out.force = -v_star / (dt * m_eff_inv);
        out.slip = 0.0;
        out.stick = true;
    } else {
        out.force = -sign(v_star) * mu_d * normal_force;
        out.slip = v_star + out.force * dt * m_eff_inv;
        out.stick = false;
    }

    nodes[fut].velocity = nodes[fut].velocity + (out.force * uniforms.mass_inv * dt) * t;
    if (r != 0.0) {
        let omega = omega_star - (r * out.force * dt) * (inertia_inv_dyn * d);
        edges[fut].angular_velocity = omega;
        var update = integrate_exponential(edges[cur].orientation, omega, dt * 0.5);
        if (dot(update, edges[cur].orientation) < 0.0) {
            update = -update;
        }
        edges[fut].orientation = update;
    }
    return out;
}

// Distance the pick tip has moved along +y at pick clock `step`: speed 0 until start_step, rising
// linearly to `velocity` over ramp_steps, then constant (the exact integral over whole steps).
fn pick_travel(step: u32) -> f32 {
    let k = f32(i32(step) - i32(surface.pick.start_step));
    if (k <= 0.0) {
        return 0.0;
    }
    let ramp = f32(max(surface.pick.ramp_steps, 1u));
    let per_step = surface.pick.velocity * uniforms.dt;
    if (k <= ramp) {
        return per_step * k * k / (2.0 * ramp);
    }
    return per_step * (k - 0.5 * ramp);
}

// Pick velocity over the step from clock `step` to step + 1: (travel(step + 1) - travel(step)) / dt.
fn pick_step_velocity(step: u32) -> f32 {
    let k = f32(i32(step) - i32(surface.pick.start_step));
    if (k < 0.0) {
        return 0.0;
    }
    let ramp = f32(max(surface.pick.ramp_steps, 1u));
    if (k + 1.0 <= ramp) {
        return surface.pick.velocity * (2.0 * k + 1.0) / (2.0 * ramp);
    }
    return surface.pick.velocity;
}

// Rigid plectrum contact on node j (see struct Pick). Normal: the gap g between the string's circle
// and the pick's capsule at this force evaluation, with the pick moving by v_pick dt to the next one;
// the normal impulse is the smallest that keeps g + (v - v_pick).n dt >= 0 (non-penetration at the
// next evaluation, linear in the step), lambda_n = m max(0, -g/dt - v_n*). Rotation does not move the
// contact point along n. Then Coulomb friction along t with the limit mu normal force.
fn pick_contact_body(current_base: u32, future_base: u32) {
    let j = surface.pick.node;
    let cur = current_base + j;
    let fut = future_base + j;
    let dt = uniforms.dt;
    let step = surface.pick.step;

    let tip = vec2<f32>(surface.pick.tip_y0 + pick_travel(step), surface.pick.tip_z);
    let axis = vec2<f32>(surface.pick.axis_y, surface.pick.axis_z);
    let v_pick = vec2<f32>(pick_step_velocity(step), 0.0);
    surface.pick.tip_y = tip.x;

    let centre = nodes[cur].displacement.yz;
    let along = clamp(dot(centre - tip, axis), 0.0, surface.pick.length);
    let rel = centre - (tip + along * axis);
    let dist = length(rel);
    let gap = dist - (surface.pick.radius + surface.pick.tip_radius);
    let n = rel / dist;
    surface.pick.max_penetration = max(surface.pick.max_penetration, -gap);

    let v_n_star = dot(nodes[fut].velocity.yz - v_pick, n);
    let v_n_start = dot(nodes[cur].velocity.yz - v_pick, n);
    let impulse_n = max(0.0, -gap / dt - v_n_star) / uniforms.mass_inv;
    let normal_force = impulse_n / dt;

    var record = surface.pick.contact;
    record.normal_y = n.x;
    record.normal_z = n.y;
    record.normal_force = normal_force;
    if (normal_force > 0.0) {
        nodes[fut].velocity = nodes[fut].velocity + vec3<f32>(0.0, n * (impulse_n * uniforms.mass_inv));
        let v_n_end = v_n_star + impulse_n * uniforms.mass_inv;

        var r = 0.0;
        if (surface.pick.torsion == 1u) {
            r = surface.pick.radius;
        }
        let body_t = -v_pick.x * n.y;   // v_pick . t
        let f = surface_friction(cur, fut, n, body_t, normal_force, surface.pick.mu_s, surface.pick.mu_d, r);
        drift_node(cur, fut, nodes[fut].velocity * dt * 0.5);

        record.in_contact = 1u;
        record.sticking = select(0u, 1u, f.stick);
        record.slip_velocity = f.slip;
        record.friction_force = f.force;
        record.body_work += (normal_force * dot(v_pick, n) + f.force * body_t) * dt;
        record.friction_work += f.force * 0.5 * (f.slip_start + f.slip) * dt;
        record.normal_work += normal_force * 0.5 * (v_n_start + v_n_end) * dt;
        record.impulse_t += f.force * dt;
        if (surface.pick.first_contact_step == 0u) {
            surface.pick.first_contact_step = step;
        }
        surface.pick.last_contact_step = step;
    } else {
        record.in_contact = 0u;
        record.sticking = 0u;
        record.slip_velocity = 0.0;
        record.friction_force = 0.0;
    }
    surface.pick.contact = record;
}

// Felt friction on the hammer's contact nodes (thread 0, before hammer_update_body_with). Normal
// force and normal from compute_forces (the felt law, unchanged); the felt surface moves with the
// hammer velocity of the start of the step, v_h t_z (the felt law is explicit in the hammer too).
// Returns the z component of the friction forces on the string; the hammer takes its reaction in
// the same step.
fn hammer_friction_body(current_base: u32, future_base: u32) -> f32 {
    let dt = uniforms.dt;
    var r = 0.0;
    if (surface.hammer.torsion == 1u) {
        r = surface.hammer.radius;
    }
    var force_z = 0.0;
    for (var k = 0u; k < surface.hammer.count; k++) {
        let j = surface.hammer.contacts[k].node;
        let cur = current_base + j;
        let fut = future_base + j;
        let h = surface.hammer_normal[j];
        var record = surface.hammer.contacts[k];
        record.normal_y = h.x;
        record.normal_z = h.y;
        record.normal_force = h.z;
        if (h.w == 1.0 && h.z > 0.0) {
            let n = h.xy;
            let body_t = hammer.velocity * n.x;   // (v_h e_z) . t
            let f = surface_friction(cur, fut, n, body_t, h.z, surface.hammer.mu_s, surface.hammer.mu_d, r);
            drift_node(cur, fut, nodes[fut].velocity * dt * 0.5);
            force_z += f.force * n.x;
            record.in_contact = 1u;
            record.sticking = select(0u, 1u, f.stick);
            record.slip_velocity = f.slip;
            record.friction_force = f.force;
            record.body_work += f.force * body_t * dt;
            record.friction_work += f.force * 0.5 * (f.slip_start + f.slip) * dt;
            record.impulse_t += f.force * dt;
        } else {
            record.in_contact = 0u;
            record.sticking = 0u;
            record.slip_velocity = 0.0;
            record.friction_force = 0.0;
        }
        surface.hammer.contacts[k] = record;
    }
    return force_z;
}

// hammer_update_body with the felt friction: the hammer also takes the reaction -extra_force_z of
// the friction forces; records the work of that reaction (the hammer's kinetic energy changes by
// F dt times its step-average velocity).
fn hammer_update_body_with(extra_force_z: f32) {
    let v_old = hammer.velocity;
    let total_force = f32(atomicLoad(&hammer.force_accumulator)) / 10000.0 + extra_force_z;
    let hammer_acceleration = - total_force * hammer.mass_inv;
    hammer.velocity = hammer.velocity + (hammer_acceleration * uniforms.dt);
    hammer.displacement= hammer.displacement + (hammer.velocity * uniforms.dt * 0.5);
    atomicStore(&hammer.force_accumulator, 0);
    surface.hammer.hammer_work += -extra_force_z * 0.5 * (v_old + hammer.velocity) * uniforms.dt;
}

// Output frame record of the pick and the hammer friction; resets the per-frame sums.
fn surface_save_body() {
    let i = c.output_index;
    surface.hammer.hammer_displacement = hammer.displacement;
    surface.hammer.hammer_velocity = hammer.velocity;
    surface.frames[i].pick = surface.pick;
    surface.frames[i].hammer = surface.hammer;
    surface.pick.contact.body_work = 0.0;
    surface.pick.contact.friction_work = 0.0;
    surface.pick.contact.normal_work = 0.0;
    surface.pick.contact.impulse_t = 0.0;
    surface.hammer.hammer_work = 0.0;
    for (var k = 0u; k < surface.hammer.count; k++) {
        surface.hammer.contacts[k].body_work = 0.0;
        surface.hammer.contacts[k].friction_work = 0.0;
        surface.hammer.contacts[k].normal_work = 0.0;
        surface.hammer.contacts[k].impulse_t = 0.0;
    }
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
            if (bow.model != 0u) {
                bow.step += 1u;
            }
            if (surface.pick.enabled == 1u) {
                surface.pick.step += 1u;
            }
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
            if (surface.hammer.enabled == 1u && c.hammering == 1u) {
                hammer_update_body_with(hammer_friction_body(current_base, future_base));
            } else {
                hammer_update_body();
            }
        }
        // Bow contacts and the pick (independent of each other and of the hammer state: each acts
        // on its own node and edge)
        if (bow.model != 0u && t < bow.contact_count) {
            bow_contact_body(t, current_base, future_base);
        }
        if (surface.pick.enabled == 1u && t == MAX_BOW_CONTACTS) {
            pick_contact_body(current_base, future_base);
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
        if (bow.model != 0u && t < bow.contact_count) {
            bow_save_body(t);
        }
        if (t == MAX_BOW_CONTACTS && (surface.pick.enabled == 1u || surface.hammer.enabled == 1u
                                      || surface.hammer.curved == 1u)) {
            surface_save_body();
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
    return edges[i].reference_offset + (node_difference(i) * edges[i].rest_frame);
}

// u_{i+1} - u_i from the two-part displacements: the difference of the high parts of neighbours
// is exact (or rounded at the size of the difference, not of the displacement), the low parts
// add what the high parts cannot resolve.
fn node_difference(i: u32) -> vec3<f32> {
    return (nodes[i + 1].displacement - nodes[i].displacement)
        + (nodes[i + 1].displacement_lo - nodes[i].displacement_lo);
}

// x itself, but opaque to the shader compiler: WGSL cannot mark arithmetic exact, and the driver's
// optimiser may rewrite float expressions algebraically ((a + b) - a -> b, Mesa NIR does), which
// turns the rounding error that TwoSum measures into 0. A bit operation with a runtime value it cannot
// know stops such rewrites.
fn opaque(x: vec3<f32>) -> vec3<f32> {
    return bitcast<vec3<f32>>(bitcast<vec3<u32>>(x) ^ vec3<u32>(uniforms.opaque_zero));
}

// Drift u(fut) = u(cur) + increment with the two-part displacement: Knuth's TwoSum of the high part
// and (increment + low part) gives the rounded sum and its exact rounding error, which becomes the
// new low part. cur == fut updates in place.
fn drift_node(cur: u32, fut: u32, increment: vec3<f32>) {
    let a = nodes[cur].displacement;
    let b = increment + nodes[cur].displacement_lo;
    let s = opaque(a + b);
    let b_virtual = opaque(s - a);
    let a_virtual = s - b_virtual;
    nodes[fut].displacement = s;
    nodes[fut].displacement_lo = (a - a_virtual) + (b - b_virtual);
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
