struct Node {
    displacement: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    velocity: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    curvature: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
   
    reference_rotation: vec4<f32>,

    internal_moments: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
}

struct Edge {
    orientation: vec4<f32>, // 16 bytes

    angular_velocity: vec3<f32>, // 12 bytes
    len: f32,

    strain: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
    
    reference_strain: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    internal_force: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
    
    reference_vector: vec3<f32>,
    // 4 bytes of implicit padding
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
}

struct PushConstants { 
    current: u32, // either 0 or 1
    future: u32,  // current == 0 ? 1 : 0
    output_index: u32,
    force: f32,

    linear_dampening: f32,
    angular_dampening: f32,
    _pad0: u32,
    _pad1: u32,
}

var <push_constant> c: PushConstants;

@group(0)
@binding(0)
var<uniform> uniforms: Uniforms;

@group(0)
@binding(1)
var<storage, read> hammer_weights: array<vec3<f32>>;


////////////////////
//  Node bindings //
////////////////////

@group(0)
@binding(2)
var<storage, read_write> displacements:  array<vec3<f32>>;

@group(0)
@binding(3)
var<storage, read_write> velocities: array<vec3<f32>>;

@group(0)
@binding(4)
var<storage, read_write> curvatures: array<vec3<f32>>;

@group(0)
@binding(5)
var<storage, read_write> reference_rotations: array<vec4<f32>>;

@group(0)
@binding(6)
var<storage, read_write> internal_moments: array<vec3<f32>>;


///////////////////
// Edge bindings //
///////////////////

@group(0)
@binding(7)
var<storage, read_write> orientations: array<vec4<f32>>;

@group(0)
@binding(8)
var<storage, read_write> angular_velocities: array<vec3<f32>>;

@group(0)
@binding(9)
var<storage, read_write> strains: array<vec3<f32>>;

@group(0)
@binding(10)
var<storage, read_write> reference_strains: array<vec3<f32>>;

@group(0)
@binding(11)
var<storage, read_write> internal_forces: array<vec3<f32>>;

@group(0)
@binding(12)
var<storage, read_write> reference_vectors: array<vec3<f32>>;

@group(0)
@binding(13)
var<storage, read_write> inverse_lengths: array<f32>;

@group(0)
@binding(14)
var<storage, read_write> dilatations: array<f32>;


/////////////////////
// Output bindings //
/////////////////////

@group(0)
@binding(15)
var<storage, read_write> output_nodes: array<Node>;

@group(0)
@binding(16)
var<storage, read_write> output_edges: array<Edge>;

@compute
@workgroup_size(64) 
fn init(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;
    let future = (c.future * uniforms.node_count) + global_id.x;

    let ref_vec = reference_vectors[current];
    let disp_this = displacements[current];
    let disp_next = displacements[current + 1];
    
    if (global_id.x < uniforms.node_count - 1) {
        let r = ref_vec + (disp_next - disp_this);

        let dilatation = length(r) * uniforms.dl_inv;
        dilatations[current] = dilatation;
        dilatations[future] = dilatation;
    }
}

@compute
@workgroup_size(64) 
fn create_references(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;
    let future = (c.future * uniforms.node_count) + global_id.x;

    let ref_vec = reference_vectors[current];
    let disp = displacements[current];
    let disp_next = displacements[current + 1];
    
    let ori = orientations[current];
    let ori_next = orientations[current + 1];
    
    if (global_id.x < uniforms.node_count - 1) {
        let r = ref_vec + (disp_next - disp);
        let tangent_LF = r * uniforms.dl_inv;
        let tangent_MF = rotate_inv(ori, tangent_LF);
        let current_strain_MF = tangent_MF - vec3<f32>(0.0, 0.0, 1.0);
        
        strains[current]= current_strain_MF;
        strains[future] = current_strain_MF;
    }

    if (global_id.x < uniforms.node_count - 2) {
        let relative_orientation = qmul(qinv(ori), ori_next);

        reference_rotations[current] = relative_orientation;
        reference_rotations[future] = relative_orientation;
    }
}
@compute
@workgroup_size(64) 
fn half_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;

    var disp_this = displacements[current];
    let vel = velocities[current];
    var ori = orientations[current];
    let ang_vel = angular_velocities[current];

    // Half step position update (drift)
    if (global_id.x > (0 + uniforms.clamp_offset) && global_id.x < (uniforms.node_count - 2 - uniforms.clamp_offset)) {
        var update = integrate_exponential(ori, ang_vel, uniforms.dt * 0.5);
        // Safety: make sure future orientation selects pole closest to current orientation
        if (dot(update, ori) < 0.0) {
            update = -update;
        }
        
        orientations[current] = update;
    }

    if (global_id.x > 0 && global_id.x < uniforms.node_count - 1) {
        disp_this = disp_this + (vel * uniforms.dt * 0.5);
        displacements[current] = disp_this;
    }
}

@compute
@workgroup_size(64) 
fn compute_internals(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;
    let future = (c.future * uniforms.node_count) + global_id.x;

    let ref_vec = reference_vectors[current];
    let ref_vec_next = reference_vectors[current + 1];
    var disp_this = displacements[current];
    let disp_next = displacements[current + 1];
    let disp_nextnext = displacements[current + 2];
    let vel = velocities[current];

    var ori = orientations[current];
    let ori_next = orientations[current + 1];
    let ang_vel = angular_velocities[current];
    
    let rot_ref = reference_rotations[current];
    let ref_strain = reference_strains[current];


    // Calculate internal responses
    if (global_id.x < uniforms.node_count - 1) {

        let r = ref_vec + (disp_next - disp_this);
        let tangent_LF = r * uniforms.dl_inv;
        let tangent_MF = rotate_inv(ori, tangent_LF);

        let current_strain_MF = tangent_MF - vec3<f32>(0.0, 0.0, 1.0);
        let len_inv = 1.0 / length(r);

        let dilatation_inv = uniforms.dl * len_inv;

        let sigma_eff = current_strain_MF - ref_strain; // (MF)

        inverse_lengths[current] = len_inv;
        strains[future] = sigma_eff;
        internal_forces[current]= rotate(ori, (uniforms.stiffness_se * sigma_eff * dilatation_inv)) ; // (LF)
        dilatations[future] = 1.0 / dilatation_inv;
    }

    if ( global_id.x < uniforms.node_count - 2) {
        // NB: The paper chooses its indexing in a way that makes this somewhat weird. nodes[i] stores the average
        // curvature whose integral is rotation needed to from edges[i].orientation to edges[i+1].orientation.
        // Therefore this convention causes the i-th region to contain node i+1. This must be accounted for when
        // defining the descrete difference/average operators (i.e. here one must use backwards differences)!
        
        let r = ref_vec_next + (disp_nextnext - disp_next);
        let next_len_inv = 1.0 / length(r);
    
        var relative_orientation = qmul(qinv(ori), ori_next);

        // Select closest quaternion pole
        if (dot(relative_orientation, rot_ref) < 0.0) {
            relative_orientation = -relative_orientation;
        }

        var effective_orientation = qmul(qinv(rot_ref), relative_orientation); // (MF)
        // if (effective_orientation.w < 0.0) { effective_orientation = -effective_orientation; }

        let len_inv = 1.0 / length(r);
        let epsilon_inv = ((len_inv * next_len_inv) / (len_inv + next_len_inv)) * uniforms.dl * 2.0;

        // This is B applied to kappa (MF); here B is diagonal -> pointwise multiplication
        let kappa = 2.0 * uniforms.dl_inv * effective_orientation.xyz;
        curvatures[current] = kappa;
        internal_moments[current]= uniforms.stiffness_bt * kappa * epsilon_inv * epsilon_inv * epsilon_inv;
    }
}


@compute
@workgroup_size(64) 
fn compute_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;
    let future = (c.future * uniforms.node_count) + global_id.x;

    let intf_prev = internal_forces[current - 1];
    let intf = internal_forces[current];

    let vel = velocities[current];
    let disp = displacements[current];
    let disp_next = displacements[current];

    let ori = orientations[current];
    let ang_vel = angular_velocities[current];
    let len_inv = inverse_lengths[current];
    let ref_vec = reference_vectors[current];

    let intm = internal_moments[current];
    let curv = curvatures[current];
    let intm_prev = internal_moments[current - 1];
    let curv_prev = curvatures[current - 1];

    let dil = dilatations[future];
    let dil_prior = dilatations[current];


    // Interior nodes
    if (global_id.x > 0 && global_id.x < uniforms.node_count - 1) {
        let ext_force = hammer_weights[global_id.x].xyz * c.force;
        let damping_force = - c.linear_dampening * vel;
        
        // Only calculate for interior edges, so naive difference operator is fine here!
        let v_tt = (intf - intf_prev + ext_force + damping_force) * vec3<f32>(0.1, 1.0, 1.0) * uniforms.mass_inv;

        // Full step velocity update (kick)
        let vel_future = vel + (v_tt * uniforms.dt);
        velocities[future] = vel_future;
        // Half step position update (drift)
        displacements[future] = disp + (vel_future * uniforms.dt * 0.5);
    } else {
        velocities[future] = vec3<f32>(0.0);
    }

    
    if (global_id.x > (0 + uniforms.clamp_offset) && global_id.x < (uniforms.node_count - 2 - uniforms.clamp_offset)) {
        
        let r = ref_vec + (disp_next - disp_next);
        let tangent_LF = r * uniforms.dl_inv;
        let tangent_MF = rotate_inv(ori, tangent_LF);
        let dilatation_inv = uniforms.dl * len_inv;
        let dilatation_t= (dil - dil_prior) * uniforms.dt_inv;

        let ext_couple = vec3<f32>(0.0, 0.0, 0.0);
        let damping_torque = - c.angular_dampening * ang_vel;

        var difference: vec3<f32>;
        var average: vec3<f32>;
        if (global_id.x == 0) {
            difference = intm;
            average = 0.5 * uniforms.dl * cross(curv, intm);
        } else if (global_id.x == uniforms.node_count - 2) {
            difference = - intm_prev;
            average = 0.5 * uniforms.dl * cross(curv_prev, intm_prev);
        } else {
            difference = intm - intm_prev;
            average = 0.5 * uniforms.dl * (cross(curv, intm) + cross(curv_prev, intm_prev));
        }

    
        let phi_tt =( ( difference
                            + average
                            + (cross(tangent_MF, rotate_inv(ori, intf)) * uniforms.dl)
                            + ext_couple
                            + damping_torque
                            ) * dil
                            + cross((uniforms.inertia * ang_vel), ang_vel) 
                            + uniforms.inertia * ang_vel * dilatation_inv * dilatation_t
                            ) * uniforms.inertia_inv;


        // Full step velocity update (kick)
        angular_velocities[future] = ang_vel + (phi_tt * uniforms.dt);
        // Half step position update (drift)
        var update = integrate_exponential(ori, ang_vel, uniforms.dt * 0.5);
        // Safety: make sure future orientation selects pole closest to current orientation
        if (dot(update, ori) < 0.0) {
            update = -update;
        }
        
        orientations[future] = update;
    }
}

@compute
@workgroup_size(64) 
fn save_output(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = (c.output_index * uniforms.node_count) + global_id.x;
    let current = (c.current * uniforms.node_count) + global_id.x;

    let disp =displacements[current];
    let vel = velocities[current];
    let curv = curvatures[current];
    let intm = internal_moments[current];
    let ref_rot = reference_rotations[current];

    let ori = orientations[current];
    let ang_vel = angular_velocities[current];
    let len = 1.0 / inverse_lengths[current];
    let strain = strains[current];
    let ref_strain = reference_strains[current];
    let intf = internal_forces[current];
    let ref_vec = reference_vectors[current];

    let node = Node(disp, vel, curv, ref_rot, intm);
    let edge = Edge(ori, ang_vel, len, strain, ref_strain, intf, ref_vec);

    output_nodes[i] = node;
    output_edges[i] = edge;
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
    
    let next_q = qmul(q, dq);
    
    return normalize(next_q);
}
