struct Node {
    position: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    velocity: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    curvature: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
   
    reference_orientation: vec4<f32>, // 16 bytes
    // this is the quaterion encoding the rotation needed to transform
    // the one set of adjacent directors to the other adjacent one at rest
        
    internal_moment: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
}

struct Edge {
// alignment is 16 bytes
    orientation: vec4<f32>, // 16 bytes

    angular_velocity: vec3<f32>, // 12 bytes
    len_inv: f32, // 12 bytes

    strain: vec3<f32>, // 12 bytes
    dilatation: f32, // 4 bytes; this is actually the previous dilatation.
    
    reference_strain: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    internal_force: vec3<f32>, // 12 bytes
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
    pad0: u32,
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

@compute
@workgroup_size(64) 
fn create_reference(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;
    let future = (c.future * uniforms.node_count) + global_id.x;


    if (global_id.x < uniforms.node_count - 1) {
        // This must match the strain calculation performed when calculating internal forces!
        // Otherwise numerical inaccuracies inject energy into the system!
        let tangent_LF = (nodes[current + 1].position - nodes[current].position) * uniforms.dl_inv;
        let tangent_MF = rotate_inv(edges[current].orientation, tangent_LF);
        var current_strain_MF = tangent_MF - vec3<f32>(0.0, 0.0, 1.0);
        current_strain_MF.z = 0.0;
        edges[current].reference_strain = current_strain_MF;
        edges[future].reference_strain = current_strain_MF;

        var reference_orientation = qmul(qinv(edges[current].orientation), edges[current+1].orientation);
        // nodes[current].reference_orientation = reference_orientation;
        // nodes[future].reference_orientation = reference_orientation;
        nodes[current].reference_orientation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        nodes[future].reference_orientation = vec4<f32>(0.0, 0.0, 0.0, 1.0);

        // Finally initialize the dilatations
        let dilatation = length(nodes[current+1].position - nodes[current].position) * uniforms.dl_inv;
        edges[current].dilatation = dilatation;
        edges[future].dilatation = dilatation;
    }
}

@compute
@workgroup_size(64) 
fn half_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;

    // Half step position update (drift)
    if (global_id.x > 0 && global_id.x < uniforms.node_count - 2) {
        let dq = 0.5 * qmul(edges[current].orientation, vec4<f32>(edges[current].angular_velocity, 0.0));
        var update = integrate_exponential(edges[current].orientation, edges[current].angular_velocity, uniforms.dt * 0.5);
        // Safety: make sure future orientation selects pole closest to current orientation
        if (dot(update, edges[current].orientation) < 0.0) {
            update = -update;
        }
        
        edges[current].orientation = update;
    }

    if (global_id.x > 0 && global_id.x < uniforms.node_count - 1) {
        nodes[current].position = nodes[current].position + (nodes[current].velocity * uniforms.dt * 0.5);
    }
}

@compute
@workgroup_size(64) 
fn compute_internals(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;
    let future = (c.future * uniforms.node_count) + global_id.x;
    var len_inv: f32;

    // Only run this segment on edges
    if (global_id.x < uniforms.node_count - 1) {
        let r = (nodes[current+1].position - nodes[current].position);
        len_inv = inverseSqrt(dot(r,r));
        edges[current].len_inv = len_inv;

        let dilatation_inv = uniforms.dl * len_inv;


        let tangent_LF = (nodes[current + 1].position - nodes[current].position) * uniforms.dl_inv;
        let tangent_MF = rotate_inv(edges[current].orientation, tangent_LF);
        let current_strain_MF = tangent_MF - vec3<f32>(0.0, 0.0, 1.0);

        
        let sigma_eff = current_strain_MF - edges[current].reference_strain; // (MF)
        edges[future].strain = sigma_eff;
        edges[current].internal_force = rotate(edges[current].orientation, (uniforms.stiffness_se * sigma_eff * dilatation_inv)) ; // (LF)
        edges[future].dilatation = 1.0 / dilatation_inv;
    }

    if (global_id.x < uniforms.node_count - 2) {
        // NB: The paper chooses its indexing in a way that makes this somewhat weird. nodes[i] stores the average
        // curvature whose integral is rotation needed to from edges[i].orientation to edges[i+1].orientation.
        // Therefore this convention causes the i-th region to contain node i+1. This must be accounted for when
        // defining the descrete difference/average operators (i.e. here one must use backwards differences)!
        
        let r = nodes[current+2].position - nodes[current+1].position;
        let next_len_inv = inverseSqrt(dot(r,r));
    
        var relative_orientation = qmul(qinv(edges[current].orientation), edges[current+1].orientation);

        // Select closest quaternion pole
        if (dot(relative_orientation, nodes[current].reference_orientation) < 0.0) {
            relative_orientation = -relative_orientation;
        }

        let effective_orientation = qmul(qinv(nodes[current].reference_orientation), relative_orientation); // (MF)

        let epsilon_inv = ((len_inv * next_len_inv) / (len_inv + next_len_inv)) * uniforms.dl * 2.0;

        // This is B applied to kappa (MF); here B is diagonal -> pointwise multiplication
        let kappa = 2.0 * uniforms.dl_inv * effective_orientation.xyz;
        nodes[current].curvature = kappa;
        nodes[current].internal_moment = uniforms.stiffness_bt * kappa * epsilon_inv * epsilon_inv * epsilon_inv;
    }
}


@compute
@workgroup_size(64) 
fn compute_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;
    let future = (c.future * uniforms.node_count) + global_id.x;

    // Interior nodes
    if (global_id.x > 0 && global_id.x < uniforms.node_count - 1) {
        let ext_force = vec3<f32>(0.0, 0.0, 0.0);
        let damping_force = - 0.0001 * nodes[current].velocity;
        
        // Only calculate for interior edges, so naive difference operator is fine here!
        let v_tt = (edges[current].internal_force - edges[current- 1].internal_force + ext_force + damping_force) * uniforms.mass_inv;

        // Full step velocity update (kick)
        nodes[future].velocity = nodes[current].velocity + (v_tt * uniforms.dt);
        // Half step position update (drift)
        nodes[future].position = nodes[current].position + (nodes[future].velocity * uniforms.dt * 0.5);
    } else {
        nodes[future].velocity = vec3<f32>(0.0);
    }

    
    if (global_id.x > 0 && global_id.x < uniforms.node_count - 2) {
        
        let tangent_MF = rotate_inv(edges[current].orientation, (nodes[current+1].position - nodes[current].position) * edges[current].len_inv);
        let dilatation_inv = uniforms.dl * edges[current].len_inv;
        let dilatation_t= (edges[future].dilatation - edges[current].dilatation) * uniforms.dt_inv;

        let ext_couple = vec3<f32>(0.0, 0.0, 0.0);
        let damping_torque = - 0.0001 * edges[current].angular_velocity;

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

    
        let phi_tt =( ( difference
                            + average
                            + (cross(tangent_MF, rotate_inv(edges[current].orientation, edges[current].internal_force)) * uniforms.dl)
                            + ext_couple
                            + damping_torque
                            ) * edges[current].dilatation
                            + cross((uniforms.inertia * edges[current].angular_velocity), edges[current].angular_velocity) 
                            + uniforms.inertia * edges[current].angular_velocity * dilatation_inv * dilatation_t
                            ) * uniforms.inertia_inv;


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
@workgroup_size(64) 
fn save_output(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = (c.output_index * uniforms.node_count) + global_id.x;
    let current = (c.current * uniforms.node_count) + global_id.x;
    output_nodes[i] = nodes[current];
    output_edges[i] = edges[current];
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

fn tangent(q: vec4<f32>) -> vec3<f32> {
    let d3 = vec3<f32>(
        2 * (q.x * q.z + q.w * q.y),
        2 * (q.y * q.z - q.w * q.x),
        1 - 2 * (q.x * q.x + q.y * q.y)
    );

    return normalize(d3);
}

fn fast_normalize(q: vec4<f32>) -> vec4<f32> {
    return q * inverseSqrt(dot(q,q));
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
    
    return fast_normalize(next_q);
}
