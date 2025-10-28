// alignOf(vec3<f32>) is 16 but sizeOf(vec3<f32>) is only 12
// this causes implicit padding
// this could be packed more efficiently (TODO)
struct Node {
    positions: vec3<f32>,
    p1: u32,
    velocities: vec3<f32>,
    p2: u32,
    // implicit padding 
    angles: vec3<f32>,
    p3: u32,
    // implicit padding 
    angular_velocities: vec3<f32>,
    p4: u32,
    // implicit padding 
    moments: vec3<f32>,
    p5: u32,
    // implicit padding 
    forces: vec3<f32>,
    p6: u32,
    
}

// this could be packed more efficiently (TODO)
struct Uniforms {
    dt: f32, 
    ds: f32,
    node_count: u32,
    loss: f32,

    tau: f32,
    kappa: f32,
    m_coil: f32,
    c2_core: f32,

    beta: vec3<f32>,
    // implicit padding 
    sigma: vec3<f32>,
    // implicit padding 
    k: vec3<f32>,
    // implicit padding 
}

struct PushConstants { 
    current_index: u32,
    future_index: u32,
    vel: f32,
    _pad2: u32,
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
var<storage, read_write> output_buffer: array<Node>;

@compute
@workgroup_size(64) 
fn external_force(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if id == 10 {
        let current = id + (c.current_index * uniforms.node_count);

        nodes[current].velocities.x += c.vel * uniforms.dt;

    }
}

@compute
@workgroup_size(64) 
fn calculate_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if id > 0 && id < uniforms.node_count - 1 {
        let current = id + (c.current_index * uniforms.node_count);
        let future = id + (c.future_index * uniforms.node_count);

        let theta1_s = ((nodes[current+1].angles.x) - (nodes[current-1].angles.x)) / (2 * uniforms.ds);
        let theta2_s = ((nodes[current+1].angles.y) - (nodes[current-1].angles.y)) / (2 * uniforms.ds);
        let theta3_s = ((nodes[current+1].angles.z) - (nodes[current-1].angles.z)) / (2 * uniforms.ds);

        let u_s = ((nodes[current+1].positions.x) - (nodes[current-1].positions.x)) / (2 * uniforms.ds);
        let v_s = ((nodes[current+1].positions.y) - (nodes[current-1].positions.y)) / (2 * uniforms.ds);
        let w_s = ((nodes[current+1].positions.z) - (nodes[current-1].positions.z)) / (2 * uniforms.ds);

        let m1 = (theta1_s + (uniforms.tau * nodes[current].angles.y)) * uniforms.beta.x;
        let m2 = (theta2_s - (uniforms.tau * nodes[current].angles.x) + (uniforms.kappa * nodes[current].angles.z)) * uniforms.beta.y;
        let m3 = (theta3_s - (uniforms.kappa * nodes[current].angles.y)) * uniforms.beta.z;

        let q1 = (u_s - (uniforms.tau * nodes[current].positions.y) + (uniforms.kappa * nodes[current].positions.z) - nodes[current].angles.x) * uniforms.sigma.x;
        let q2 = (v_s + (uniforms.tau * nodes[current].positions.x) + nodes[current].angles.y) * uniforms.sigma.y;
        let p = (w_s - (uniforms.kappa * nodes[current].positions.x)) * uniforms.sigma.z;

        nodes[future].moments.x = m1;
        nodes[future].moments.y = m2;
        nodes[future].moments.z = m3;

        nodes[future].forces.x = q1;
        nodes[future].forces.y = q2;
        nodes[future].forces.z = p;
    }
}

@compute
@workgroup_size(64) 
fn calculate_displacements(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if id > 0 && id < uniforms.node_count - 1 {
        let current = id + (c.current_index * uniforms.node_count);
        let future = id + (c.future_index * uniforms.node_count);

        
        let m1_s = ((nodes[future+1].moments.x) - (nodes[future-1].moments.x)) / (2 * uniforms.ds);
        let m2_s = ((nodes[future+1].moments.y) - (nodes[future-1].moments.y)) / (2 * uniforms.ds);
        let m3_s = ((nodes[future+1].moments.z) - (nodes[future-1].moments.z)) / (2 * uniforms.ds);

        let q1_s = ((nodes[future+1].forces.x) - (nodes[future-1].forces.x)) / (2 * uniforms.ds);
        let q2_s = ((nodes[future+1].forces.y) - (nodes[future-1].forces.y)) / (2 * uniforms.ds);
        let p_s = ((nodes[future+1].forces.z) - (nodes[future-1].forces.z)) / (2 * uniforms.ds);

        let theta1_tt = (m1_s + nodes[future].forces.x + (uniforms.tau * nodes[future].moments.y)) / (uniforms.k.x * uniforms.m_coil);
        let theta2_tt = (m2_s - nodes[future].forces.y - (uniforms.tau * nodes[future].moments.x) + (uniforms.kappa * nodes[future].moments.z)) / (uniforms.k.y * uniforms.m_coil);
        let theta3_tt = (m3_s - (uniforms.kappa * nodes[future].moments.y)) / (uniforms.k.z * uniforms.m_coil);

        let u_tt = (q1_s - (uniforms.tau * nodes[future].forces.y) + (uniforms.kappa * nodes[future].forces.z)) / uniforms.m_coil;
        let v_tt = (q2_s + (uniforms.tau * nodes[future].forces.x)) / uniforms.m_coil;
        let w_tt = (p_s - (uniforms.kappa * nodes[future].forces.x)) / uniforms.m_coil;

        nodes[future].velocities.x = (nodes[current].velocities.x + (u_tt * uniforms.dt)) * uniforms.loss;
        nodes[future].velocities.y = (nodes[current].velocities.y + (v_tt * uniforms.dt)) * uniforms.loss;
        nodes[future].velocities.z = (nodes[current].velocities.z + (w_tt * uniforms.dt)) * uniforms.loss;

        nodes[future].angular_velocities.x = (nodes[current].angular_velocities.x + (theta1_tt * uniforms.dt)) * uniforms.loss;
        nodes[future].angular_velocities.y = (nodes[current].angular_velocities.y + (theta2_tt * uniforms.dt)) * uniforms.loss;
        nodes[future].angular_velocities.z = (nodes[current].angular_velocities.z + (theta3_tt * uniforms.dt)) * uniforms.loss;

        nodes[future].positions.x = nodes[current].positions.x + (nodes[future].velocities.x * uniforms.dt);
        nodes[future].positions.y = nodes[current].positions.y + (nodes[future].velocities.y * uniforms.dt);
        nodes[future].positions.z = nodes[current].positions.z + (nodes[future].velocities.z * uniforms.dt);

        
        nodes[future].angles.x = nodes[current].angles.x + (nodes[future].angular_velocities.x * uniforms.dt);
        nodes[future].angles.y = nodes[current].angles.y + (nodes[future].angular_velocities.y * uniforms.dt);
        nodes[future].angles.z = nodes[current].angles.z + (nodes[future].angular_velocities.z * uniforms.dt);
    }
}

@compute
@workgroup_size(64) 
fn save_output(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let future = global_id.x + (c.future_index * uniforms.node_count);
    output_buffer[global_id.x] = nodes[future];
}
