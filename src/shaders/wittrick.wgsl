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
    _pad1: u32,
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
fn setup(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if id == 10 {
        // TODO: make physically accurate
        let vel = uniforms.dt * 6.5f;
        let current = global_id.x + (c.current_index * uniforms.node_count);
        // Add to normal vector velocity
        nodes[current].velocities[1] += vel;
        
    }
}

@compute
@workgroup_size(64) 
fn calculate_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;

    if id > 0u && id < uniforms.node_count - 2u {
        let current = id + (c.current_index * uniforms.node_count);
        let future = id + (c.future_index * uniforms.node_count);

        // Finite difference approximation of dtheta/ds
        let theta_s = (nodes[current+1].angles - nodes[current-1].angles) / (2 * uniforms.ds);

        // Finite difference approximation of r/ds
        let uvw_s = (nodes[current+1].positions - nodes[current-1].positions) / (uniforms.ds * 2);

        // Calculate new moments for bending/torque
        nodes[future].moments = (theta_s + vec3(
            uniforms.tau * nodes[current].angles[1],
            -1 * uniforms.tau * nodes[current].angles[0] +  uniforms.kappa * nodes[current].angles[2],
            -1 * uniforms.kappa * nodes[current].angles[1]
        )) / uniforms.beta;
 
        // Calculate new forces for shearing/tension
        nodes[future].forces = (uvw_s + vec3(
            - uniforms.tau * nodes[current].positions[1] + uniforms.kappa * nodes[current].positions[2] - nodes[current].angles[0],
            uniforms.tau * nodes[current].positions[0] + nodes[current].angles[1],
            uniforms.kappa * nodes[current].positions[0]
        )) / uniforms.sigma;
    }
}


@compute
@workgroup_size(64) 
fn apply_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;

    if id > 0 && id < uniforms.node_count - 2 {
        let current = id + (c.current_index * uniforms.node_count);
        let future = id + (c.future_index * uniforms.node_count);

       // Finite difference approximation of dtheta/ds
        let M_s = (nodes[future+1].moments - nodes[future-1].moments) / (2 * uniforms.ds);

        // Finite difference approximation of r/ds
        let QP_s = (nodes[future+1].forces - nodes[future-1].forces) / (2 * uniforms.ds);

        // Calculate new moment angular accelerations and integrate them
        nodes[future].angular_velocities = nodes[current].angular_velocities + uniforms.dt * (M_s + vec3(
         nodes[future].forces[0] + uniforms.tau * nodes[future].moments[1],
         -1 * nodes[future].forces[1] - uniforms.tau * nodes[future].moments[0] + uniforms.kappa * nodes[future].moments[2],
         -1 * uniforms.kappa * nodes[future].moments[1]
        )) / ( uniforms.m_coil * uniforms.k * uniforms.k);
 
        // Calculate new accelerations and integrate them
        nodes[future].velocities = nodes[current].velocities + uniforms.dt * (QP_s + vec3(
            -1 *  uniforms.tau * nodes[future].forces[1] + uniforms.kappa * nodes[future].forces[2],
            uniforms.tau * nodes[future].forces[0],
            -1 * uniforms.kappa * nodes[future].forces[0]
        )) / uniforms.m_coil;

        // Finally integrate the velocities and update the positions
        nodes[future].angles = nodes[current].angles + nodes[future].angular_velocities * uniforms.dt * uniforms.loss;
        nodes[future].positions = nodes[current].positions + nodes[future].velocities * uniforms.dt * uniforms.loss;
    }
}

@compute
@workgroup_size(64) 
fn save_output(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let future = global_id.x + (c.future_index * uniforms.node_count);
    output_buffer[global_id.x] = nodes[future];
}
