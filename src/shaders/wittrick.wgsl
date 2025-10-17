// alignOf(vec3<f32>) is 16 but sizeOf(vec3<f32>) is only 12
// this causes implicit padding
// this could be packed more efficiently (TODO)
struct Node {
    positions: vec3<f32>,
    // implicit padding 
    velocities: vec3<f32>,
    // implicit padding 
    angles: vec3<f32>,
    // implicit padding 
    angular_velocities: vec3<f32>,
    // implicit padding 
    moments: vec3<f32>,
    // implicit padding 
    forces: vec3<f32>,
    // implicit padding 
    
}

// this could be packed more efficiently (TODO)
struct Uniforms {
    dt: f32, 
    ds: f32,
    node_count: u32,
    chunk_size: u32,
    beta: vec3<f32>,
    // implicit padding 
    sigma: vec3<f32>,
    // implicit padding 
    k: vec3<f32>,
    tau: f32,
    kappa: f32,
    m_coil: f32,
    c2_core: f32,
    loss: f32,
}

struct PushConstants { 
    current_index: u32,
    future_index: u32,
    output_index: u32,
    // implicit padding 
    
}

var <push_constant> c: PushConstants;

@group(0)
@binding(0)
var <uniform> uniforms: Uniforms;

@group(0)
@binding(1)
var<storage, read_write> nodes: array<vec2<Node>>;

@group(0)
@binding(2)
var<storage, read_write> output_buffer: array<Node>;

@compute
@workgroup_size(64) 
fn update_moments(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;

    if id > 0 && id < uniforms.node_count - 2u {
        let current = c.current_index;
        let future = c.future_index;

        let c_ptr = &nodes[id][current];

        // Finite difference approximation of dtheta/ds
        let theta_s = (nodes[id+1][current].angles - nodes[id-1][current].angles) / uniforms.ds;

        // Finite difference approximation of r/ds
        let uvw_s = (nodes[id+1][current].positions - nodes[id-1][current].positions) / uniforms.ds;

        // Calculate new moments for bending/torque
        nodes[id][future].moments = (theta_s + vec3(
            uniforms.tau * (*c_ptr).angles[1],
            - uniforms.tau * (*c_ptr).angles[0] +  uniforms.kappa + (*c_ptr).angles[2],
            - uniforms.kappa * (*c_ptr).angles[1]
        )) / uniforms.beta;
 
        // Calculate new forces for shearing/tension
        nodes[id][future].forces = (uvw_s + vec3(
            - uniforms.tau * (*c_ptr).positions[1] + uniforms.kappa * (*c_ptr).positions[2] - (*c_ptr).angles[0],
            uniforms.tau * (*c_ptr).positions[0] + (*c_ptr).angles[1],
            uniforms.kappa * (*c_ptr).positions[0]
        )) / uniforms.sigma;


        // Since we've already computed dr/ds here, it makes sense to update the core contribution now
        nodes[id][future].velocities = nodes[id][current].velocities + uniforms.c2_core * uvw_s * uniforms.dt;
    }
}


@compute
@workgroup_size(64) 
fn update_positions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;

    if id > 0 && id < node_count - 2 {
        let current = c.current_index;
        let future = c.future_index;

        // Finite difference approximation of dtheta/ds
        let M_s = (nodes[id+1][future].moments - nodes[id-1][future].moments) / uniforms.ds;

        // Finite difference approximation of r/ds
        let QP_s = (nodes[id+1][future].forces - nodes[id-1][future].forces) / uniforms.ds;

        // Calculate new moment angular accelerations and integrate them
        nodes[id][future].angular_velocities = nodes[id][current].angular_velocities + uniforms.dt * (M_s + vec3(
            nodes[id][future].forces[0] + uniforms.tau + nodes[id][future].moments[1],
            - nodes[id][future].forces[1] - uniforms.tau * nodes[id][future].moments[0] + uniforms.kappa * nodes[id][future].moments[2]
            - uniforms.kappa * nodes[id][future].moments[1]
        )) / ( uniforms.m_coil * uniforms.k);
 
        // Calculate new accelerations and integrate them
        // Here we must update the future velocity, as that is the most recent one since it already contains
        // the contribution from the core computed earlier.
        nodes[id][future].velocities = nodes[id][future].velocities + uniforms.dt * (QP_s + vec3(
            - uniforms.tau * nodes[id][future].forces[1] + uniforms.kappa * nodes[id][future].forces[2],
            uniforms.tau * nodes[id][future].forces[0],
            - uniforms.kappa * nodes[id][future].forces[0]
        )) / uniforms.m_coil;

        // Finally integrate the velocities and update the positions
        nodes[id][future].angles = nodes[id][current].angles + nodes[id][future].angular_velocities * uniforms.dt;
        nodes[id][future].positions = nodes[id][current].positions + nodes[id][future].velocities * uniforms.dt;
    }
}

@compute
@workgroup_size(64) 
fn save_ouput(@builtin(global_invocation_id) global_id: vec3<u32>) {
    output_buffer[global_id] = nodes[global_id][c.future_index];
}
