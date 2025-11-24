// alignOf(vec3<f32>) is 16 but sizeOf(vec3<f32>) is only 12
// this causes implicit padding
// this could be packed more efficiently (TODO)
struct Node {
    positions: vec3<f32>,
    _pad0: f32,
    velocities: vec3<f32>,
    _pad1: f32,
    angles: vec3<f32>,
    _pad2: f32,
    angular_velocities: vec3<f32>,
    _pad3: f32,
    internal_moments: vec3<f32>,
    _pad4: f32,
    internal_forces: vec3<f32>,
    _pad5: f32,
    helix_forces: vec3<f32>,
    _pad6: f32,
    core_forces: vec3<f32>,
    _pad7: f32,
}

// this could be packed more efficiently (TODO)
struct Uniforms {
    dt: f32, 
    two_ds_inv: f32,
    node_count: u32,
    loss: f32,

    tau: f32,
    kappa: f32,
    m_inv: f32,
    dxf_t: f32,

    beta: vec3<f32>,
    dx2_inv: f32,

    sigma: vec3<f32>,
    chunk_size: u32,

    muk2_inv: vec3<f32>,
    pad: u32,
}

struct PushConstants { 
    current_index: u32,
    future_index: u32,
    external_force: f32,
    output_idx: u32,
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
    if id == 200 {
        let current = id + (c.current_index * uniforms.node_count);

        nodes[current].velocities.x += c.external_force * uniforms.m_inv * uniforms.dt;

    }
}

@compute
@workgroup_size(64) 
fn calculate_internal_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if id > 0 && id < uniforms.node_count - 1 {
        let current = id + (c.current_index * uniforms.node_count);
        let future = id + (c.future_index * uniforms.node_count);

        let theta1_s = ((nodes[current+1].angles.x) - (nodes[current-1].angles.x)) * uniforms.two_ds_inv;
        let theta2_s = ((nodes[current+1].angles.y) - (nodes[current-1].angles.y)) * uniforms.two_ds_inv;
        let theta3_s = ((nodes[current+1].angles.z) - (nodes[current-1].angles.z)) * uniforms.two_ds_inv;

        let u_s = ((nodes[current+1].positions.x) - (nodes[current-1].positions.x)) * uniforms.two_ds_inv;
        let v_s = ((nodes[current+1].positions.y) - (nodes[current-1].positions.y)) * uniforms.two_ds_inv;
        let w_s = ((nodes[current+1].positions.z) - (nodes[current-1].positions.z)) * uniforms.two_ds_inv;

        let m1 = (theta1_s + (uniforms.tau * nodes[current].angles.y)) * uniforms.beta.x;
        let m2 = (theta2_s - (uniforms.tau * nodes[current].angles.x) + (uniforms.kappa * nodes[current].angles.z)) * uniforms.beta.y;
        let m3 = (theta3_s - (uniforms.kappa * nodes[current].angles.y)) * uniforms.beta.z;

        let q1 = (u_s - (uniforms.tau * nodes[current].positions.y) + (uniforms.kappa * nodes[current].positions.z) - nodes[current].angles.x) * uniforms.sigma.x;
        let q2 = (v_s + (uniforms.tau * nodes[current].positions.x) + nodes[current].angles.y) * uniforms.sigma.y;
        let p = (w_s - (uniforms.kappa * nodes[current].positions.x)) * uniforms.sigma.z;

        nodes[future].internal_moments.x = m1;
        nodes[future].internal_moments.y = m2;
        nodes[future].internal_moments.z = m3;

        nodes[future].internal_forces.x = q1;
        nodes[future].internal_forces.y = q2;
        nodes[future].internal_forces.z = p;
    }
}

@compute
@workgroup_size(64) 
fn calculate_displacements(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if id > 0 && id < uniforms.node_count - 1 {

        let current = id + (c.current_index * uniforms.node_count);
        let future = id + (c.future_index * uniforms.node_count);

        
        let m1_s = ((nodes[future+1].internal_moments.x) - (nodes[future-1].internal_moments.x)) * uniforms.two_ds_inv;
        let m2_s = ((nodes[future+1].internal_moments.y) - (nodes[future-1].internal_moments.y)) * uniforms.two_ds_inv;
        let m3_s = ((nodes[future+1].internal_moments.z) - (nodes[future-1].internal_moments.z)) * uniforms.two_ds_inv;

        let q1_s = ((nodes[future+1].internal_forces.x) - (nodes[future-1].internal_forces.x)) * uniforms.two_ds_inv;
        let q2_s = ((nodes[future+1].internal_forces.y) - (nodes[future-1].internal_forces.y)) * uniforms.two_ds_inv;
        let p_s = ((nodes[future+1].internal_forces.z) - (nodes[future-1].internal_forces.z)) * uniforms.two_ds_inv;

        let theta1_tt = (m1_s + nodes[future].internal_forces.x + (uniforms.tau * nodes[future].internal_moments.y)) * (uniforms.muk2_inv.x);
        let theta2_tt = (m2_s - nodes[future].internal_forces.y - (uniforms.tau * nodes[future].internal_moments.x) + (uniforms.kappa * nodes[future].internal_moments.z)) * (uniforms.muk2_inv.y);
        let theta3_tt = (m3_s - (uniforms.kappa * nodes[future].internal_moments.y)) * (uniforms.muk2_inv.z);

        let helix_force_x = (q1_s - (uniforms.tau * nodes[future].internal_forces.y) + (uniforms.kappa * nodes[future].internal_forces.z));
        let helix_force_y = (q2_s + (uniforms.tau * nodes[future].internal_forces.x));
        let helix_force_z = (p_s - (uniforms.kappa * nodes[future].internal_forces.x));

        nodes[future].helix_forces.x = helix_force_x;
        nodes[future].helix_forces.y = helix_force_y;
        nodes[future].helix_forces.z = helix_force_z;

        let u_xx = (nodes[current + 1].positions.x - 2.0 * nodes[current].positions.x + nodes[current - 1].positions.x) * (uniforms.dx2_inv);
        let v_xx = (nodes[current + 1].positions.y - 2.0 * nodes[current].positions.y + nodes[current - 1].positions.y) * (uniforms.dx2_inv);
        let w_xx = (nodes[current + 1].positions.z - 2.0 * nodes[current].positions.z + nodes[current - 1].positions.z) * (uniforms.dx2_inv);

        let core_forces_x = u_xx * uniforms.dxf_t;
        let core_forces_y = v_xx * uniforms.dxf_t;
        let core_forces_z = w_xx * uniforms.dxf_t;

        nodes[future].core_forces.x = core_forces_x;
        nodes[future].core_forces.y = core_forces_y;
        nodes[future].core_forces.z = core_forces_z;

        let u_tt = (core_forces_x + helix_force_x) * uniforms.m_inv;
        let v_tt = (core_forces_y + helix_force_y) * uniforms.m_inv;
        let w_tt = (core_forces_z + helix_force_z) * uniforms.m_inv;

        nodes[future].velocities.x = (nodes[current].velocities.x + (u_tt * uniforms.dt)) * uniforms.loss;
        nodes[future].velocities.y = (nodes[current].velocities.y + (v_tt * uniforms.dt)) * uniforms.loss;
        nodes[future].velocities.z = (nodes[current].velocities.z + (w_tt * uniforms.dt)) * uniforms.loss;

        nodes[future].angular_velocities.x = (nodes[current].angular_velocities.x + (theta1_tt * uniforms.dt)) * uniforms.loss;
        nodes[future].angular_velocities.y = (nodes[current].angular_velocities.y + (theta2_tt * uniforms.dt)) * uniforms.loss;
        nodes[future].angular_velocities.z = (nodes[current].angular_velocities.z + (theta3_tt * uniforms.dt)) * uniforms.loss;

        nodes[future].positions.x = nodes[current].positions.x + (nodes[future].velocities.x * uniforms.dt);
        nodes[future].positions.y = nodes[current].positions.y + (nodes[future].velocities.y * uniforms.dt);
        nodes[future].positions.z = nodes[current].positions.z + (nodes[future].velocities.z * uniforms.dt);

        nodes[future].angles.x = (nodes[current].angles.x + (nodes[future].angular_velocities.x * uniforms.dt));
        nodes[future].angles.y = (nodes[current].angles.y + (nodes[future].angular_velocities.y * uniforms.dt));
        nodes[future].angles.z = (nodes[current].angles.z + (nodes[future].angular_velocities.z * uniforms.dt));
    } else {

        let neighbor = min(id+1u, uniforms.node_count - 2u);
        let current = neighbor + (c.current_index * uniforms.node_count);
        let future =  neighbor + (c.future_index * uniforms.node_count);


        nodes[future].positions.x = 0.0;
        nodes[future].positions.y = 0.0;
        nodes[future].positions.z = 0.0;

        nodes[future].angles.x = nodes[current].angles.x;
        nodes[future].angles.y = nodes[current].angles.y;
        nodes[future].angles.z = nodes[current].angles.z;
    }

}

@compute
@workgroup_size(64) 
fn save_output(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = global_id.x + (c.current_index * uniforms.node_count);
    let offset = c.output_idx * uniforms.node_count;
    output_buffer[global_id.x + offset] = nodes[current];
}
