struct Node {
    positions: vec4<f32>,
    velocities: vec4<f32>,
}

struct Uniforms {
    output_node: u32,
    node_count: u32,
    chunk_size: u32,
    dt: f32,

    dx: f32,
    j: f32,
    k: f32,
    l: f32,

    m: f32,
    n: f32,
    o: f32,
    p: f32,
}

struct PushConstants { 
    current_index: u32,
    future_index: u32,
    output_index: u32,
    save: u32,
    
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
var<storage, read_write> output_buffer: array<vec2<f32>>;

@compute
@workgroup_size(64) 
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // This shouldn't be touched as it corresponds with the save logic on the rust side.
    // Modify fdm_step instead.

    let idx = global_id.x;
    let current = c.current_index;
    let future = c.future_index;
    

    fdm_step(idx, future, current);
   
    if idx != uniforms.output_node { return; }

    output_buffer[c.output_index] = vec2(nodes[idx].positions[future], nodes[idx].positions[future + 1]);
}


fn fdm_step(index: u32, future: u32, current: u32) {
    let dt = uniforms.dt;
    let dx = uniforms.dx;

    let kAG = uniforms.j;
    let EI = uniforms.k;
    let m = uniforms.m;
    let rhoI = uniforms.l;



    if (index > 1u && index < uniforms.node_count - 2u) {

        // Derivatives
        let w_xx = (nodes[index+1].positions[current] - 2 * nodes[index].positions[current]
                    + nodes[index-1].positions[current]) / (dx * dx);
        let phi_xx = (nodes[index+1].positions[current+1] - 2 * nodes[index].positions[current+1]
                    + nodes[index-1].positions[current+1]) / (dx * dx);

        let w_x = (nodes[index+1].positions[current] - nodes[index-1].positions[current]) / (2 * dx);
        let phi_x = (nodes[index+1].positions[current+1] - nodes[index-1].positions[current+1]) / (2 * dx);

        let phi = nodes[index].positions[current + 1];

        // Accelerations
        let w_tt = (kAG * (w_xx - phi_x)) / m;
        let phi_tt = (EI * phi_xx + kAG * (w_x - phi)) / rhoI;
    
        // Update velocities
        nodes[index].velocities[future] = nodes[index].velocities[current] + (w_tt * dt);
        nodes[index].velocities[future+1] = nodes[index].velocities[current+1] + (phi_tt * dt);

        // Update displacements
        nodes[index].positions[future] = nodes[index].positions[current] + (nodes[index].velocities[future] * dt);
        nodes[index].positions[future+1] = nodes[index].positions[current+1] + (nodes[index].velocities[future+1] * dt);

    } else {
        nodes[index].positions[future] = 0.0;
        nodes[index].positions[future + 1] = 0.0;
        nodes[index].velocities[future] = 0.0;
        nodes[index].velocities[future + 1] = 0.0;
    }
}

