struct Node {
    position: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    velocity: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
   
    reference_curvature: vec4<f32>, // 12 bytes
    // this is the quaterion encoding the rotation needed to transform
    // the one set of adjacent directors to the other adjacent one at rest
        
    internal_moment: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
}

struct Edge {
// alignment is 16 bytes
    orientation: vec4<f32>, // 16 bytes

    angular_velocity: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    len_inv: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
      
    strain: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
    
    reference_strain: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    internal_force: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
}

struct Uniforms {
    node_count: u32,
    edge_count: u32,
    chunk_size: u32,
    dt: f32,

    dl: f32,
    dl_inv: f32,
    mass_inv: f32,
    // implicit padding

    stiffness_se: vec3<f32>, // 12 bytes - shearing / extension stiffness matrix
    // 4 bytes of implicit padding
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
    _pad1: u32,
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
fn create_reference(@builtin(global_invocation_id) global_id: vec3<u32>) {}

@compute
@workgroup_size(64) 
fn compute_length(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;

    // only run this segment on edges
    if (global_id.x < uniforms.node_count - 1) {
        let r = (nodes[current+1].position - nodes[current].position);
        edges[current].len_inv = inverseSqrt(r.x * r.x + r.y * r.y + r.z * r.z);
    }
}

@compute
@workgroup_size(64) 
fn compute_internals(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let current = (c.current * uniforms.node_count) + global_id.x;
    let future = (c.future * uniforms.node_count) + global_id.x;

    let dilatation_inv = uniforms.dl * edges[current].len_inv;
    
    let d3 = tangent(edges[current].orientation); // (LF)
    let sigma_eff = (((nodes[current + 1].position - nodes[current].position) * uniforms.dl_inv) - d3) - edges[current].reference_strain; //  (LF)
    edges[current].internal_force = rotate(edges[current].orientation, (uniforms.stiffness_se * rotate_inv(edges[current].orientation, sigma_eff))) * dilatation_inv ; // (MF)
    
    let relative_orientation = qmul(qinv(edges[current].orientation), edges[current+1].orientation);
    let effective_orientation = qmul(qinv(nodes[current].reference_curvature), relative_orientation); // (MF)

    nodes[current].internal_moment = uniforms.stiffness_bt * (2.0 * uniforms.dl_inv * effective_orientation.xyz);
    
    
    let kappa; 
    let sigma;
    let dilation;
}


@compute
@workgroup_size(64) 
fn compute_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {}

@compute
@workgroup_size(64) 
fn integrate(@builtin(global_invocation_id) global_id: vec3<u32>) {}

@compute
@workgroup_size(64) 
fn save_output(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = (c.output_index * uniforms.node_count) + global_id.x;
    let current = (c.current * uniforms.node_count) + global_id.x;
    output_nodes[i] = nodes[current];
    output_edges[i] = edges[current];
}

fn qinv(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(q.xyz, q.w);
}

fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = vec4<f32>(v, 0.0);

    let q_inv= vec4<f32>(-q.xyz, q.w); // inverse of a unit quaternion
    let rotated = qmul(qmul(q, u), q_inv);

    return rotated.xyz;
}

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

    return d3;
}

fn fast_normalize(q: vec4<f32>) -> vec4<f32> {
    return q * inverseSqrt(dot(q,q));
}

