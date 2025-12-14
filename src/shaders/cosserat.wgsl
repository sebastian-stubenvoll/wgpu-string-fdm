struct Node {
    position: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    velocity: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
    
    reference_strain: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
    
    internal_forces: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
    
    resulting_forces: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
}

struct Edge {
// alignment is 16 bytes
    orientation: vec4<f32>, // 16 bytes

     angular_velocity: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
    
    reference_strain: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
    
    internal_moments: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding

    moment_density: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
    // 
    resulting_forces: vec3<f32>, // 12 bytes
    // 4 bytes of implicit padding
    
}

struct Uniforms {
    node_count: u32,
    edge_count: u32,
    dt: f32,
    dx: f32,

    dx_inv: f32, // rest length ^ -1
    mass_inv: f32,
    // implicit padding
    // implicit padding

    stiffness_se: vec3<f32>, // 12 bytes - shearing / extension stiffness matrix
    // 4 bytes of implicit padding
    stiffness_bt: vec3<f32>, // 12 bytes - bending / twisting extension matrix
    // 4 bytes of implicit padding
    inertia: vec3<f32> // 12 bytes, diag(I1, I2, I3)
    // 4 bytes of implicit padding
    inertia_inv: vec3<f32> // 12 bytes, 
    // 4 bytes of implicit padding
    
}

struct PushConstants { 
    current: u32, // either 0 or 1
    future: u32,  // current == 0 ? 1 : 0
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

@compute
@workgroup_size(64) 
fn compute_internals(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    let current = (c.current * uniforms.edge_count) + id;
    let future = (c.future * uniforms.edge_count) + id;

    // This is in the laboratory frame (LF)
    let sigma = ((nodes[current + 1].position - nodes[current].position) * uniforms.dx_inv) - tangent(edges[current].orientation);

    // Compute strain as r_x * Im(q_i-1,conj * q_i)
    let q_inv = vec4<f32>( -1.0 * edges[current].orientation.xyz, edges[current].orientation.w);
    let kappa = 2.0 * uniforms.dx_inv * qmul(q_inv, edges[current + 1]).xyz;
    
    // Internal force is SE stiffness matrix * effective strain (LF)
    nodes[current].internal_forces = uniforms.stiffness_se * (sigma - nodes[current].reference_strain);

    // Internal moments is BT stiffness matrix * effective strain (MF)
    edges[current].internal_moments = uniforms.stiffness_bt * (kappa - edges[current].reference_strain)

    // Compute the moment density for each edges (used for angular momentum calculations later) (MF)
    edges[current].moment_density = cross((kappa - edges[current].reference_strain), edges[current].internal_moments) * uniforms.dx;

}


@compute
@workgroup_size(64) 
fn compute_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    let current = (c.current * uniforms.edge_count) + id;
    let future = (c.future * uniforms.edge_count) + id;

    // The force (density) is the sum of external forces and the spatial derivate of the internal force (LF)
    let f_int_x= 2.0 * uniforms.dx_inv * (nodes[current + 1].position - nodes[current - 1].position) ;
    nodes[current].resulting_forces = f_int_x; // this is where external forces can be added!

    // The angular momentum is calculated entirely in the material frame (MF)
    // The benefit of doing so is that the inertia tensor becomes a constant, diagonalised matrix

    // Compute the spatial derivate for the internal moments.
    // The indexing here works out so that for one timestep at i, this is influnced by edges [i-1, i+1]
    // So while this appears to look like a backwards difference, it is actually central?
    let tau_x = (edges[current].internal_moments - edges[current - 1].internal_moments) * uniforms.dx_inv;
    let m_bar = (edges[current].moment_density + edges[current - 1].moment_density) * 0.5;

    // Assuming inextensibility and since quaterion multiplication / rotation preserves length:
    // tangent vector * li = edge = r_i+1 - r_i = dr/ds
    // cross(q* t q, S sigma) l_i = cross(l_i q* t q, S sigma) = cross (q* li_i t q, S sigma) = cross(q* r q, S sigma)
    // Find the vector representing the current edge (LF)
    let dr = nodes[current + 1].position - nodes[current].position;

    // Calculate the cross product and rotate the result into the material frame (MF).
    let shear_couple = rotate_inv(edges[current].orientation, cross(dr, nodes[current].internal_forces));
    let transport = uniforms.inertia * cross(edges[current].angular_velocity, edges[current].angular_velocity);
    let dilation = vec3<f32>(0.0, 0.0, 0.0);

    
    edges[current].resulting_forces = tau_x + m_bar + shear_couple + transport + dilation; // this is where external forces can be added!


}

@compute
@workgroup_size(64) 
fn integrate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    let current = (c.current * uniforms.edge_count) + id;
    let future = (c.future * uniforms.edge_count) + id;

    let linear_acceleration = nodes[current].resulting_forces * uniforms.mass_inv;
    nodes[future].velocity = nodes[current].velocity + (linear_acceleration * dt);
    nodes[future].position = nodes[current].position + (nodes[future].velocity * dt);
    
    let angular_acceleration = edges[current].resulting_forces * uniforms.inertia_inv;
    edges[future].angular_velocity = edges[current].angular_velocity + (angular_acceleration * uniforms.dt);

    let omega_q = vec4<f32>(edges[future].angular_velocity, 0.0);

    // https://fgiesen.wordpress.com/2012/08/24/quaternion-differentiation/
    // Exponential map approximation, stable for small dt
    let dq = qmul(wq, edges[current].orientation) * uniforms.dt * 0.5;
    
    // This _should_ be a unit quarternion, but we normalize this to account for numerical drift
    edges[future].orientation = fast_normalize(edges[current].orientation + dq);
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
    let rotated = qmul(qmul(q_inv, u), q_);

    return rotated.xyz;
}

fn tangent(q: vec4<f32>) -> vec3<f32> {
    let d3 = vec3<f32>(
        2 * (q.x * q.z + q.w * q.y),
        2 * (q.y * q.z - q.w * w.x),
        1 - 2 * (q.x * q.x + q.y * q.y)
    );

    return d3;
}

fn fast_normalize(q: vec4<f32>) -> vec4<f32> {
    return q * inverseSqrt(dot(q,q));
}
