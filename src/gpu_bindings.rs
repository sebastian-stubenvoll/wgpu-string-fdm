#![allow(dead_code)]
use std::{error::Error, fmt::Display, io::Write, iter};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
// Don't derive Default on purpose! Implicit initialization via Default::default() can be a footgun here!
pub struct Node {
    position: [f32; 3],
    _pad0: u32,
    velocity: [f32; 3],
    _pad1: u32,
    curvature: [f32; 3],
    bend_stretch: f32,
    reference_curvature: [f32; 4],
    internal_moment: [f32; 3],
    _pad3: u32,
    // q0_i^-1 * q0_{i+1}, see the shader
    rest_relative_orientation: [f32; 4],
    // low part of the displacement (position + position_lo), see the shader's drift_node
    position_lo: [f32; 3],
    _pad4: u32,
}

// The structs are uploaded with bytemuck, so the repr(C) layout must equal the WGSL storage
// layout (vec3/vec4/mat3x3 aligned to 16 bytes, mat3x3 columns padded to 16 bytes).
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(offset_of!(Node, position) == 0);
    assert!(offset_of!(Node, velocity) == 16);
    assert!(offset_of!(Node, curvature) == 32);
    assert!(offset_of!(Node, bend_stretch) == 44);
    assert!(offset_of!(Node, reference_curvature) == 48);
    assert!(offset_of!(Node, internal_moment) == 64);
    assert!(offset_of!(Node, rest_relative_orientation) == 80);
    assert!(offset_of!(Node, position_lo) == 96);
    assert!(size_of::<Node>() == 112);
};

impl Node {
    pub fn new(
        position: &[f32; 3],
        velocity: &[f32; 3],
        reference_curvature: &[f32; 4],
        rest_relative_orientation: &[f32; 4],
    ) -> Self {
        Self {
            position: *position,
            _pad0: 0,
            velocity: *velocity,
            _pad1: 0,
            curvature: [0.0; 3],
            bend_stretch: 0.0,
            reference_curvature: *reference_curvature,
            internal_moment: [0.0; 3],
            _pad3: 0,
            rest_relative_orientation: *rest_relative_orientation,
            position_lo: [0.0; 3],
            _pad4: 0,
        }
    }

    pub fn to_raw(self) -> [[f32; 3]; 4] {
        [
            self.position,
            self.velocity,
            self.internal_moment,
            self.curvature,
        ]
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
// Don't derive Default on purpose! Implicit initialization via Default::default() can be a footgun here!
pub struct Edge {
    orientation: [f32; 4],
    angular_velocity: [f32; 3],
    len_inv: f32,
    strain: [f32; 3],
    _pad0: u32,
    reference_strain: [f32; 3],
    _pad1: u32,
    internal_force: [f32; 3],
    _pad2: u32,
    reference_vector: [f32; 3],
    _pad3: u32,
    // R0^T * reference_vector - dl * e3, see the shader
    reference_offset: [f32; 3],
    _pad4: u32,
    // mat3x3<f32>: columns d1, d2, d3 of the rest frame, each padded to 16 bytes
    rest_frame: [[f32; 4]; 3],
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(offset_of!(Edge, orientation) == 0);
    assert!(offset_of!(Edge, angular_velocity) == 16);
    assert!(offset_of!(Edge, len_inv) == 28);
    assert!(offset_of!(Edge, strain) == 32);
    assert!(offset_of!(Edge, reference_strain) == 48);
    assert!(offset_of!(Edge, internal_force) == 64);
    assert!(offset_of!(Edge, reference_vector) == 80);
    assert!(offset_of!(Edge, reference_offset) == 96);
    assert!(offset_of!(Edge, rest_frame) == 112);
    assert!(size_of::<Edge>() == 160);
};

impl Edge {
    pub fn new(
        orientation: &[f32; 4],
        reference_vector: &[f32; 3],
        angular_velocity: &[f32; 3],
        reference_strain: &[f32; 3],
        reference_offset: &[f32; 3],
        rest_frame: &[[f32; 3]; 3],
    ) -> Self {
        let column = |c: &[f32; 3]| [c[0], c[1], c[2], 0.0];
        Self {
            orientation: *orientation,
            angular_velocity: *angular_velocity,
            len_inv: 0.0,
            strain: [0.0; 3],
            _pad0: 0,
            reference_strain: *reference_strain,
            _pad1: 0,
            internal_force: [0.0; 3],
            _pad2: 0,
            reference_vector: *reference_vector,
            _pad3: 0,
            reference_offset: *reference_offset,
            _pad4: 0,
            rest_frame: rest_frame.each_ref().map(column),
        }
    }

    pub fn to_raw(self) -> ([f32; 4], [[f32; 3]; 3]) {
        (
            self.orientation,
            [self.angular_velocity, self.internal_force, self.strain],
        )
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
// Don't derive Default on purpose! Implicit initialization via Default::default() can be a footgun here!
pub struct FDMUniform {
    node_count: u32,
    edge_count: u32,
    chunk_size: u32,
    mass_inv: f32,
    dt: f32,
    dt_inv: f32,
    dl: f32,
    dl_inv: f32,
    stiffness_se: [f32; 3],
    clamp_offset: u32,
    stiffness_bt: [f32; 3],
    _pad1: u32,
    inertia: [f32; 3],
    _pad2: u32,
    inertia_inv: [f32; 3],
    _pad3: u32,
    // per material axis (tilt, tilt, torsion), see the shader
    angular_damping: [f32; 3],
    linear_damping: f32,
    // 0 legacy, 1 variational (see the shader)
    force_law: u32,
    // Kelvin-Voigt twist-rate damping D3 (N m^2 s), see the shader; 0 disables it
    twist_rate_damping: f32,
    // always 0; the shader hides values from its compiler's algebraic rewrites with it (see opaque)
    opaque_zero: u32,
    _pad4: u32,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(offset_of!(FDMUniform, stiffness_se) == 32);
    assert!(offset_of!(FDMUniform, stiffness_bt) == 48);
    assert!(offset_of!(FDMUniform, inertia) == 64);
    assert!(offset_of!(FDMUniform, inertia_inv) == 80);
    assert!(offset_of!(FDMUniform, angular_damping) == 96);
    assert!(offset_of!(FDMUniform, linear_damping) == 108);
    assert!(offset_of!(FDMUniform, force_law) == 112);
    assert!(offset_of!(FDMUniform, twist_rate_damping) == 116);
    assert!(offset_of!(FDMUniform, opaque_zero) == 120);
    assert!(size_of::<FDMUniform>() == 128);
};

impl FDMUniform {
    pub fn new(
        nodes: &[Node],
        edges: &[Edge],
        chunk_size: u32,
        dt: f32,
        dl: f32,
        mass: f32,
        stiffness_se: &[f32; 3],
        stiffness_bt: &[f32; 3],
        intertia: &[f32; 3],
        clamp_offset: u32,
        linear_damping: f32,
        angular_damping: &[f32; 3],
        force_law: u32,
    ) -> Self {
        assert!(!edges.is_empty());
        assert_eq!(
            nodes.len(),
            edges.len(),
            "To simplify indexing a ghost edge must be provided!"
        );
        assert!(chunk_size > 0);
        assert!(dt > 0.0);
        assert!(dl > 0.0);
        assert!(mass > 0.0);
        assert!(stiffness_se.iter().any(|s| *s > 0.0));
        assert!(stiffness_bt.iter().any(|s| *s > 0.0));
        assert!(intertia.iter().any(|s| *s > 0.0));

        Self {
            node_count: nodes.len() as u32,
            edge_count: edges.len() as u32,
            chunk_size,
            mass_inv: 1.0 / mass,
            dt,
            dt_inv: 1.0 / dt,
            dl,
            dl_inv: 1.0 / dl,
            stiffness_se: *stiffness_se,
            clamp_offset,
            stiffness_bt: *stiffness_bt,
            _pad1: 0,
            inertia: *intertia,
            _pad2: 0,
            inertia_inv: intertia.map(|i| 1.0 / i),
            _pad3: 0,
            angular_damping: *angular_damping,
            linear_damping,
            force_law,
            twist_rate_damping: 0.0,
            opaque_zero: 0,
            _pad4: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    current_index: u32,
    future_index: u32,
    output_index: u32,
    hammering: u32,

    write_output: u32,
    _reserved1: f32,
    hammer_offset_y: f32,
    steps: u32, // solver steps per output frame, read by advance_frame
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
// Don't derive Default on purpose! Implicit initialization via Default::default() can be a footgun here!
pub struct Hammer {
    displacement: f32,
    velocity: f32,
    mass_inv: f32,
    stiffness: f32,

    exponent: f32,
    hysteresis_factor: f32,
    relaxation_time: f32,
    force_accumulator: i32,
}

impl Hammer {
    pub fn new(
        displacement: f32,
        velocity: f32,
        mass: f32,
        stiffness: f32,
        exponent: f32,
        hysteresis_factor: f32,
        relaxation_time: f32,
    ) -> Self {
        assert!(mass > 0.0);
        assert!(stiffness > 0.0);
        assert!(exponent > 0.0);
        assert!(hysteresis_factor > 0.0);
        assert!(relaxation_time > 0.0);

        Self {
            displacement,
            velocity,
            mass_inv: mass.recip(),
            stiffness,

            exponent,
            hysteresis_factor,
            relaxation_time,
            force_accumulator: 0i32,
        }
    }
}

pub const MAX_BOW_CONTACTS: usize = 32;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BowContact {
    pub node: u32,
    pub weight: f32,
    pub sticking: u32,
    pub slip_velocity: f32,
    pub force: f32,
    pub impulse: f32,
    pub bow_work: f32,
    pub friction_work: f32,
    pub normal_work: f32,
    pub temperature: f32,
    pub heat_power: f32,
    _pad1: f32,
}

pub const BOW_SOE_TERMS: usize = 32;
const BOW_SOE_FLOATS: usize = MAX_BOW_CONTACTS * 2 * BOW_SOE_TERMS;
const BOW_RING: usize = 65536;
const BOW_RING_FLOATS: usize = MAX_BOW_CONTACTS * BOW_RING;

/// Thermal friction model parameters (WGSL BowThermal; batches/bowing.thermal_contact).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BowThermal {
    pub c_over_h: f32,
    pub conv_coef: f32,
    pub lam_string: f32,
    pub lam_bow: f32,
    pub window: u32,
    pub f_window: f32,
    pub terms: u32,
    _pad0: u32,
    pub cal_k1: f32,
    pub cal_k2: f32,
    pub cal_force: f32,
    _pad1: f32,
    pub exact: [f32; 32],
    pub decay: [f32; BOW_SOE_TERMS],
    pub weight: [f32; BOW_SOE_TERMS],
    pub weight_window: [f32; BOW_SOE_TERMS],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BowFrame {
    pub normal_force: f32,
    pub velocity: f32,
    pub step: u32,
    _pad: u32,
    pub contacts: [BowContact; MAX_BOW_CONTACTS],
}

/// Header and contact list of the WGSL `Bow` storage struct; the output frames follow it.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Bow {
    model: u32,
    torsion: u32,
    contact_count: u32,
    step: u32,
    radius: f32,
    normal_force: f32,
    velocity: f32,
    mu_d: f32,
    mu_a1: f32,
    mu_v1: f32,
    mu_a2: f32,
    mu_v2: f32,
    start_step: u32,
    ramp_steps: u32,
    stop_step: u32,
    _pad: u32,
    contacts: [BowContact; MAX_BOW_CONTACTS],
    thermal: BowThermal,
}

/// Byte offset of the output frames in the bow buffer: header, contacts and thermal parameters
/// (struct Bow above), then the sum-of-exponentials states and the temperature rings.
const BOW_FRAMES_OFFSET: u64 =
    (std::mem::size_of::<Bow>() + 4 * (BOW_SOE_FLOATS + BOW_RING_FLOATS)) as u64;

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<BowContact>() == 48);
    assert!(offset_of!(BowFrame, contacts) == 16);
    assert!(size_of::<BowFrame>() == 1552);
    assert!(offset_of!(Bow, radius) == 16);
    assert!(offset_of!(Bow, start_step) == 48);
    assert!(offset_of!(Bow, contacts) == 64);
    assert!(offset_of!(Bow, thermal) == 1600);
    assert!(offset_of!(BowThermal, exact) == 48);
    assert!(size_of::<BowThermal>() == 560);
    assert!(size_of::<Bow>() == 2160);
};

pub const MAX_HAMMER_CONTACTS: usize = 32;
const MAX_SURFACE_NODES: usize = 1024;

/// Contact record of the pick or of one hammer friction node (WGSL SurfaceContact).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SurfaceContact {
    pub node: u32,
    pub sticking: u32,
    pub in_contact: u32,
    pub slip_velocity: f32,
    pub normal_force: f32,
    pub friction_force: f32,
    pub normal_y: f32,
    pub normal_z: f32,
    pub body_work: f32,
    pub friction_work: f32,
    pub normal_work: f32,
    pub impulse_t: f32,
}

/// Rigid plectrum (WGSL Pick).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Pick {
    pub enabled: u32,
    pub torsion: u32,
    pub node: u32,
    pub step: u32,
    pub radius: f32,
    pub tip_radius: f32,
    pub length: f32,
    pub mu_s: f32,
    pub mu_d: f32,
    pub axis_y: f32,
    pub axis_z: f32,
    pub tip_y0: f32,
    pub tip_z: f32,
    pub velocity: f32,
    pub start_step: u32,
    pub ramp_steps: u32,
    pub first_contact_step: u32,
    pub last_contact_step: u32,
    pub max_penetration: f32,
    pub tip_y: f32,
    pub contact: SurfaceContact,
}

/// Hammer contact geometry and felt friction (WGSL HammerFriction).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HammerFriction {
    pub enabled: u32,
    pub torsion: u32,
    pub count: u32,
    pub curved: u32,
    pub radius: f32,
    pub crown_radius: f32,
    pub mu_s: f32,
    pub mu_d: f32,
    pub hammer_work: f32,
    pub hammer_displacement: f32,
    pub hammer_velocity: f32,
    _pad: f32,
    pub contacts: [SurfaceContact; MAX_HAMMER_CONTACTS],
}

/// One output frame of the surface buffer: pick and hammer records.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SurfaceFrame {
    pub pick: Pick,
    pub hammer: HammerFriction,
}

/// Byte offset of the output frames in the surface buffer: pick, hammer friction, then the per-node
/// felt normals (vec4 each).
const SURFACE_FRAMES_OFFSET: u64 = (std::mem::size_of::<Pick>()
    + std::mem::size_of::<HammerFriction>()
    + 16 * MAX_SURFACE_NODES) as u64;

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<SurfaceContact>() == 48);
    assert!(offset_of!(Pick, radius) == 16);
    assert!(offset_of!(Pick, first_contact_step) == 64);
    assert!(offset_of!(Pick, contact) == 80);
    assert!(size_of::<Pick>() == 128);
    assert!(offset_of!(HammerFriction, hammer_work) == 32);
    assert!(offset_of!(HammerFriction, contacts) == 48);
    assert!(size_of::<HammerFriction>() == 1584);
    assert!(size_of::<SurfaceFrame>() == 1712);
    // the vec4 array must start 16-byte aligned
    assert!((size_of::<Pick>() + size_of::<HammerFriction>()) % 16 == 0);
};

impl Pick {
    pub fn off() -> Self {
        bytemuck::Zeroable::zeroed()
    }

    /// Pick at node `node`: capsule of half-thickness tip_radius along the unit axis (y, z) from the
    /// tip centre (tip_y0, tip_z), face length `length`; moves along +y, starting at start_step with
    /// a linear speed ramp over ramp_steps to `velocity`. Coulomb friction mu_s / mu_d.
    pub fn new(
        node: u32,
        torsion: bool,
        radius: f32,
        tip_radius: f32,
        length: f32,
        axis: [f32; 2],
        tip: [f32; 2],
        velocity: f32,
        start_step: u32,
        ramp_steps: u32,
        mu_s: f32,
        mu_d: f32,
    ) -> Self {
        assert!(radius > 0.0 && tip_radius > 0.0 && length > 0.0);
        assert!((axis[0] * axis[0] + axis[1] * axis[1] - 1.0).abs() < 1e-5, "pick axis must be a unit vector");
        assert!(mu_s >= mu_d && mu_d >= 0.0, "need mu_s >= mu_d >= 0");
        let mut p = Self::off();
        p.enabled = 1;
        p.torsion = torsion as u32;
        p.node = node;
        p.radius = radius;
        p.tip_radius = tip_radius;
        p.length = length;
        p.mu_s = mu_s;
        p.mu_d = mu_d;
        p.axis_y = axis[0];
        p.axis_z = axis[1];
        p.tip_y0 = tip[0];
        p.tip_z = tip[1];
        p.tip_y = tip[0];
        p.velocity = velocity;
        p.start_step = start_step;
        p.ramp_steps = ramp_steps;
        p.contact.node = node;
        p
    }
}

impl HammerFriction {
    pub fn off() -> Self {
        bytemuck::Zeroable::zeroed()
    }

    /// crown_radius > 0: curved crown (contact normal from the geometry), 0: flat. mu_s, mu_d: felt
    /// friction on the given contact nodes (mu_s = 0 and no nodes: no friction).
    pub fn new(
        nodes: &[u32],
        torsion: bool,
        radius: f32,
        crown_radius: f32,
        mu_s: f32,
        mu_d: f32,
    ) -> Self {
        assert!(nodes.len() <= MAX_HAMMER_CONTACTS);
        assert!(crown_radius >= 0.0 && radius > 0.0);
        assert!(mu_s >= mu_d && mu_d >= 0.0, "need mu_s >= mu_d >= 0");
        let mut h = Self::off();
        h.enabled = (!nodes.is_empty() && mu_s > 0.0) as u32;
        h.torsion = torsion as u32;
        h.count = if h.enabled == 1 { nodes.len() as u32 } else { 0 };
        h.curved = (crown_radius > 0.0) as u32;
        h.radius = radius;
        h.crown_radius = crown_radius;
        h.mu_s = mu_s;
        h.mu_d = mu_d;
        for (c, &n) in h.contacts.iter_mut().zip(nodes) {
            c.node = n;
        }
        h
    }
}

impl Bow {
    /// Bow off: the frame kernel skips every bow phase.
    pub fn off() -> Self {
        bytemuck::Zeroable::zeroed()
    }

    /// model 1: friction curve mu(v) = mu[0] + mu[1] exp(-v / mu[2]) + mu[3] exp(-v / mu[4]).
    /// Profile (in solver steps): linear rise over ramp_steps from start_step, steady, linear
    /// fall over ramp_steps from stop_step.
    pub fn new(
        model: u32,
        torsion: bool,
        nodes: &[u32],
        weights: &[f32],
        radius: f32,
        normal_force: f32,
        velocity: f32,
        mu: &[f32; 5],
        start_step: u32,
        ramp_steps: u32,
        stop_step: u32,
    ) -> Self {
        assert!(model == 1 || model == 2, "bow model {model} not implemented");
        assert!(!nodes.is_empty() && nodes.len() <= MAX_BOW_CONTACTS);
        assert_eq!(nodes.len(), weights.len());
        assert!(radius > 0.0 && normal_force >= 0.0);
        assert!(mu[2] > 0.0 && mu[4] > 0.0);
        let mut contacts: [BowContact; MAX_BOW_CONTACTS] = bytemuck::Zeroable::zeroed();
        for (c, (&n, &w)) in contacts.iter_mut().zip(nodes.iter().zip(weights)) {
            c.node = n;
            c.weight = w;
            c.sticking = 1; // at rest against a bow that starts at rest
        }
        Self {
            model,
            torsion: torsion as u32,
            contact_count: nodes.len() as u32,
            step: 0,
            radius,
            normal_force,
            velocity,
            mu_d: mu[0],
            mu_a1: mu[1],
            mu_v1: mu[2],
            mu_a2: mu[3],
            mu_v2: mu[4],
            start_step,
            ramp_steps,
            stop_step,
            _pad: 0,
            contacts,
            thermal: bytemuck::Zeroable::zeroed(),
        }
    }
}

impl BowThermal {
    pub fn new(
        c_over_h: f32,
        conv_coef: f32,
        lam_string: f32,
        lam_bow: f32,
        window: u32,
        f_window: f32,
        exact: &[f32],
        decay: &[f32],
        weight: &[f32],
        weight_window: &[f32],
        cal: [f32; 3],
    ) -> Self {
        assert_eq!(exact.len(), 32);
        assert!(decay.len() <= BOW_SOE_TERMS && decay.len() == weight.len() && decay.len() == weight_window.len());
        assert!(window as usize >= 32 && (window as usize) + 34 < BOW_RING);
        let mut t: BowThermal = bytemuck::Zeroable::zeroed();
        t.c_over_h = c_over_h;
        t.conv_coef = conv_coef;
        t.lam_string = lam_string;
        t.lam_bow = lam_bow;
        t.window = window;
        t.f_window = f_window;
        t.terms = decay.len() as u32;
        t.cal_k1 = cal[0];
        t.cal_k2 = cal[1];
        t.cal_force = cal[2];
        t.exact.copy_from_slice(exact);
        t.decay[..decay.len()].copy_from_slice(decay);
        t.weight[..weight.len()].copy_from_slice(weight);
        t.weight_window[..weight_window.len()].copy_from_slice(weight_window);
        t
    }
}

#[derive(Debug)]
pub enum StateError {
    AdapterRequest,
    UnstableParams,
}

impl Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::AdapterRequest => {
                let _ = write!(f, "Failed to obtain adapter");
            }
            StateError::UnstableParams => {
                let _ = write!(f, "Unstable parameters");
            }
        }
        Ok(())
    }
}

impl Error for StateError {}

#[allow(dead_code)]
pub struct State {
    nodes_buffer_size: wgpu::BufferAddress,
    edges_buffer_size: wgpu::BufferAddress,
    compute_bind_group: wgpu::BindGroup,
    init_pipeline: wgpu::ComputePipeline,
    create_reference_pipeline: wgpu::ComputePipeline,
    half_step_pipeline: wgpu::ComputePipeline,
    hammer_half_step_pipeline: wgpu::ComputePipeline,
    compute_internals_pipeline: wgpu::ComputePipeline,
    compute_forces_pipeline: wgpu::ComputePipeline,
    hammer_update_pipeline: wgpu::ComputePipeline,
    output_pipeline: wgpu::ComputePipeline,
    advance_frame_pipeline: wgpu::ComputePipeline,
    device: wgpu::Device,
    fdm_uniform: FDMUniform,
    fdm_uniform_buffer: wgpu::Buffer,
    nodes_buffer: wgpu::Buffer,
    nodes_output_buffer: wgpu::Buffer,
    edges_buffer: wgpu::Buffer,
    edges_output_buffer: wgpu::Buffer,
    oversampling_factor: usize,
    push_constants: PushConstants,
    queue: wgpu::Queue,
    nodes_staging_buffer: wgpu::Buffer,
    edges_staging_buffer: wgpu::Buffer,
    bow_buffer: wgpu::Buffer,
    bow_staging_buffer: wgpu::Buffer,
    bow: Bow,
    surface_buffer: wgpu::Buffer,
    surface_staging_buffer: wgpu::Buffer,
    pick: Pick,
    hammer_friction: HammerFriction,
    initialized: bool,
    hammer_frames: usize,
}

impl State {
    pub async fn new(
        mut nodes_vec: Vec<Node>,
        mut edges_vec: Vec<Edge>,
        hammer_weights: Vec<f32>,
        hammer: Hammer,
        hammer_offset_y: f32,
        uniforms: FDMUniform,
        oversampling_factor: usize,
    ) -> Result<State, Box<dyn Error>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(StateError::AdapterRequest)?;

        let info = adapter.get_info();
        println!("Selected GPU:");
        println!("  Name:    {}", info.name);
        println!("  Vendor:  {:#06x}", info.vendor);
        println!("  Device:  {:#06x}", info.device);
        println!("  Backend: {:?}", info.backend);
        println!("  Type:    {:?}", info.device_type);
        // Flush now so the adapter is visible even if request_device fails below.
        let _ = std::io::stdout().flush();

        // The frame kernel runs as one workgroup: 1024 threads where the adapter allows it.
        let adapter_limits = adapter.limits();
        let frame_threads: u32 = if adapter_limits.max_compute_invocations_per_workgroup >= 1024
            && adapter_limits.max_compute_workgroup_size_x >= 1024
        {
            1024
        } else {
            256
        };
        println!("  Frame kernel: one workgroup of {frame_threads} threads");
        let _ = std::io::stdout().flush();
        // Nine storage buffers per stage (nodes, edges, their outputs, hammer weights, hammer, felt
        // memory, bow, surface contacts); the WebGPU default guarantees eight.
        const STORAGE_BUFFERS: u32 = 9;
        if adapter_limits.max_storage_buffers_per_shader_stage < STORAGE_BUFFERS {
            return Err(format!(
                "adapter allows {} storage buffers per shader stage, the solver needs {STORAGE_BUFFERS}",
                adapter_limits.max_storage_buffers_per_shader_stage
            )
            .into());
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::PUSH_CONSTANTS,
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits {
                            max_push_constant_size: 32,
                            max_compute_invocations_per_workgroup: frame_threads,
                            max_compute_workgroup_size_x: frame_threads,
                            max_storage_buffers_per_shader_stage: STORAGE_BUFFERS,
                            ..wgpu::Limits::default()
                        }
                    },
                    label: None,
                },
                None,
            )
            .await?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FDM Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cosserat.wgsl").into()),
        });

        let fdm_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FDM Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let nodes_buffer_size = (std::mem::size_of::<Node>() as u64
            * nodes_vec.len() as u64
            * uniforms.chunk_size as u64) as wgpu::BufferAddress;

        let edges_buffer_size = (std::mem::size_of::<Edge>() as u64
            * edges_vec.len() as u64
            * uniforms.chunk_size as u64) as wgpu::BufferAddress;

        let memory_integral = vec![0.0; nodes_vec.len()];

        nodes_vec.extend_from_within(..);
        let nodes_in = nodes_vec.into_boxed_slice();

        edges_vec.extend_from_within(..);
        let edges_in = edges_vec.into_boxed_slice();

        let hammer_weights_in = hammer_weights.into_boxed_slice();

        let nodes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("node buffer"),
            contents: bytemuck::cast_slice(nodes_in.as_ref()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let edges_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("edge buffer"),
            contents: bytemuck::cast_slice(edges_in.as_ref()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let nodes_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index buffer"),
            size: nodes_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let edges_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index buffer"),
            size: edges_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let nodes_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: nodes_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let edges_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: edges_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hammer_weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hammer weights buffer"),
            contents: bytemuck::cast_slice(hammer_weights_in.as_ref()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let hammer_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hammer buffer"),
            contents: bytemuck::cast_slice(&[hammer]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let memory_integral_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("felt memory buffer"),
            contents: bytemuck::cast_slice(&memory_integral),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Bow header and contacts, then chunk_size output frames (WGSL struct Bow).
        let bow = Bow::off();
        let bow_frames_size =
            (std::mem::size_of::<BowFrame>() as u64 * uniforms.chunk_size as u64) as wgpu::BufferAddress;
        let bow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bow buffer"),
            size: BOW_FRAMES_OFFSET + bow_frames_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&bow_buffer, 0, bytemuck::cast_slice(&[bow]));
        let bow_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bow_frames_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pick, hammer friction, per-node felt normals, then chunk_size output frames (WGSL Surface).
        let pick = Pick::off();
        let hammer_friction = HammerFriction::off();
        let surface_frames_size = (std::mem::size_of::<SurfaceFrame>() as u64
            * uniforms.chunk_size as u64) as wgpu::BufferAddress;
        let surface_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface contact buffer"),
            size: SURFACE_FRAMES_OFFSET + surface_frames_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&surface_buffer, 0, bytemuck::cast_slice(&[pick]));
        queue.write_buffer(
            &surface_buffer,
            std::mem::size_of::<Pick>() as u64,
            bytemuck::cast_slice(&[hammer_friction]),
        );
        let surface_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: surface_frames_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let push_constants = PushConstants {
            current_index: 0,
            future_index: 1,
            output_index: 0,
            hammering: 0,

            write_output: 1,
            _reserved1: 0.0,
            hammer_offset_y: hammer_offset_y,
            steps: oversampling_factor as u32,
        };

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Compute Bind Group Layout"),
            });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fdm_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nodes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: edges_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: nodes_output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: edges_output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: hammer_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: hammer_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: memory_integral_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: bow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: surface_buffer.as_entire_binding(),
                },
            ],
            label: Some("Compute Bind Group Layout"),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..32,
                }],
            });

        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "init",
            compilation_options: Default::default(),
        });

        let create_reference_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "create_references",
                compilation_options: Default::default(),
            });

        let half_step_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "half_step",
            compilation_options: Default::default(),
        });

        let hammer_half_step_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "hammer_half_step",
                compilation_options: Default::default(),
            });

        let compute_internals_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "compute_internals",
                compilation_options: Default::default(),
            });

        let compute_forces_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "compute_forces",
                compilation_options: Default::default(),
            });

        let hammer_update_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "hammer_update",
                compilation_options: Default::default(),
            });

        let output_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "save_output",
            compilation_options: Default::default(),
        });

        let advance_frame_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Advance Frame Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: if frame_threads == 1024 { "advance_frame_1024" } else { "advance_frame_256" },
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            fdm_uniform: uniforms,
            fdm_uniform_buffer,
            compute_bind_group,
            nodes_buffer,
            edges_buffer,
            edges_output_buffer,
            nodes_output_buffer,
            nodes_staging_buffer,
            edges_staging_buffer,
            bow_buffer,
            bow_staging_buffer,
            bow,
            surface_buffer,
            surface_staging_buffer,
            pick,
            hammer_friction,
            push_constants,
            init_pipeline,
            create_reference_pipeline,
            half_step_pipeline,
            hammer_half_step_pipeline,
            compute_internals_pipeline,
            compute_forces_pipeline,
            hammer_update_pipeline,
            output_pipeline,
            advance_frame_pipeline,
            nodes_buffer_size,
            edges_buffer_size,
            oversampling_factor,
            initialized: false,
            hammer_frames: 0,
        })
    }

    pub fn initialize(&mut self) -> Result<(), Box<dyn Error>> {
        println!("Initializing simulation!");
        let num_dispatches = self.fdm_uniform.node_count.div_ceil(64);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.set_pipeline(&self.init_pipeline);
            compute_pass.set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
            compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
        }
        self.queue.submit(iter::once(encoder.finish()));

        self.initialized = true;
        Ok(())
    }

    pub fn create_references(&mut self) -> Result<(), Box<dyn Error>> {
        println!("Considering current configuration as stress free.");
        let num_dispatches = self.fdm_uniform.node_count.div_ceil(64);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.set_pipeline(&self.create_reference_pipeline);
            compute_pass.set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
            compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
        }
        self.queue.submit(iter::once(encoder.finish()));

        self.initialized = true;
        Ok(())
    }

    pub fn enable_hammer(&mut self, enable: bool) -> Result<(), Box<dyn Error>> {
        if enable {
            self.push_constants.hammering = 1;
        } else {
            self.push_constants.hammering = 0;
        }
        Ok(())
    }

    /// One chunk (chunk_size output frames). Each frame is one dispatch of the single-workgroup
    /// `advance_frame` kernel, which runs all `oversampling_factor` solver steps of the frame with
    /// storage barriers between the phases, then writes the output frame. All frames go into one
    /// compute pass; wgpu synchronizes consecutive dispatches. Same arithmetic as
    /// `compute_reference`, with 1 dispatch per frame instead of 5 passes per solver step.
    pub fn compute(&mut self) -> Result<(), Box<dyn Error>> {
        // No single GPU job may run long: a GPU that also drives a display resets work that blocks
        // its graphics queue (amdgpu "ring gfx timeout", seen at 320 steps in one dispatch, ~18 ms on an
        // RX 460; none at 160). Each extra dispatch boundary costs about as much as 32 steps there.
        // So a frame is split into dispatches of at most MAX_STEPS_PER_DISPATCH steps (only the
        // last writes the output frame), and frames are submitted in batches.
        const FRAMES_PER_SUBMIT: u32 = 16;
        const MAX_STEPS_PER_DISPATCH: usize = 160;
        let mut frame = 0;
        while frame < self.fdm_uniform.chunk_size {
            let batch = FRAMES_PER_SUBMIT.min(self.fdm_uniform.chunk_size - frame);
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Advance Frames Encoder"),
                });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Advance Frames Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                compute_pass.set_pipeline(&self.advance_frame_pipeline);
                for _ in 0..batch {
                    let mut remaining = self.oversampling_factor;
                    while remaining > 0 {
                        let steps = remaining.min(MAX_STEPS_PER_DISPATCH);
                        self.push_constants.steps = steps as u32;
                        self.push_constants.write_output = (steps == remaining) as u32;
                        compute_pass
                            .set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                        compute_pass.dispatch_workgroups(1, 1, 1);

                        // The kernel swaps current/future once per step.
                        if steps % 2 == 1 {
                            (
                                self.push_constants.current_index,
                                self.push_constants.future_index,
                            ) = (
                                self.push_constants.future_index,
                                self.push_constants.current_index,
                            );
                        }
                        remaining -= steps;
                    }
                    self.push_constants.output_index =
                        (self.push_constants.output_index + 1) % self.fdm_uniform.chunk_size;
                }
            }
            let index = self.queue.submit(iter::once(encoder.finish()));
            // Wait for each batch: wgpu-core 0.21 waits on a fence for at most 5 s
            // (CLEANUP_WAIT_MS) and then treats the submission as finished anyway, so if a whole
            // chunk were queued and took longer, save() would map its staging buffer before the
            // copy ran and return stale frames. One batch takes a fraction of a second.
            self.device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(index));
            frame += batch;
        }
        Ok(())
    }

    /// The original multi-pass loop (5 compute passes per solver step), kept as the reference
    /// that `compute` is checked against.
    pub fn compute_reference(&mut self) -> Result<(), Box<dyn Error>> {
        // The bow phases exist only in the frame kernel.
        assert!(self.bow.model == 0, "compute_reference does not support the bow");
        assert!(
            self.pick.enabled == 0 && self.hammer_friction.enabled == 0,
            "compute_reference does not support the pick or the felt friction"
        );
        // The 64 comes from the @workgroup_size(64) inside the shaders
        let num_dispatches = self.fdm_uniform.node_count.div_ceil(64);

        // Too many commands break command buffers.
        // Compute CHUNK_SIZE * OVERSAMPLING_FACTOR iterations.
        // Every OVERSAMPLING_FACTOR generate a new command buffer, to avoid overflow.
        for _ in 0..self.fdm_uniform.chunk_size {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Encoder"),
                });
            // FOR LOOP START
            for _ in 0..self.oversampling_factor {
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Compute Pass"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                    compute_pass.set_pipeline(&self.half_step_pipeline);
                    compute_pass
                        .set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                    compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
                }
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Compute Pass"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                    compute_pass.set_pipeline(&self.hammer_half_step_pipeline);
                    compute_pass
                        .set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                    compute_pass.dispatch_workgroups(1, 1, 1);
                }
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Compute Pass"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                    compute_pass.set_pipeline(&self.compute_internals_pipeline);
                    compute_pass
                        .set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                    compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
                }
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Compute Pass"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                    compute_pass.set_pipeline(&self.compute_forces_pipeline);
                    compute_pass
                        .set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                    compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
                }
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Compute Pass"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                    compute_pass.set_pipeline(&self.hammer_update_pipeline);
                    compute_pass
                        .set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                    compute_pass.dispatch_workgroups(1, 1, 1);
                }
                (
                    self.push_constants.current_index,
                    self.push_constants.future_index,
                ) = (
                    self.push_constants.future_index,
                    self.push_constants.current_index,
                );
            }
            // FOR LOOP END
            self.queue.submit(iter::once(encoder.finish()));
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Encoder"),
                });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                compute_pass.set_pipeline(&self.output_pipeline);
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);

                self.push_constants.output_index =
                    (self.push_constants.output_index + 1) % self.fdm_uniform.chunk_size;
            }
            self.queue.submit(iter::once(encoder.finish()));
        }
        Ok(())
    }

    pub fn save(&mut self) -> Result<(Vec<Vec<Node>>, Vec<Vec<Edge>>), Box<dyn Error>> {
        assert_eq!(self.push_constants.output_index, 0);
        // Begin memory transfer to CPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &self.nodes_output_buffer,
            0,
            &self.nodes_staging_buffer,
            0,
            self.nodes_buffer_size,
        );
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let nodes_buffer_slice = self.nodes_staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        nodes_buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Awaits until `buffer_future` can be read from
        let nodes = if let Ok(Ok(())) = rx.recv() {
            let data = nodes_buffer_slice.get_mapped_range();
            let result: Vec<Node> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.nodes_staging_buffer.unmap(); // Unmaps buffer from memory

            let vecs: Vec<Vec<Node>> = result
                .chunks(self.fdm_uniform.node_count as usize)
                .map(|slice| slice.to_vec())
                .collect();
            Some(vecs)
        } else {
            None
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &self.edges_output_buffer,
            0,
            &self.edges_staging_buffer,
            0,
            self.edges_buffer_size,
        );
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let edges_buffer_slice = self.edges_staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        edges_buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        let edges = if let Ok(Ok(())) = rx.recv() {
            let data = edges_buffer_slice.get_mapped_range();
            let result: Vec<Edge> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.edges_staging_buffer.unmap(); // Unmaps buffer from memory

            let vecs: Vec<Vec<Edge>> = result
                .chunks(self.fdm_uniform.node_count as usize)
                .map(|slice| slice.to_vec())
                .collect();
            Some(vecs)
        } else {
            None
        };
        if let (Some(n), Some(e)) = (nodes, edges) {
            return Ok((n, e));
        }

        Ok((Vec::new(), Vec::new()))
    }

    pub fn set_twist_rate_damping(&mut self, twist_rate_damping: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.twist_rate_damping = twist_rate_damping;
        self.write_uniforms()
    }

    /// Thermal-model parameters for the next set_bow with model 2.
    pub fn set_bow_thermal(&mut self, thermal: BowThermal) -> Result<(), Box<dyn Error>> {
        self.bow.thermal = thermal;
        Ok(())
    }

    /// Replaces the bow (header and contacts; resets its clock, contact states, temperature
    /// histories). Keeps the thermal parameters of set_bow_thermal.
    pub fn set_bow(&mut self, mut bow: Bow) -> Result<(), Box<dyn Error>> {
        bow.thermal = self.bow.thermal;
        if bow.model == 2 {
            assert!(bow.thermal.terms > 0, "set_bow_thermal first");
            let w0 = bow.contacts[0].weight;
            // one contact size (and history window) for all contacts
            assert!(bow.contacts[..bow.contact_count as usize].iter().all(|c| c.weight == w0));
        }
        for c in &bow.contacts[..bow.contact_count as usize] {
            let j = c.node;
            // node j and edge j must both be dynamic (see the clamp ranges in compute_forces)
            assert!(
                j > self.fdm_uniform.clamp_offset
                    && j + 2 + self.fdm_uniform.clamp_offset < self.fdm_uniform.node_count,
                "bow contact node {j} too close to an end"
            );
        }
        self.bow = bow;
        self.queue
            .write_buffer(&self.bow_buffer, 0, bytemuck::cast_slice(&[self.bow]));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(
            &self.bow_buffer,
            std::mem::size_of::<Bow>() as u64,
            Some(BOW_FRAMES_OFFSET - std::mem::size_of::<Bow>() as u64),
        );
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// The bow output frames of the last chunk (call after save, same chunk).
    pub fn save_bow(&mut self) -> Result<Vec<BowFrame>, Box<dyn Error>> {
        let size = self.bow_staging_buffer.size();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &self.bow_buffer,
            BOW_FRAMES_OFFSET,
            &self.bow_staging_buffer,
            0,
            size,
        );
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let slice = self.bow_staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        let frames = if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let result: Vec<BowFrame> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.bow_staging_buffer.unmap();
            result
        } else {
            Vec::new()
        };
        Ok(frames)
    }

    pub fn bow_contact_count(&self) -> usize {
        self.bow.contact_count as usize
    }

    fn assert_dynamic_node(&self, j: u32, what: &str) {
        // node j and edge j must both be dynamic (see the clamp ranges in compute_forces)
        assert!(
            j > self.fdm_uniform.clamp_offset
                && j + 2 + self.fdm_uniform.clamp_offset < self.fdm_uniform.node_count,
            "{what} node {j} too close to an end"
        );
    }

    /// Replaces the pick (resets its clock and contact record).
    pub fn set_pick(&mut self, pick: Pick) -> Result<(), Box<dyn Error>> {
        if pick.enabled == 1 {
            self.assert_dynamic_node(pick.node, "pick");
            assert!(
                self.bow.contact_count == 0
                    || self.bow.contacts[..self.bow.contact_count as usize]
                        .iter()
                        .all(|c| c.node != pick.node),
                "pick and bow on the same node"
            );
        }
        self.pick = pick;
        // only this header: the GPU advances the other one
        self.queue
            .write_buffer(&self.surface_buffer, 0, bytemuck::cast_slice(&[self.pick]));
        Ok(())
    }

    /// Replaces the hammer contact geometry and felt friction.
    pub fn set_hammer_friction(&mut self, h: HammerFriction) -> Result<(), Box<dyn Error>> {
        if h.enabled == 1 {
            assert!(
                self.fdm_uniform.node_count as usize <= MAX_SURFACE_NODES,
                "felt friction supports at most {MAX_SURFACE_NODES} nodes"
            );
            for c in &h.contacts[..h.count as usize] {
                self.assert_dynamic_node(c.node, "hammer friction");
            }
        }
        self.hammer_friction = h;
        self.queue.write_buffer(
            &self.surface_buffer,
            std::mem::size_of::<Pick>() as u64,
            bytemuck::cast_slice(&[self.hammer_friction]),
        );
        Ok(())
    }

    /// The pick and hammer friction output frames of the last chunk (call after save, same chunk).
    pub fn save_surface(&mut self) -> Result<Vec<SurfaceFrame>, Box<dyn Error>> {
        let size = self.surface_staging_buffer.size();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &self.surface_buffer,
            SURFACE_FRAMES_OFFSET,
            &self.surface_staging_buffer,
            0,
            size,
        );
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let slice = self.surface_staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        let frames = if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let result: Vec<SurfaceFrame> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.surface_staging_buffer.unmap();
            result
        } else {
            Vec::new()
        };
        Ok(frames)
    }

    pub fn hammer_friction_count(&self) -> usize {
        self.hammer_friction.count as usize
    }

    fn write_uniforms(&mut self) -> Result<(), Box<dyn Error>> {
        // Same bytes as the initial upload: the repr(C) layout is asserted to match the WGSL one.
        self.queue.write_buffer(
            &self.fdm_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.fdm_uniform]),
        );
        Ok(())
    }

    pub fn set_dt(&mut self, dt: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.dt = dt;
        self.fdm_uniform.dt_inv = dt.recip();

        self.write_uniforms()
    }

    pub fn set_damping(
        &mut self,
        linear_damping: f32,
        angular_damping: [f32; 3],
    ) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.linear_damping = linear_damping;
        self.fdm_uniform.angular_damping = angular_damping;
        self.write_uniforms()
    }
}
