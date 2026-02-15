#![allow(dead_code)]
use std::{error::Error, fmt::Display, iter};
use wgpu::util::DeviceExt;

trait WgslPad<T> {
    fn pad(self) -> T;
}

impl WgslPad<[f32; 4]> for [f32; 3] {
    fn pad(self) -> [f32; 4] {
        [self[0], self[1], self[2], 0.0]
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
// Don't derive Default on purpose! Implicit initialization via Default::default() can be a footgun here!
pub struct Node {
    displacement: [f32; 3],
    _pad0: u32,
    velocity: [f32; 3],
    _pad1: u32,
    curvature: [f32; 3],
    _pad2: u32,
    reference_rotation: [f32; 4],
    internal_moment: [f32; 3],
    _pad3: u32,
}

impl Node {
    pub fn new(
        displacement: &[f32; 3],
        velocity: &[f32; 3],
        reference_rotation: &[f32; 4],
    ) -> Self {
        Self {
            displacement: *displacement,
            _pad0: 0,
            velocity: *velocity,
            _pad1: 0,
            curvature: [0.0; 3],
            _pad2: 0,
            reference_rotation: *reference_rotation,
            internal_moment: [0.0; 3],
            _pad3: 0,
        }
    }

    pub fn to_raw(self) -> [[f32; 3]; 4] {
        [
            self.displacement,
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
    len: f32,
    strain: [f32; 3],
    _pad0: u32,
    reference_strain: [f32; 3],
    _pad1: u32,
    internal_force: [f32; 3],
    _pad2: u32,
    reference_vector: [f32; 3],
    _pad3: u32,
}

impl Edge {
    pub fn new(
        orientation: &[f32; 4],
        reference_vector: &[f32; 3],
        angular_velocity: &[f32; 3],
        reference_strain: &[f32; 3],
    ) -> Self {
        Self {
            orientation: *orientation,
            angular_velocity: *angular_velocity,
            len: 0.0,
            strain: [0.0; 3],
            _pad0: 0,
            reference_strain: *reference_strain,
            _pad1: 0,
            internal_force: [0.0; 3],
            _pad2: 0,
            reference_vector: *reference_vector,
            _pad3: 0,
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
    dampening: f32,
    inertia: [f32; 3],
    _pad2: u32,
    inertia_inv: [f32; 3],
    _pad3: u32,
}

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
            dampening: 0.0,
            inertia: *intertia,
            _pad2: 0,
            inertia_inv: intertia.map(|i| 1.0 / i),
            _pad3: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    current_index: u32,
    future_index: u32,
    output_index: u32,
    force: f32,

    linear_dampening: f32,
    angular_dampening: f32,
    _pad0: u32,
    _pad1: u32,
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

struct NodeBuffers {
    displacements: wgpu::Buffer,
    velocities: wgpu::Buffer,
    curvatures: wgpu::Buffer,
    reference_rotations: wgpu::Buffer,
    internal_moments: wgpu::Buffer,
}

impl NodeBuffers {
    fn new(device: &wgpu::Device, nodes_vec: &[Node]) -> Self {
        let displacement_vals = nodes_vec
            .iter()
            .map(|n| n.displacement.pad())
            .collect::<Vec<[f32; 4]>>();

        let displacements = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("displacements buffer"),
            contents: bytemuck::cast_slice(displacement_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let velocity_vals = nodes_vec
            .iter()
            .map(|n| n.velocity.pad())
            .collect::<Vec<[f32; 4]>>();

        let velocities = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("velocities buffer"),
            contents: bytemuck::cast_slice(velocity_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let curvature_vals = nodes_vec
            .iter()
            .map(|n| n.curvature.pad())
            .collect::<Vec<[f32; 4]>>();

        let curvatures = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("curvatures buffer"),
            contents: bytemuck::cast_slice(curvature_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let reference_rotation_vals = nodes_vec
            .iter()
            .map(|n| n.reference_rotation)
            .collect::<Vec<[f32; 4]>>();

        let reference_rotations = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reference rotations buffer"),
            contents: bytemuck::cast_slice(reference_rotation_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let internal_moment_vals = nodes_vec
            .iter()
            .map(|n| n.internal_moment.pad())
            .collect::<Vec<[f32; 4]>>();

        let internal_moments = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("internal_moments buffer"),
            contents: bytemuck::cast_slice(internal_moment_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        Self {
            displacements,
            velocities,
            curvatures,
            reference_rotations,
            internal_moments,
        }
    }
}

struct EdgeBuffers {
    orientations: wgpu::Buffer,
    angular_velocities: wgpu::Buffer,
    strains: wgpu::Buffer,
    reference_strains: wgpu::Buffer,
    internal_forces: wgpu::Buffer,
    reference_vectors: wgpu::Buffer,
    inverse_lengths: wgpu::Buffer,
    dilatations: wgpu::Buffer,
}

impl EdgeBuffers {
    fn new(device: &wgpu::Device, edges_vec: &[Edge]) -> Self {
        let orientation_vals = edges_vec
            .iter()
            .map(|e| e.orientation)
            .collect::<Vec<[f32; 4]>>();

        let orientations = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("orientations buffer"),
            contents: bytemuck::cast_slice(orientation_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let angular_velocity_vals = edges_vec
            .iter()
            .map(|e| e.angular_velocity.pad())
            .collect::<Vec<[f32; 4]>>();

        let angular_velocities = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("angular_velocitys buffer"),
            contents: bytemuck::cast_slice(angular_velocity_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let strain_vals = edges_vec
            .iter()
            .map(|e| e.strain.pad())
            .collect::<Vec<[f32; 4]>>();

        let strains = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("strains buffer"),
            contents: bytemuck::cast_slice(strain_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let reference_strain_vals = edges_vec
            .iter()
            .map(|e| e.reference_strain.pad())
            .collect::<Vec<[f32; 4]>>();

        let reference_strains = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reference_strains buffer"),
            contents: bytemuck::cast_slice(reference_strain_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let internal_force_vals = edges_vec
            .iter()
            .map(|e| e.internal_force.pad())
            .collect::<Vec<[f32; 4]>>();

        let internal_forces = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("internal_forces buffer"),
            contents: bytemuck::cast_slice(internal_force_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let reference_vector_vals = edges_vec
            .iter()
            .map(|e| e.reference_vector.pad())
            .collect::<Vec<[f32; 4]>>();

        let reference_vectors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reference_vectors buffer"),
            contents: bytemuck::cast_slice(reference_vector_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let inverse_length_vals = edges_vec
            .iter()
            .map(|e| e.len.recip())
            .collect::<Vec<f32>>();

        let inverse_lengths = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inverse_lengths buffer"),
            contents: bytemuck::cast_slice(inverse_length_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let dilatation_vals = edges_vec.iter().map(|_| 1.0).collect::<Vec<f32>>();

        let dilatations = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dilatations buffer"),
            contents: bytemuck::cast_slice(dilatation_vals.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        Self {
            orientations,
            angular_velocities,
            strains,
            reference_strains,
            internal_forces,
            reference_vectors,
            inverse_lengths,
            dilatations,
        }
    }
}

struct OutputBuffers {
    nodes: wgpu::Buffer,
    edges: wgpu::Buffer,
}

impl OutputBuffers {
    fn new(device: &wgpu::Device, nodes_in: &[Node], edges_in: &[Edge], chunk_size: u64) -> Self {
        let nodes_buffer_size = (std::mem::size_of::<Node>() as u64
            * nodes_in.len() as u64
            * chunk_size) as wgpu::BufferAddress;
        let edges_buffer_size = (std::mem::size_of::<Edge>() as u64
            * edges_in.len() as u64
            * chunk_size) as wgpu::BufferAddress;

        let nodes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nodes output buffer"),
            size: nodes_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let edges = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edges output buffer"),
            size: edges_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self { nodes, edges }
    }
}

struct StagingBuffers {
    nodes: wgpu::Buffer,
    edges: wgpu::Buffer,
    nodes_buffer_size: u64,
    edges_buffer_size: u64,
}

impl StagingBuffers {
    fn new(device: &wgpu::Device, nodes_in: &[Node], edges_in: &[Edge], chunk_size: u64) -> Self {
        let nodes_buffer_size = (std::mem::size_of::<Node>() as u64
            * nodes_in.len() as u64
            * chunk_size) as wgpu::BufferAddress;
        let edges_buffer_size = (std::mem::size_of::<Edge>() as u64
            * edges_in.len() as u64
            * chunk_size) as wgpu::BufferAddress;

        let nodes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nodes staging buffer"),
            size: nodes_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let edges = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edges staging buffer"),
            size: edges_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            nodes,
            edges,
            nodes_buffer_size,
            edges_buffer_size,
        }
    }
}

#[allow(dead_code)]
pub struct State {
    compute_bind_group: wgpu::BindGroup,
    init_pipeline: wgpu::ComputePipeline,
    create_reference_pipeline: wgpu::ComputePipeline,
    half_step_pipeline: wgpu::ComputePipeline,
    compute_internals_pipeline: wgpu::ComputePipeline,
    compute_forces_pipeline: wgpu::ComputePipeline,
    output_pipeline: wgpu::ComputePipeline,
    device: wgpu::Device,
    fdm_uniform: FDMUniform,
    fdm_uniform_buffer: wgpu::Buffer,
    hammer_weights_buffer: wgpu::Buffer,
    node_buffers: NodeBuffers,
    edge_buffers: EdgeBuffers,
    output_buffers: OutputBuffers,
    staging_buffers: StagingBuffers,
    oversampling_factor: usize,
    push_constants: PushConstants,
    queue: wgpu::Queue,
    initialized: bool,
}

impl State {
    pub async fn new(
        mut nodes_vec: Vec<Node>,
        mut edges_vec: Vec<Edge>,
        hammer_weights: Vec<[f32; 4]>,
        uniforms: FDMUniform,
        oversampling_factor: usize,
        dampening: [f32; 2],
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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::PUSH_CONSTANTS,
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits {
                            max_push_constant_size: 32,
                            max_storage_buffers_per_shader_stage: 16,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cosserat_improved.wgsl").into()),
        });

        let fdm_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FDM Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffers =
            OutputBuffers::new(&device, &nodes_vec, &edges_vec, uniforms.chunk_size as u64);

        let staging_buffers =
            StagingBuffers::new(&device, &nodes_vec, &edges_vec, uniforms.chunk_size as u64);

        nodes_vec.extend_from_within(..);
        edges_vec.extend_from_within(..);

        let node_buffers = NodeBuffers::new(&device, &nodes_vec);
        let edge_buffers = EdgeBuffers::new(&device, &edges_vec);

        let hammer_weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hammer weights buffer"),
            contents: bytemuck::cast_slice(hammer_weights.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let push_constants = PushConstants {
            current_index: 0,
            future_index: 1,
            output_index: 0,
            force: 0.0,

            linear_dampening: dampening[0],
            angular_dampening: dampening[1],
            _pad0: 0,
            _pad1: 0,
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 13,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 14,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 15,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 16,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("compute bind group layout"),
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
                    resource: hammer_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: node_buffers.displacements.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: node_buffers.velocities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: node_buffers.curvatures.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: node_buffers.reference_rotations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: node_buffers.internal_moments.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: edge_buffers.orientations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: edge_buffers.angular_velocities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: edge_buffers.strains.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: edge_buffers.reference_strains.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: edge_buffers.internal_forces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: edge_buffers.reference_vectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: edge_buffers.inverse_lengths.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: edge_buffers.dilatations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: output_buffers.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: output_buffers.edges.as_entire_binding(),
                },
            ],
            label: Some("compute bind group"),
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

        let output_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "save_output",
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            fdm_uniform: uniforms,
            fdm_uniform_buffer,
            compute_bind_group,
            hammer_weights_buffer,
            node_buffers,
            edge_buffers,
            output_buffers,
            staging_buffers,
            push_constants,
            init_pipeline,
            create_reference_pipeline,
            half_step_pipeline,
            compute_internals_pipeline,
            compute_forces_pipeline,
            output_pipeline,
            oversampling_factor,
            initialized: false,
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

    pub fn hammer(&mut self, steps: usize, force: f32) -> Result<(), Box<dyn Error>> {
        // This is basically a normal simulation pass, except we are setting the external force to >0.0 for this many steps
        // The 64 comes from the @workgroup_size(64) inside the shaders
        let num_dispatches = self.fdm_uniform.node_count.div_ceil(64);
        self.push_constants.force = force;
        for _ in 0..steps {
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
                compute_pass.set_pipeline(&self.half_step_pipeline);
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
            }
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                compute_pass.set_pipeline(&self.compute_internals_pipeline);
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
            }
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                compute_pass.set_pipeline(&self.compute_forces_pipeline);
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
            }
            self.queue.submit(iter::once(encoder.finish()));
            (
                self.push_constants.current_index,
                self.push_constants.future_index,
            ) = (
                self.push_constants.future_index,
                self.push_constants.current_index,
            );
        }
        // Remove external force
        self.push_constants.force = 0.0;
        Ok(())
    }

    pub fn compute(&mut self) -> Result<(), Box<dyn Error>> {
        // The 64 comes from the @workgroup_size(64) inside the shaders
        let num_dispatches = self.fdm_uniform.node_count.div_ceil(64);

        // Too many commands break command buffers.
        // Compute CHUNK_SIZE * OVERSAMPLING_FACTOR iterations.
        // Every OVERSAMPLING_FACTOR generate a new command buffer, to avoid overflow.
        for _ in 0..self.fdm_uniform.chunk_size {
            // FOR LOOP START
            for _ in 0..self.oversampling_factor {
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Compute Encoder"),
                        });
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
                self.queue.submit(iter::once(encoder.finish()));
                (
                    self.push_constants.current_index,
                    self.push_constants.future_index,
                ) = (
                    self.push_constants.future_index,
                    self.push_constants.current_index,
                );
            }
            // FOR LOOP END
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
            &self.output_buffers.nodes,
            0,
            &self.staging_buffers.nodes,
            0,
            self.staging_buffers.nodes_buffer_size,
        );
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let nodes_buffer_slice = self.staging_buffers.nodes.slice(..);
        let (tx, rx) = flume::bounded(1);
        nodes_buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Awaits until `buffer_future` can be read from
        let nodes = if let Ok(Ok(())) = rx.recv() {
            let data = nodes_buffer_slice.get_mapped_range();
            let result: Vec<Node> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.staging_buffers.nodes.unmap(); // Unmaps buffer from memory

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
            &self.output_buffers.edges,
            0,
            &self.staging_buffers.edges,
            0,
            self.staging_buffers.edges_buffer_size,
        );
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let edges_buffer_slice = self.staging_buffers.edges.slice(..);
        let (tx, rx) = flume::bounded(1);
        edges_buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        let edges = if let Ok(Ok(())) = rx.recv() {
            let data = edges_buffer_slice.get_mapped_range();
            let result: Vec<Edge> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.staging_buffers.edges.unmap(); // Unmaps buffer from memory

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

    fn write_uniforms(&mut self) -> Result<(), Box<dyn Error>> {
        let mut buffer = encase::UniformBuffer::new(Vec::new());
        buffer.write(&self.fdm_uniform)?;
        self.queue
            .write_buffer(&self.fdm_uniform_buffer, 0, &buffer.into_inner());
        Ok(())
    }

    pub fn set_dt(&mut self, dt: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.dt = dt;
        self.fdm_uniform.dt_inv = dt.recip();

        self.write_uniforms()
    }

    pub fn set_dampening(&mut self, dampening: [f32; 2]) -> Result<(), Box<dyn Error>> {
        [
            self.push_constants.linear_dampening,
            self.push_constants.angular_dampening,
        ] = dampening;
        Ok(())
    }
}
