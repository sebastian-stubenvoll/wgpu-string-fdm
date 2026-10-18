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
    assert!(size_of::<Node>() == 96);
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
    _pad4: [u32; 3],
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
            _pad4: [0; 3],
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
