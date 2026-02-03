#![allow(dead_code)]
use std::{error::Error, fmt::Display, iter};
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
    _pad2: u32,
    reference_curvature: [f32; 4],
    internal_moment: [f32; 3],
    _pad3: u32,
}

impl Node {
    pub fn new(position: &[f32; 3], velocity: &[f32; 3]) -> Self {
        Self {
            position: *position,
            _pad0: 0,
            velocity: *velocity,
            _pad1: 0,
            curvature: [0.0; 3],
            _pad2: 0,
            reference_curvature: [0.0; 4],
            internal_moment: [0.0; 3],
            _pad3: 0,
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
    dilation: f32,
    reference_strain: [f32; 3],
    _pad0: u32,
    internal_force: [f32; 3],
    _pad1: u32,
}

impl Edge {
    pub fn new(orientation: &[f32; 4], angular_velocity: &[f32; 3]) -> Self {
        Self {
            orientation: *orientation,
            angular_velocity: *angular_velocity,
            len_inv: 0.0,
            strain: [0.0; 3],
            dilation: 0.0,
            reference_strain: [0.0; 3],
            _pad0: 0,
            internal_force: [0.0; 3],
            _pad1: 0,
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
            clamp_offset: 0,
            stiffness_bt: *stiffness_bt,
            _pad1: 0,
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
    pad0: u32,
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
    create_reference_pipeline: wgpu::ComputePipeline,
    half_step_pipeline: wgpu::ComputePipeline,
    compute_internals_pipeline: wgpu::ComputePipeline,
    compute_forces_pipeline: wgpu::ComputePipeline,
    output_pipeline: wgpu::ComputePipeline,
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
}

impl State {
    pub async fn new(
        mut nodes_vec: Vec<Node>,
        mut edges_vec: Vec<Edge>,
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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::PUSH_CONSTANTS,
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits {
                            max_push_constant_size: 16,
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

        nodes_vec.extend_from_within(..);
        let nodes_in = nodes_vec.into_boxed_slice();

        edges_vec.extend_from_within(..);
        let edges_in = edges_vec.into_boxed_slice();

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

        let push_constants = PushConstants {
            current_index: 0,
            future_index: 1,
            output_index: 0,
            pad0: 0,
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
            ],
            label: Some("Compute Bind Group Layout"),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });

        let create_reference_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "create_reference",
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
            nodes_buffer,
            edges_buffer,
            edges_output_buffer,
            nodes_output_buffer,
            nodes_staging_buffer,
            edges_staging_buffer,
            push_constants,
            create_reference_pipeline,
            half_step_pipeline,
            compute_internals_pipeline,
            compute_forces_pipeline,
            output_pipeline,
            nodes_buffer_size,
            edges_buffer_size,
            oversampling_factor,
            initialized: false,
        })
    }

    pub fn initialize(&mut self, vel: f32, steps: usize) -> Result<(), Box<dyn Error>> {
        println!("Initializing simulation!");
        let num_dispatches = self.fdm_uniform.node_count.div_ceil(64);

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
                compute_pass.set_pipeline(&self.create_reference_pipeline);
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
            }
            self.queue.submit(iter::once(encoder.finish()));
        }

        self.initialized = true;
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

    fn update_uniforms(&mut self, uniforms: FDMUniform) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform = uniforms;
        let mut buffer = encase::UniformBuffer::new(Vec::new());
        buffer.write(&self.fdm_uniform)?;
        self.queue
            .write_buffer(&self.fdm_uniform_buffer, 0, &buffer.into_inner());
        Ok(())
    }
}
