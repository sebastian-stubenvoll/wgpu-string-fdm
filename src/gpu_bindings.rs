#![allow(dead_code)]
use std::{error::Error, fmt::Display, iter};

use wgpu::util::DeviceExt;
#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
pub struct Node {
    positions: [f32; 3],
    _padding0: u32,
    velocities: [f32; 3],
    _padding1: u32,
    angles: [f32; 3],
    _padding2: u32,
    angular_velocites: [f32; 3],
    _padding3: u32,
    moments: [f32; 3],
    _padding4: u32,
    forces: [f32; 3],
    _padding5: u32,
}

impl Node {
    pub fn new(
        positions: [f32; 3],
        velocities: [f32; 3],
        angles: [f32; 3],
        angular_velocites: [f32; 3],
    ) -> Self {
        Self {
            positions,
            velocities,
            angles,
            angular_velocites,
            ..Default::default()
        }
    }

    pub fn to_raw(self) -> [[f32; 3]; 6] {
        [
            self.positions,
            self.velocities,
            self.angles,
            self.angular_velocites,
            self.moments,
            self.forces,
        ]
    }
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
pub struct FDMUniform {
    dt: f32,
    ds: f32,
    node_count: u32,
    loss: f32,

    tau: f32,
    kappa: f32,
    m_coil: f32,
    c2_core: f32,

    beta: [f32; 3],
    _padding3: u32,

    sigma: [f32; 3],
    _padding4: u32,

    k: [f32; 3],
    _padding5: u32,
}

impl FDMUniform {
    pub fn new(
        dt: f32,
        ds: f32,
        nodes: &[Node],
        loss: f32,
        tau: f32,
        kappa: f32,
        m_coil: f32,
        c2_core: f32,
        beta: [f32; 3],
        sigma: [f32; 3],
        k: [f32; 3],
    ) -> Self {
        Self {
            dt,
            ds,
            node_count: nodes.len() as u32,
            loss,
            tau,
            kappa,
            m_coil,
            c2_core,
            beta,
            sigma,
            k,
            ..Default::default()
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    current_index: u32,
    future_index: u32,
    output_index: u32,
    save: u32,
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
    buffer_size: usize,
    compute_bind_group: wgpu::BindGroup,
    external_pipeline: wgpu::ComputePipeline,
    moments_pipeline: wgpu::ComputePipeline,
    positions_pipeline: wgpu::ComputePipeline,
    output_pipeline: wgpu::ComputePipeline,
    device: wgpu::Device,
    fdm_uniform: FDMUniform,
    fdm_uniform_buffer: wgpu::Buffer,
    nodes_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    oversampling_factor: usize,
    push_constants: PushConstants,
    queue: wgpu::Queue,
    staging_buffer: wgpu::Buffer,
}

impl State {
    pub async fn new(
        nodes_vec: Vec<Node>,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/wittrick.wgsl").into()),
        });

        //Uniform Buffers
        let fdm_uniform = uniforms;

        let fdm_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FDM Uniform Buffer"),
            contents: bytemuck::cast_slice(&[fdm_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let nodes = nodes_vec
            .into_iter()
            .map(|n| [n.clone(), n])
            .collect::<Vec<[Node; 2]>>()
            .into_boxed_slice();

        let nodes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("node buffer"),
            contents: bytemuck::cast_slice(nodes.as_ref()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index buffer"),
            contents: bytemuck::cast_slice(nodes.as_ref()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Staging
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of_val(&*(nodes)) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let push_constants = PushConstants {
            current_index: 0,
            future_index: 1,
            output_index: 0,
            save: 0,
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
                    resource: output_buffer.as_entire_binding(),
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

        let external_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "apply_ext_force",
            compilation_options: Default::default(),
        });

        let moments_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "update_moments",
            compilation_options: Default::default(),
        });

        let positions_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "update_positions",
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
            fdm_uniform,
            fdm_uniform_buffer,
            compute_bind_group,
            nodes_buffer,
            output_buffer,
            staging_buffer,
            push_constants,
            external_pipeline,
            moments_pipeline,
            positions_pipeline,
            output_pipeline,
            buffer_size: std::mem::size_of_val(&(*nodes)),
            oversampling_factor,
        })
    }

    pub fn compute(&mut self) -> Result<Vec<Node>, Box<dyn Error>> {
        // The 64 comes from the @workgroup_size(64) inside the shaders
        let num_dispatches = self.fdm_uniform.node_count.div_ceil(64) as u32;
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
            // Start by applying an external force
            for _ in 0..1000 {
                compute_pass.set_push_constants(0, bytemuck::bytes_of(&self.push_constants));
                compute_pass.set_pipeline(&self.external_pipeline);
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
                compute_pass.set_pipeline(&self.moments_pipeline);
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
                compute_pass.set_pipeline(&self.positions_pipeline);
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);

                // Update push constants for next loop
                (
                    self.push_constants.current_index,
                    self.push_constants.future_index,
                ) = (
                    self.push_constants.future_index,
                    self.push_constants.current_index,
                );
            }

            println!("System is now in initial state!");

            for _ in 0..self.oversampling_factor {
                // Compute once
                compute_pass.set_push_constants(0, bytemuck::bytes_of(&self.push_constants));
                compute_pass.set_pipeline(&self.moments_pipeline);
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
                compute_pass.set_pipeline(&self.positions_pipeline);
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);

                // Update push constants for next loop
                (
                    self.push_constants.current_index,
                    self.push_constants.future_index,
                ) = (
                    self.push_constants.future_index,
                    self.push_constants.current_index,
                );
            }
            compute_pass.set_pipeline(&self.output_pipeline);
            compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
        }

        self.queue.submit(iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Begin memory transfer to CPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.buffer_size as u64,
        );
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Awaits until `buffer_future` can be read from
        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<Node> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.staging_buffer.unmap(); // Unmaps buffer from memory

            return Ok(result);
        } else {
            eprintln!("Failed to map staging buffer!");
        }

        Ok(Vec::new())
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
