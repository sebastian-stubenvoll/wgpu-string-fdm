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
    shell_moments: [f32; 3],
    _padding4: u32,
    shell_forces: [f32; 3],
    _padding5: u32,
    helix_forces: [f32; 3],
    _padding6: u32,
    core_forces: [f32; 3],
    _padding7: u32,
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
            shell_forces: [0.0; 3],
            shell_moments: [0.0; 3],
            helix_forces: [0.0; 3],
            core_forces: [0.0; 3],
            _padding0: 0,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
            _padding4: 0,
            _padding5: 0,
            _padding6: 0,
            _padding7: 0,
        }
    }

    pub fn to_raw(self) -> [[f32; 3]; 8] {
        [
            self.positions,
            self.velocities,
            self.angles,
            self.angular_velocites,
            self.shell_forces,
            self.shell_moments,
            self.helix_forces,
            self.core_forces,
        ]
    }
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
pub struct FDMUniform {
    dt: f32,
    two_ds_inv: f32,
    node_count: u32,
    loss: f32,

    tau: f32,
    kappa: f32,
    m_inv: f32,
    c2_core: f32,

    beta: [f32; 3],
    dx2_inv: f32,

    sigma: [f32; 3],
    chunk_size: u32,

    k_inv: [f32; 3],
    _padding5: u32,
}

impl FDMUniform {
    pub fn new(
        dt: f32,
        two_ds_inv: f32,
        nodes: &[Node],
        loss: f32,
        tau: f32,
        kappa: f32,
        m_inv: f32,
        c2_core: f32,
        beta: [f32; 3],
        dx2_inv: f32,
        sigma: [f32; 3],
        chunk_size: u32,
        k_inv: [f32; 3],
    ) -> Self {
        assert!(two_ds_inv != 0.0);
        assert!(m_inv != 0.0);
        assert!(c2_core != 0.0);
        assert!(dx2_inv != 0.0);
        assert!(k_inv.iter().all(|v| *v != 0.0));
        assert!(beta.iter().all(|v| *v != 0.0));
        assert!(sigma.iter().all(|v| *v != 0.0));

        Self {
            dt,
            two_ds_inv,
            node_count: nodes.len() as u32,
            loss,
            tau,
            kappa,
            m_inv,
            c2_core,
            beta,
            dx2_inv,
            chunk_size,
            sigma,
            k_inv,
            ..Default::default()
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    current_index: u32,
    future_index: u32,
    vel: f32,
    output_idx: u32,
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
    buffer_size: u64,
    compute_bind_group: wgpu::BindGroup,
    external_force_pipeline: wgpu::ComputePipeline,
    forces_pipeline: wgpu::ComputePipeline,
    displacements_pipeline: wgpu::ComputePipeline,
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
    initialized: bool,
}

impl State {
    pub async fn new(
        mut nodes_vec: Vec<Node>,
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

        nodes_vec.extend_from_within(..);
        let nodes_in = nodes_vec.into_boxed_slice();
        let buffer_size = (std::mem::size_of::<Node>() as u64
            * nodes_in.len() as u64
            * uniforms.chunk_size as u64) as wgpu::BufferAddress;

        let nodes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("node buffer"),
            contents: bytemuck::cast_slice(nodes_in.as_ref()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let push_constants = PushConstants {
            current_index: 0,
            future_index: 1,
            vel: 1.0e-3,
            output_idx: 0,
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

        let external_force_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "external_force",
                compilation_options: Default::default(),
            });

        let forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "calculate_internal_forces",
            compilation_options: Default::default(),
        });

        let displacements_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "calculate_displacements",
                compilation_options: Default::default(),
            });

        let output_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "save_output",
            compilation_options: Default::default(),
        });

        dbg!(&fdm_uniform);

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
            external_force_pipeline,
            forces_pipeline,
            displacements_pipeline,
            output_pipeline,
            buffer_size,
            oversampling_factor,
            initialized: false,
        })
    }

    pub fn initialize(&mut self, steps: usize) -> Result<(), Box<dyn Error>> {
        println!("Calling init!");
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
                compute_pass.set_pipeline(&self.external_force_pipeline);
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
            }
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                compute_pass.set_pipeline(&self.forces_pipeline);
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
            }
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                compute_pass.set_pipeline(&self.displacements_pipeline);
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
            }
            (
                self.push_constants.current_index,
                self.push_constants.future_index,
            ) = (
                self.push_constants.future_index,
                self.push_constants.current_index,
            );
            self.queue.submit(iter::once(encoder.finish()));
        }
        self.device.poll(wgpu::Maintain::Wait);

        Ok(())
    }

    pub fn compute(&mut self) -> Result<(), Box<dyn Error>> {
        // The 64 comes from the @workgroup_size(64) inside the shaders
        let num_dispatches = self.fdm_uniform.node_count.div_ceil(64);

        // Too many commands break command buffers.
        assert!(self.oversampling_factor <= 256);
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
                    compute_pass.set_pipeline(&self.forces_pipeline);
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
                    compute_pass.set_pipeline(&self.displacements_pipeline);
                    compute_pass
                        .set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                    compute_pass.dispatch_workgroups(num_dispatches, 1, 1);
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
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                compute_pass.set_pipeline(&self.output_pipeline);
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[self.push_constants]));
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);

                self.push_constants.output_idx =
                    (self.push_constants.output_idx + 1) % self.fdm_uniform.chunk_size;
            }
            self.queue.submit(iter::once(encoder.finish()));
        }
        Ok(())
    }

    pub fn save(&mut self) -> Result<Vec<Vec<Node>>, Box<dyn Error>> {
        assert_eq!(self.push_constants.output_idx, 0);
        // Begin memory transfer to CPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.buffer_size,
        );
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        println!("Transferring buffer with size {}", self.buffer_size);

        // Awaits until `buffer_future` can be read from
        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<Node> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.staging_buffer.unmap(); // Unmaps buffer from memory

            let vecs: Vec<Vec<Node>> = result
                .chunks(self.fdm_uniform.node_count as usize)
                .map(|slice| slice.to_vec())
                .collect();
            return Ok(vecs);
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
