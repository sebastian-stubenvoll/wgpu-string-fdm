#![allow(dead_code)]
use std::{error::Error, fmt::Display, iter};

use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
pub struct Node {
    pub positions: [f32; 4],
    pub velocities: [f32; 4],
}

impl Node {
    pub fn new(positions: [f32; 4], velocities: [f32; 4]) -> Self {
        Self {
            positions,
            velocities,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
struct FDMUniform {
    //Alignment is 16 bytes for Uniform buffers :)
    output_node: u32,
    node_count: u32,
    chunk_size: u32,
    dt: f32,

    dx: f32,
    // j - o  are arbitrary simluation parameters
    // This is kept generic so the compute pipeline can be reused for different simulations by only exchanging shaders
    j: f32,
    k: f32,
    l: f32,

    m: f32,
    n: f32,
    o: f32,
    p: f32,
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
    compute_pipeline: wgpu::ComputePipeline,
    device: wgpu::Device,
    fdm_uniform: FDMUniform,
    fdm_uniform_buffer: wgpu::Buffer,
    nodes: Box<[Node]>,
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
        chunk_size: usize,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
        });

        //Uniform Buffers
        let fdm_uniform = FDMUniform {
            output_node: nodes_vec.len().div_euclid(2) as u32,
            node_count: nodes_vec.len() as u32,
            chunk_size: chunk_size as u32,
            dt: 0.000022676,

            dx: 0.0068125,
            j: 0.1,
            k: 0.0001,
            l: 0.0001,

            m: 0.0,
            n: 0.0,
            o: 0.0,
            p: 0.0,
        };
        let fdm_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FDM Uniform Buffer"),
            contents: bytemuck::cast_slice(&[fdm_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let nodes = nodes_vec.into_boxed_slice();

        let nodes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("node buffer"),
            contents: bytemuck::cast_slice(nodes.as_ref()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Buffer whose contents will eventually be copied into the CPU staging buffer
        // We store this in GPU memory for CHUNK_SIZE passes, to minimize the calls to the
        // (expensive) buffer copy operation.
        let mut output_vec = Vec::with_capacity(chunk_size);
        for _ in 0..chunk_size {
            output_vec.push([0.0f32; 2]);
        }

        let output = output_vec.into_boxed_slice();

        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index buffer"),
            contents: bytemuck::cast_slice(output.as_ref()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Staging
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of_val(&*(output)) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let push_constants = PushConstants {
            current_index: 0,
            // (u_current, v_current, u_future, v_future)
            future_index: 2,
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

        //Uses the compute node buffer as storage buffer
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "compute_main",
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            fdm_uniform,
            fdm_uniform_buffer,
            compute_bind_group,
            nodes,
            nodes_buffer,
            output_buffer,
            staging_buffer,
            push_constants,
            compute_pipeline,
            buffer_size: std::mem::size_of_val(&(*output)),
            oversampling_factor,
        })
    }

    pub fn compute(&mut self) -> Result<Vec<[f32; 2]>, Box<dyn Error>> {
        // The 64 comes from the @workgroup_size(64) inside the shaders
        let num_dispatches = (*self.nodes).len().div_ceil(64) as u32;
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
            compute_pass.set_pipeline(&self.compute_pipeline);
            for i in 0..self.fdm_uniform.chunk_size as usize * self.oversampling_factor {
                // Toggle save
                // self.push_constants.save = (i % self.oversampling_factor) as u32;

                // Compute once
                compute_pass.set_push_constants(0, bytemuck::bytes_of(&self.push_constants));
                compute_pass.dispatch_workgroups(num_dispatches, 1, 1);

                // Update push constants for next loop
                (
                    self.push_constants.current_index,
                    self.push_constants.future_index,
                ) = (
                    self.push_constants.future_index,
                    self.push_constants.current_index,
                );

                // If previous output was saved, move saveloc pointer
                // This ordering allows the shader to just write to the output array on each pass
                // While seemingly wasteful, this avoids a branch
                if i % self.oversampling_factor == self.oversampling_factor - 1 {
                    self.push_constants.output_index =
                        (self.push_constants.output_index + 1) % self.fdm_uniform.chunk_size;
                }
            }
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
            let result: Vec<[f32; 2]> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.staging_buffer.unmap(); // Unmaps buffer from memory

            return Ok(result);
        } else {
            eprintln!("Failed to map staging buffer!");
        }
        Ok(Vec::new())
    }

    pub fn set_dx(&mut self, dx: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.dx = dx;
        self.update_uniforms()
    }

    pub fn set_dt(&mut self, dt: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.dt = dt;
        self.update_uniforms()
    }

    pub fn set_j(&mut self, val: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.j = val;
        self.update_uniforms()
    }

    pub fn set_k(&mut self, val: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.k = val;
        self.update_uniforms()
    }

    pub fn set_l(&mut self, val: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.l = val;
        self.update_uniforms()
    }

    pub fn set_m(&mut self, val: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.m = val;
        self.update_uniforms()
    }

    pub fn set_n(&mut self, val: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.n = val;
        self.update_uniforms()
    }

    pub fn set_o(&mut self, val: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.o = val;
        self.update_uniforms()
    }

    pub fn set_p(&mut self, val: f32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.p = val;
        self.update_uniforms()
    }

    pub fn set_output_node(&mut self, id: u32) -> Result<(), Box<dyn Error>> {
        self.fdm_uniform.output_node = id;
        self.update_uniforms()
    }

    fn update_uniforms(&mut self) -> Result<(), Box<dyn Error>> {
        let mut buffer = encase::UniformBuffer::new(Vec::new());
        buffer.write(&self.fdm_uniform)?;
        self.queue
            .write_buffer(&self.fdm_uniform_buffer, 0, &buffer.into_inner());
        Ok(())
    }
}
