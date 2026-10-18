mod gpu_bindings;

#[pyo3::pymodule]
mod py_wgpu_fdm {
    use crate::gpu_bindings;
    use pyo3::{
        exceptions::{PyRuntimeError, PyValueError},
        prelude::*,
        PyResult,
    };

    #[pyclass]
    struct Simulation {
        state: gpu_bindings::State,
    }

    #[pymethods]
    impl Simulation {
        #[new]
        fn new(
            // (position, velocity, reference orientation, rest relative orientation)
            nodes: Vec<([f32; 3], [f32; 3], [f32; 4], [f32; 4])>,
            // (orientation deviation p, reference vector, angular velocity, reference strain,
            //  reference offset, rest frame columns [d1, d2, d3])
            edges: Vec<([f32; 4], [f32; 3], [f32; 3], [f32; 3], [f32; 3], [[f32; 3]; 3])>,
            hammer_weights: Vec<f32>,
            oversampling_factor: usize,
            chunk_size: u32,
            dt: f32,
            dx: f32,
            mass: f32,
            stiffness_se: [f32; 3],
            stiffness_bt: [f32; 3],
            inertia: [f32; 3],
            clamp_offset: u32,
            linear_damping: f32,
            angular_damping: [f32; 3],
            // 0: legacy, 1: variational
            force_law: u32,
            hammer_displacement: f32,
            hammer_velocity: f32,
            hammer_mass: f32,
            hammer_stiffness: f32,
            hammer_exponent: f32,
            hammer_hysteresis_factor: f32,
            hammer_relaxation_time: f32,
            hammer_offset_y: f32,
        ) -> PyResult<Self> {
            let nodes: Vec<gpu_bindings::Node> = nodes
                .iter()
                .map(|n| gpu_bindings::Node::new(&n.0, &n.1, &n.2, &n.3))
                .collect();

            let edges: Vec<gpu_bindings::Edge> = edges
                .iter()
                .map(|e| gpu_bindings::Edge::new(&e.0, &e.1, &e.2, &e.3, &e.4, &e.5))
                .collect();

            let uniforms = gpu_bindings::FDMUniform::new(
                nodes.as_slice(),
                edges.as_slice(),
                chunk_size,
                dt,
                dx,
                mass,
                &stiffness_se,
                &stiffness_bt,
                &inertia,
                clamp_offset,
                linear_damping,
                &angular_damping,
                force_law,
            );

            let hammer = gpu_bindings::Hammer::new(
                hammer_displacement,
                hammer_velocity,
                hammer_mass,
                hammer_stiffness,
                hammer_exponent,
                hammer_hysteresis_factor,
                hammer_relaxation_time,
            );

            let state = pollster::block_on(gpu_bindings::State::new(
                nodes,
                edges,
                hammer_weights,
                hammer,
                hammer_offset_y,
                uniforms,
                oversampling_factor,
            ));

            match state {
                Ok(state) => Ok(Self { state }),
                Err(e) => Err(PyValueError::new_err(format!(
                    "unable to initialize GPU for simulation: {e}"
                ))),
            }
        }

        fn compute(&mut self) {
            self.state.compute().unwrap();
        }

        /// The original multi-pass solver loop, for checking `compute` against.
        fn compute_reference(&mut self) {
            self.state.compute_reference().unwrap();
        }

        fn save(
            &mut self,
        ) -> PyResult<(Vec<Vec<[[f32; 3]; 4]>>, Vec<Vec<([f32; 4], [[f32; 3]; 3])>>)> {
            let (node_frames, edge_frames) = self
                .state
                .save()
                .map_err(|_| PyRuntimeError::new_err("error running GPU computation"))?;

            let n = node_frames
                .into_iter()
                .map(|nodes| {
                    nodes
                        .into_iter()
                        .map(gpu_bindings::Node::to_raw)
                        .collect::<Vec<[[f32; 3]; 4]>>()
                })
                .collect();

            let e = edge_frames
                .into_iter()
                .map(|edges| {
                    edges
                        .into_iter()
                        .map(gpu_bindings::Edge::to_raw)
                        .collect::<Vec<([f32; 4], [[f32; 3]; 3])>>()
                })
                .collect();

            Ok((n, e))
        }

        fn initialize(&mut self) -> PyResult<()> {
            _ = self.state.initialize();
            Ok(())
        }

        fn create_references(&mut self) -> PyResult<()> {
            _ = self.state.create_references();
            Ok(())
        }

        fn enable_hammer(&mut self, enable: bool) -> PyResult<()> {
            _ = self.state.enable_hammer(enable);
            Ok(())
        }

        fn set_dt(&mut self, dt: f32) -> PyResult<()> {
            _ = self.state.set_dt(dt);
            Ok(())
        }

        fn set_damping(&mut self, linear_damping: f32, angular_damping: [f32; 3]) -> PyResult<()> {
            _ = self.state.set_damping(linear_damping, angular_damping);
            Ok(())
        }
    }
}
