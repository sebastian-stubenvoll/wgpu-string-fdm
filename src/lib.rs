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
            nodes: Vec<[[f32; 3]; 2]>,
            edges: Vec<([f32; 4], [f32; 3])>,
            oversampling_factor: usize,
            chunk_size: u32,
            dt: f32,
            dx: f32,
            mass: f32,
            stiffness_se: [f32; 3],
            stiffness_bt: [f32; 3],
            inertia: [f32; 3],
        ) -> PyResult<Self> {
            let nodes: Vec<gpu_bindings::Node> = nodes
                .iter()
                .map(|n| gpu_bindings::Node::new(&n[0], &n[1]))
                .collect();

            let edges: Vec<gpu_bindings::Edge> = edges
                .iter()
                .map(|e| gpu_bindings::Edge::new(&e.0, &e.1))
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
            );

            let state = pollster::block_on(gpu_bindings::State::new(
                nodes,
                edges,
                uniforms,
                oversampling_factor,
            ));

            if let Ok(state) = state {
                Ok(Self { state })
            } else {
                Err(PyValueError::new_err(
                    "unable to initialize GPU for simulation",
                ))
            }
        }

        fn compute(&mut self) {
            self.state.compute().unwrap();
        }

        fn save(&mut self) -> PyResult<Vec<Vec<[[f32; 3]; 3]>>> {
            let (node_frames, edge_frames) = self
                .state
                .save()
                .map_err(|_| PyRuntimeError::new_err("error running GPU computation"))?;

            let n = Ok(node_frames
                .into_iter()
                .map(|nodes| {
                    nodes
                        .into_iter()
                        .map(gpu_bindings::Node::to_raw)
                        .collect::<Vec<[[f32; 3]; 3]>>()
                })
                .collect());

            n
        }

        fn initialize(&mut self, force: f32, steps: usize) -> PyResult<()> {
            _ = self.state.initialize(0.0, 1);
            Ok(())
        }
    }
}
