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
            nodes: Vec<[[f32; 3]; 4]>,
            oversampling_factor: usize,
            dt: f32,
            two_ds_inv: f32,
            loss: f32,
            tau: f32,
            kappa: f32,
            m_inv: f32,
            dxf_t: f32,
            beta: [f32; 3],
            dx2_inv: f32,
            sigma: [f32; 3],
            chunk_size: u32,
            muk2_inv: [f32; 3],
        ) -> PyResult<Self> {
            let nodes: Vec<gpu_bindings::Node> = nodes
                .iter()
                .map(|n| gpu_bindings::Node::new(n[0], n[1], n[2], n[3]))
                .collect();

            let uniforms = gpu_bindings::FDMUniform::new(
                dt, two_ds_inv, &nodes, loss, tau, kappa, m_inv, dxf_t, beta, dx2_inv, sigma,
                chunk_size, muk2_inv,
            );
            let state = pollster::block_on(gpu_bindings::State::new(
                nodes,
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

        fn save(&mut self) -> PyResult<Vec<Vec<[[f32; 3]; 8]>>> {
            let frames = self
                .state
                .save()
                .map_err(|_| PyRuntimeError::new_err("error running GPU computation"))?;

            Ok(frames
                .into_iter()
                .map(|nodes| {
                    nodes
                        .into_iter()
                        .map(gpu_bindings::Node::to_raw)
                        .collect::<Vec<[[f32; 3]; 8]>>()
                })
                .collect())
        }

        fn initialize(&mut self, force: f32, steps: usize) -> PyResult<Vec<[[f32; 3]; 8]>> {
            println!("Calling GPU binding");
            let result = self.state.initialize(force, steps).unwrap();
            let initial: Vec<[[f32; 3]; 8]> =
                result.into_iter().map(gpu_bindings::Node::to_raw).collect();

            Ok(initial)
        }
    }
}
