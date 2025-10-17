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
            ds: f32,
            loss: f32,
            tau: f32,
            kappa: f32,
            m_coil: f32,
            c2_core: f32,
            beta: [f32; 3],
            sigma: [f32; 3],
            k: [f32; 3],
        ) -> PyResult<Self> {
            let nodes: Vec<gpu_bindings::Node> = nodes
                .iter()
                .map(|n| gpu_bindings::Node::new(n[0], n[1], n[2], n[3]))
                .collect();

            let uniforms = gpu_bindings::FDMUniform::new(
                dt, ds, &nodes, loss, tau, kappa, m_coil, c2_core, beta, sigma, k,
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

        fn compute(&mut self) -> PyResult<Vec<[[f32; 3]; 6]>> {
            let nodes = self
                .state
                .compute()
                .map_err(|_| PyRuntimeError::new_err("error running GPU computation"))?;

            Ok(nodes.into_iter().map(gpu_bindings::Node::to_raw).collect())
        }
    }
}
