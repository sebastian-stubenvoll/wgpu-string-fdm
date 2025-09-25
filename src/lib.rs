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
            nodes: Vec<[[f32; 4]; 2]>,
            chunk_size: usize,
            oversampling_factor: usize,
        ) -> PyResult<Self> {
            let state = pollster::block_on(gpu_bindings::State::new(
                nodes
                    .iter()
                    .map(|n| gpu_bindings::Node::new(n[0], n[1]))
                    .collect(),
                chunk_size,
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

        fn compute(&mut self) -> PyResult<Vec<[f32; 2]>> {
            self.state
                .compute()
                .map_err(|_| PyRuntimeError::new_err("error running GPU computation"))
        }

        fn set_dx(&mut self, dx: f32) -> PyResult<()> {
            self.state
                .set_dx(dx)
                .map_err(|_| PyRuntimeError::new_err("unable to set dx"))
        }

        fn set_dt(&mut self, dt: f32) -> PyResult<()> {
            self.state
                .set_dt(dt)
                .map_err(|_| PyRuntimeError::new_err("unable to set dt"))
        }

        fn set_j(&mut self, val: f32) -> PyResult<()> {
            self.state
                .set_j(val)
                .map_err(|_| PyRuntimeError::new_err("unable to set j"))
        }

        fn set_k(&mut self, val: f32) -> PyResult<()> {
            self.state
                .set_k(val)
                .map_err(|_| PyRuntimeError::new_err("unable to set k"))
        }

        fn set_l(&mut self, val: f32) -> PyResult<()> {
            self.state
                .set_l(val)
                .map_err(|_| PyRuntimeError::new_err("unable to set l"))
        }

        fn set_m(&mut self, val: f32) -> PyResult<()> {
            self.state
                .set_m(val)
                .map_err(|_| PyRuntimeError::new_err("unable to set m"))
        }

        fn set_n(&mut self, val: f32) -> PyResult<()> {
            self.state
                .set_n(val)
                .map_err(|_| PyRuntimeError::new_err("unable to set n"))
        }

        fn set_o(&mut self, val: f32) -> PyResult<()> {
            self.state
                .set_o(val)
                .map_err(|_| PyRuntimeError::new_err("unable to set o"))
        }

        fn set_p(&mut self, val: f32) -> PyResult<()> {
            self.state
                .set_p(val)
                .map_err(|_| PyRuntimeError::new_err("unable to set p"))
        }
        fn set_parameters(
            &mut self,
            j: f32,
            k: f32,
            l: f32,
            m: f32,
            n: f32,
            o: f32,
            p: f32,
        ) -> PyResult<()> {
            self.state
                .set_parameters(j, k, l, m, n, o, p)
                .map_err(|_| PyRuntimeError::new_err("unable to set parameters"))
        }

        fn set_output_node(&mut self, id: u32) -> PyResult<()> {
            self.state
                .set_output_node(id)
                .map_err(|_| PyRuntimeError::new_err("unable to output_node"))
        }
    }
}
