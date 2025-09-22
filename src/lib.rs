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
        fn new(node_count: usize, chunk_size: usize) -> PyResult<Self> {
            // these must unfortunately be heap allocated since the node_count is not known at compile time
            let mut nodes = Vec::with_capacity(node_count);
            for _ in 0..node_count {
                nodes.push(gpu_bindings::Node::new(
                    (0.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0),
                ));
            }

            let state = pollster::block_on(gpu_bindings::State::new(nodes, chunk_size));

            if let Ok(state) = state {
                Ok(Self { state })
            } else {
                Err(PyValueError::new_err(
                    "unable to initialize GPU for simulation",
                ))
            }
        }

        fn compute(&mut self) -> PyResult<Vec<[f32; 2]>> {
            let result = self.state.compute();
            result.map_err(|_| PyRuntimeError::new_err("error running GPU computation"))
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
    }
}
