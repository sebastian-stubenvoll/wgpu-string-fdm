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

        /// Kelvin-Voigt damping of the twist rate D3 (N m^2 s); 0 (the default) disables it.
        fn set_twist_rate_damping(&mut self, twist_rate_damping: f32) -> PyResult<()> {
            _ = self.state.set_twist_rate_damping(twist_rate_damping);
            Ok(())
        }

        /// Bow contact (see the shader). model 1: friction curve; model 2: thermal model (call
        /// set_bow_thermal first; the friction curve is then the steady-sliding calibration). Model 1:
        /// mu(v) = mu[0] + mu[1] exp(-v / mu[2]) + mu[3] exp(-v / mu[4]). The profile of normal
        /// force and bow velocity rises linearly over ramp_steps solver steps from start_step and
        /// falls from stop_step. Resets the bow clock and contact states.
        fn set_bow(
            &mut self,
            model: u32,
            torsion: bool,
            nodes: Vec<u32>,
            weights: Vec<f32>,
            radius: f32,
            normal_force: f32,
            velocity: f32,
            mu: [f32; 5],
            start_step: u32,
            ramp_steps: u32,
            stop_step: u32,
        ) -> PyResult<()> {
            let bow = gpu_bindings::Bow::new(
                model, torsion, &nodes, &weights, radius, normal_force, velocity, &mu,
                start_step, ramp_steps, stop_step,
            );
            self.state
                .set_bow(bow)
                .map_err(|e| PyRuntimeError::new_err(format!("set_bow: {e}")))
        }

        /// Thermal friction model parameters (batches/bowing.thermal_contact), used by the next
        /// set_bow with model 2.
        fn set_bow_thermal(
            &mut self,
            c_over_h: f32,
            conv_coef: f32,
            lam_string: f32,
            lam_bow: f32,
            window: u32,
            f_window: f32,
            exact: Vec<f32>,
            decay: Vec<f32>,
            weight: Vec<f32>,
            weight_window: Vec<f32>,
            calibration: [f32; 3],
        ) -> PyResult<()> {
            let t = gpu_bindings::BowThermal::new(
                c_over_h, conv_coef, lam_string, lam_bow, window, f_window, &exact, &decay, &weight,
                &weight_window, calibration,
            );
            self.state
                .set_bow_thermal(t)
                .map_err(|e| PyRuntimeError::new_err(format!("set_bow_thermal: {e}")))
        }

        /// Rigid plectrum at `node` (see the shader's struct Pick): a capsule of half-thickness
        /// tip_radius along the unit vector axis = (y, z) from the tip centre tip = (y, z), face
        /// length `length`; moves along +y from start_step with a linear speed ramp over ramp_steps
        /// to `velocity`. Coulomb friction mu_s >= mu_d; torsion False inhibits the edge rotation in
        /// the contact. Resets the pick clock and record.
        fn set_pick(
            &mut self,
            node: u32,
            torsion: bool,
            radius: f32,
            tip_radius: f32,
            length: f32,
            axis: [f32; 2],
            tip: [f32; 2],
            velocity: f32,
            start_step: u32,
            ramp_steps: u32,
            mu_s: f32,
            mu_d: f32,
        ) -> PyResult<()> {
            let pick = gpu_bindings::Pick::new(
                node, torsion, radius, tip_radius, length, axis, tip, velocity, start_step,
                ramp_steps, mu_s, mu_d,
            );
            self.state
                .set_pick(pick)
                .map_err(|e| PyRuntimeError::new_err(format!("set_pick: {e}")))
        }

        /// Hammer contact geometry and felt friction: crown_radius > 0 is a crown curved across
        /// the string (its centre hammer_offset_y beside the string), 0 a flat crown; Coulomb felt
        /// friction mu_s >= mu_d on `nodes` (empty or mu_s = 0: none).
        fn set_hammer_friction(
            &mut self,
            nodes: Vec<u32>,
            torsion: bool,
            radius: f32,
            crown_radius: f32,
            mu_s: f32,
            mu_d: f32,
        ) -> PyResult<()> {
            let h = gpu_bindings::HammerFriction::new(&nodes, torsion, radius, crown_radius, mu_s, mu_d);
            self.state
                .set_hammer_friction(h)
                .map_err(|e| PyRuntimeError::new_err(format!("set_hammer_friction: {e}")))
        }

        /// Pick and hammer records of the last chunk, per frame: (pick, hammer). pick: (step,
        /// first contact step, last contact step, max penetration, tip y, contact); hammer:
        /// (hammer displacement, hammer velocity, friction work on the hammer, [contact per friction
        /// node]). contact: (in contact, sticking, [slip velocity, normal force, friction force,
        /// normal y, normal z, body work, friction work, normal work, friction impulse]). Works and
        /// the impulse are sums over the frame, the rest is the frame's last step.
        fn save_surface(
            &mut self,
        ) -> PyResult<
            Vec<(
                (u32, u32, u32, f32, f32, (u32, u32, [f32; 9])),
                (f32, f32, f32, Vec<(u32, u32, [f32; 9])>),
            )>,
        > {
            let count = self.state.hammer_friction_count();
            let frames = self
                .state
                .save_surface()
                .map_err(|_| PyRuntimeError::new_err("error reading the surface output"))?;
            let contact = |c: &gpu_bindings::SurfaceContact| {
                (
                    c.in_contact,
                    c.sticking,
                    [
                        c.slip_velocity,
                        c.normal_force,
                        c.friction_force,
                        c.normal_y,
                        c.normal_z,
                        c.body_work,
                        c.friction_work,
                        c.normal_work,
                        c.impulse_t,
                    ],
                )
            };
            Ok(frames
                .iter()
                .map(|f| {
                    let p = &f.pick;
                    let h = &f.hammer;
                    (
                        (
                            p.step,
                            p.first_contact_step,
                            p.last_contact_step,
                            p.max_penetration,
                            p.tip_y,
                            contact(&p.contact),
                        ),
                        (
                            h.hammer_displacement,
                            h.hammer_velocity,
                            h.hammer_work,
                            h.contacts[..count].iter().map(contact).collect(),
                        ),
                    )
                })
                .collect())
        }

        /// Bow output of the last chunk, per frame: (normal force, bow velocity, solver step,
        /// per contact (sticking, [slip velocity, force, impulse, bow work, friction work,
        /// normal-force work, contact temperature, heat power])). Temperature: the step's contact
        /// temperature (computed before its friction); heat power: that step's frictional heat.
        /// Sums (impulse, works) are over the frame; the rest is the frame's last step.
        fn save_bow(&mut self) -> PyResult<Vec<(f32, f32, u32, Vec<(u32, [f32; 8])>)>> {
            let count = self.state.bow_contact_count();
            let frames = self
                .state
                .save_bow()
                .map_err(|_| PyRuntimeError::new_err("error reading the bow output"))?;
            Ok(frames
                .iter()
                .map(|f| {
                    let contacts = f.contacts[..count]
                        .iter()
                        .map(|c| {
                            (
                                c.sticking,
                                [
                                    c.slip_velocity,
                                    c.force,
                                    c.impulse,
                                    c.bow_work,
                                    c.friction_work,
                                    c.normal_work,
                                    c.temperature,
                                    c.heat_power,
                                ],
                            )
                        })
                        .collect();
                    (f.normal_force, f.velocity, f.step, contacts)
                })
                .collect())
        }
    }
}
