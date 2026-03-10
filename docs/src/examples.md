**Examples and tests**

The repository includes runnable scripts in `examples/`.

Run from repository root:

```bash
julia --project=. examples/04_mms_convergence.jl
julia --project=. examples/08_two_phase_mms_fixed_interface.jl
julia --project=. examples/09_two_phase_planar_couette.jl
julia --project=. examples/10_two_phase_planar_poiseuille.jl
julia --project=. examples/11_two_phase_oscillatory_couette.jl
julia --project=. examples/12_two_phase_viscous_drop_drag.jl
julia --project=. examples/07_unsteady_sphere_drag.jl
julia --project=. examples/13_unsteady_moving_body_translation.jl
julia --project=. examples/14_unsteady_oscillating_cylinder.jl
julia --project=. examples/15_channel_pressure_outlet_traction.jl
julia --project=. examples/16_channel_poiseuille_pressure_outlet.jl
julia --project=. examples/17_fsi_free_falling_circle.jl
julia --project=. examples/18_fsi_drag_calibration_circle.jl
julia --project=. examples/19_fsi_neutral_buoyancy_decay.jl
```

General test command:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
