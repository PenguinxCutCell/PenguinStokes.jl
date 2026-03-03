**Numerical algorithms and routines**

This page summarizes high-level routines used in the package.

1. `assemble_steady!(sys, model, t)`

- Builds block sparse system for momentum, interface tie, and continuity equations.
- Inserts velocity box boundary-condition contributions.
- Applies pressure gauge handling to remove nullspace.
- Enforces identity rows for inactive/halo rows to keep the system well-posed.

2. `assemble_unsteady!(sys, model, un, t, dt, scheme)`

- Reuses steady assembly structure at `t + theta*dt`.
- Adds transient mass terms on momentum omega unknowns.
- Adds previous-state contribution to the right-hand side.

3. `solve_steady!` / `solve_unsteady!`

- Build a `LinearSystem`, call assembly, and delegate linear solve to `PenguinSolverCore.solve!`.
- Unsteady solve advances in time and can save step history.
