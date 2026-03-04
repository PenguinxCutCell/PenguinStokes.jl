**Examples and tests**

The repository includes runnable scripts in `examples/`.

Run from repository root:

```bash
julia --project=. examples/04_mms_convergence.jl
julia --project=. examples/07_unsteady_sphere_drag.jl
```

General test command:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
