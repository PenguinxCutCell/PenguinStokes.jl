"""
    31_periodic_cylinder_array.jl

Stokes flow past a periodic array of cylinders, comparing numerical results with 
the multipole expansion solution of Sangani and Acrivos (1982).

The domain is a periodic unit square with a single cylinder at the center. 
Due to periodicity, this represents an infinite array of cylinders.

This example demonstrates the setup for validating PenguinStokes against the
Sangani & Acrivos analytical solution using the same non-dimensionalization
as the Basilisk reference implementation.

Reference:
  Sangani, A.S. and Acrivos, A. (1982). Slow flow past periodic arrays of cylinders 
  with application to heat transfer. International Journal of Multiphase Flow, 8(3):193-206.

Table 1 from Sangani & Acrivos (1982):
  Volume fraction Φ | Non-dimensional drag F/(μU)
  0.05             | 15.56
  0.10             | 24.83
  0.20             | 51.53
  0.30             | 102.90
  0.40             | 217.89
  0.50             | 532.55
  0.60             | 1763.0
  0.70             | 13520.0
  0.75             | 126300.0
"""

using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes
using PenguinSolverCore: LinearSystem
using Printf

# Sangani & Acrivos reference data (Table 1)
sangani_ref = [
    (0.05,  15.56),
    (0.10,  24.83),
    (0.20,  51.53),
    (0.30,  102.90),
    (0.40,  217.89),
    (0.50,  532.55),
    (0.60,  1763.0),
    (0.70,  13520.0),
    (0.75,  126300.0),
]

function periodic_2d_bc()
    """Create periodic boundary conditions for all sides."""
    return (
        BorderConditions(; left=Periodic(), right=Periodic(), 
                         bottom=Periodic(), top=Periodic()),
        BorderConditions(; left=Periodic(), right=Periodic(), 
                         bottom=Periodic(), top=Periodic()),
    )
end

function compute_drag_force(model::StokesModelMono{2,T}, sys::LinearSystem{T}) where {T}
    """
    Compute drag and lift from integrated embedded-boundary traction.

    Returns a named tuple with:
    - drag: x-component of force on the body (positive opposes flow)
    - lift: y-component of force on the body
    - force: full force vector
    """
    force_dict = integrated_embedded_force(model, sys)
    force_vec = force_dict.force
    return (drag=-force_vec[1], lift=-force_vec[2], force=force_vec)
end

function compute_average_velocity(model::StokesModelMono{2,T}, sys::LinearSystem{T}) where {T}
    """
    Compute the spatial average velocity in the domain.
    U_avg = ∫ u dV / ∫ dV  (averaged over fluid volume)
    """
    u1 = sys.x[model.layout.uomega[1]]
    
    total_flux = zero(T)
    total_vol = zero(T)
    
    for i in 1:model.cap_u[1].ntotal
        V = model.cap_u[1].buf.V[i]
        if isfinite(V) && V > 0
            total_flux += V * u1[i]
            total_vol += V
        end
    end
    
    return total_vol > 0 ? total_flux / total_vol : one(T)
end

function compute_volume_fraction(model::StokesModelMono{2,T}) where {T}
    """
    Compute the solid volume fraction from the pressure-grid capacities.
    """
    cap_p = model.cap_p
    fluid_vol = zero(T)
    for i in 1:cap_p.ntotal
        V = cap_p.buf.V[i]
        if isfinite(V) && V > 0
            fluid_vol += V
        end
    end

    Lx = model.gridp.hc[1] - model.gridp.lc[1]
    Ly = model.gridp.hc[2] - model.gridp.lc[2]
    domain_vol = Lx * Ly
    phi = one(T) - fluid_vol / domain_vol
    return (phi=phi, fluid_vol=fluid_vol, domain_vol=domain_vol)
end

function solve_cylinder_case(n::Int, phi::T, f_magnitude::T=1.0) where {T}
    """
    Solve steady Stokes flow for a single cylinder in periodic domain.
    
    Args:
        n: Number of grid points per side
        phi: Volume fraction of cylinder (0 < phi < 1)
        f_magnitude: Magnitude of body force (drives the flow)
    
    Returns:
        Named tuple with (drag_nondim, u_avg, residual, model, sys)
    """
    
    # Set up periodic domain: unit square [0, 1] × [0, 1]
    L = one(T)
    grid = CartesianGrid((zero(T), zero(T)), (L, L), (n, n))
    
    # Cylinder radius from volume fraction: Φ = π*r²/L²  =>  r = sqrt(Φ*L²/π)
    radius = sqrt(phi * L^2 / π)
    
    # Center the cylinder in the domain
    xc = L / 2
    yc = L / 2
    
    # Level-set function: negative in fluid, positive in solid
    # Using r - distance makes the cylinder solid with fluid outside.
    body(x, y) = radius - sqrt((x - xc)^2 + (y - yc)^2)
    
    # Use uniform body force in x-direction to drive the flow
    # This avoids having to specify inlet/outlet boundary conditions
    force = (f_magnitude, zero(T))
    
    # Periodic boundary conditions on all sides
    bc_u = periodic_2d_bc()
    bc_p = BorderConditions(; left=Periodic(), right=Periodic(),
                            bottom=Periodic(), top=Periodic())
    
    # Create model
    mu = one(T)  # Dynamic viscosity
    rho = one(T) # Density (not used in steady Stokes)
    
    model = StokesModelMono(
        grid,
        body,
        mu,
        rho;
        bc_u=bc_u,
        bc_p=bc_p,
        bc_cut=Dirichlet(zero(T)),  # No-slip on cylinder surface
        force=force,
        gauge=MeanPressureGauge(),  # Use mean pressure to avoid singular pressure
    )
    
    # Solve the system
    sys = solve_steady!(model)
    
    # Compute average velocity (used for non-dimensionalization)
    u_avg = compute_average_velocity(model, sys)
    
    # Compute drag and lift from embedded traction (diagnostics)
    drag_info = compute_drag_force(model, sys)
    drag_force = drag_info.drag
    lift_force = drag_info.lift
    
    # Residual norm
    residual = norm(sys.A * sys.x - sys.b)

    vol = compute_volume_fraction(model)

    # Non-dimensional drag: replicate Sangani & Acrivos/Basilisk convention
    # F/(μU) = (L^2 / (1 - Phi)) / (μ * U)
    drag_nondim = (vol.domain_vol / (1 - vol.phi)) / (mu * u_avg + 1e-15)

    return (
        drag_nondim=drag_nondim,
        drag_force=drag_force,
        lift_force=lift_force,
        u_avg=u_avg,
        residual=residual,
        phi=vol.phi,
        fluid_vol=vol.fluid_vol,
        domain_vol=vol.domain_vol,
        model=model,
        sys=sys,
    )
end

function run_convergence_study()
    """
    Run convergence study: solve for multiple volume fractions and mesh refinements,
    comparing with Sangani & Acrivos reference data.
    """
    
    println("\n" * "="^80)
    println("PERIODIC CYLINDER ARRAY - STOKES FLOW")
    println("Sangani & Acrivos (1982) Validation Study")
    println("="^80)
    println()
    println()
    
    mesh_sizes = (17, 25, 33, 49)
    
    println("Reference data (Sangani & Acrivos 1982, Table 1):")
    println("-" * "="^79)
    @printf "%10s %10s %12s %12s %12s\n" "Φ (vol)" "F/(μU) ref" "n=17" "n=33" "n=49"
    println("-" * "="^79)
    
    for (phi_ref, f_ref) in sangani_ref
        @printf "%10.2f %12.2f" phi_ref f_ref
        
        for n in mesh_sizes
            try
                result = solve_cylinder_case(n, phi_ref, 1.0)
                @printf " %12.2f" result.drag_nondim
            catch e
                @printf " %12s" "ERROR"
            end
        end
        println()
    end
    
    println("-" * "="^79)
    println()
    
    # Detailed study for one volume fraction
    println("Detailed convergence study for Φ = 0.30:")
    println("-" * "="^79)
    
    phi_detail = 0.30
    idx_ref = findfirst(p -> abs(p[1] - phi_detail) < 1e-6, sangani_ref)
    f_ref_detail = sangani_ref[idx_ref][2]
    
    println(@sprintf "%10s %10s %12s %12s %12s %12s" "n" "h" "F/(μU)" "Ref" "RelErr" "Residual")
    println("-" * "="^79)
    
    for n in (13, 17, 25, 33, 49, 65)
        try
            result = solve_cylinder_case(n, phi_detail, 1.0)
            h = 1.0 / (n - 1)
            error = abs(result.drag_nondim - f_ref_detail) / f_ref_detail
            
            @printf "%10d %10.4f %12.2f %12.2f %12.2e %12.2e\n" n h result.drag_nondim f_ref_detail error result.residual
        catch e
            println("n=$n: ERROR - $(string(e))")
        end
    end
    
    println("-" * "="^79)
    println()
    println("Summary:")
    println("-" * "="^79)
    println("This example validates PenguinStokes against the analytical multipole")
    println("expansion results of Sangani & Acrivos for Stokes flow around periodic")
    println("cylinder arrays. The domain is a periodic unit square with a single")
    println("cylinder at the center, driven by a uniform body force.")
    println()
    println("The non-dimensional drag force F/(μU) should approach the reference")
    println("values as the mesh is refined. Agreement typically improves with")
    println("increasing mesh resolution.")
    println("="^80)
    
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    run_convergence_study()
end
