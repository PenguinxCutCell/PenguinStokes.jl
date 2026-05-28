struct FreeSurfaceStokesOptions{T}
    max_iter::Int
    tol::T
    reltol::T
    damping::T
    scheme::Symbol
    update_mode::Symbol
    drive_phase::Int
    log_diagnostics::Bool
end

function FreeSurfaceStokesOptions(;
    max_iter::Integer=8,
    tol::Real=1e-10,
    reltol::Real=1e-8,
    damping::Real=0.8,
    scheme::Symbol=:CN,
    update_mode::Symbol=:diag_newton,
    drive_phase::Integer=1,
    log_diagnostics::Bool=true,
)
    T = promote_type(typeof(float(tol)), typeof(float(reltol)), typeof(float(damping)))
    drive_phase in (1, 2) || throw(ArgumentError("drive_phase must be 1 or 2"))
    return FreeSurfaceStokesOptions{T}(
        Int(max_iter),
        convert(T, tol),
        convert(T, reltol),
        convert(T, damping),
        scheme,
        update_mode,
        Int(drive_phase),
        log_diagnostics,
    )
end

struct FreeSurfaceStokesProblem{N,T,M,IR}
    model::M
    rep::IR
    options::FreeSurfaceStokesOptions{T}
end

function FreeSurfaceStokesProblem(
    model::M,
    rep::IR;
    options::FreeSurfaceStokesOptions=FreeSurfaceStokesOptions(),
) where {N,T,M<:Union{MovingStokesModelMono{N,T},MovingStokesModelTwoPhase{N,T}},IR<:GlobalHFRep{N,T}}
    model.body === rep.body ||
        throw(ArgumentError("construct the moving model with body=rep.body so the free-surface step can update geometry in-place"))
    opts = FreeSurfaceStokesOptions{T}(
        options.max_iter,
        convert(T, options.tol),
        convert(T, options.reltol),
        convert(T, options.damping),
        options.scheme,
        options.update_mode,
        options.drive_phase,
        options.log_diagnostics,
    )
    return FreeSurfaceStokesProblem{N,T,M,IR}(model, rep, opts)
end

function _gcl_step_terms(
    model::MovingStokesModelMono{N,T},
    x_new::AbstractVector,
    Vn,
    A::SparseMatrixCSC{T,Int},
    drive_phase::Int,
) where {N,T}
    drive_phase == 1 || throw(ArgumentError("monophasic free-surface coupling only supports drive_phase=1"))
    terms = stokes_gcl_terms_mono(model, x_new, Vn)
    R = mask_inactive_pressure_cells(model, terms.R_gcl, A)
    return terms, R, terms.R_gcl, terms.R_kin, terms.R_div
end

function _gcl_step_terms(
    model::MovingStokesModelTwoPhase{N,T},
    x_new::AbstractVector,
    Vn,
    A::SparseMatrixCSC{T,Int},
    drive_phase::Int,
) where {N,T}
    terms = stokes_gcl_terms_diph(model, x_new, Vn[1], Vn[2])
    phase_terms = drive_phase == 1 ? terms.phase1 : drive_phase == 2 ? terms.phase2 :
        throw(ArgumentError("drive_phase must be 1 or 2"))
    R = mask_inactive_pressure_cells(model, phase_terms.R_gcl, A; phase=drive_phase)
    return terms, R, phase_terms.R_gcl, phase_terms.R_kin, phase_terms.R_div
end

_start_pressure_volume(model::MovingStokesModelMono{N,T}, t::T) where {N,T} =
    _pressure_volume_at_time(model, t)

_start_pressure_volume(model::MovingStokesModelTwoPhase{N,T}, t::T) where {N,T} =
    _pressure_volumes_at_time(model, t)

"""
    step_free_surface_stokes!(prob, x_prev; t, dt, method=:direct, kwargs...)

Advance a 2D graph free boundary by a residual-based GCL correction. The update
uses the conservative pressure-cell residual `R_gcl = ΔV + Fω`.
"""
function step_free_surface_stokes!(
    prob::FreeSurfaceStokesProblem{N,T},
    x_prev::AbstractVector;
    t::T=zero(T),
    dt::T,
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    model = prob.model
    rep = prob.rep
    opts = prob.options
    opts.update_mode === :diag_newton || throw(ArgumentError("only update_mode=:diag_newton is implemented"))

    Vn = _start_pressure_volume(model, t)
    xf_guess = predict_xf(rep, dt)
    jac = _column_jacobian(rep)
    history = StokesGHFCouplingHistory(T)
    x_new = similar(Vector{T}(x_prev), nunknowns(model.layout))
    sys_last = nothing

    atol_col = opts.tol
    rtol_col = opts.reltol

    for k in 1:opts.max_iter
        _set_slab_graph!(rep, xf_guess, t, dt)
        sys = solve_unsteady_moving!(model, x_prev; t=t, dt=dt, scheme=opts.scheme, method=method, kwargs...)
        sys_last = sys
        x_new = Vector{T}(sys.x)

        terms, R, R_gcl, R_kin, R_div = _gcl_step_terms(model, x_new, Vn, sys.A, opts.drive_phase)
        R_col = _column_sum_profile(R, model.gridp, rep.axis)
        step = opts.damping .* R_col ./ jac
        step_norm = norm(step, Inf)

        gcl_norm = norm(R_gcl, Inf)
        kin_norm = norm(R_kin, Inf)
        div_norm = norm(R_div, Inf)
        _push_history!(
            history;
            iter=k,
            gcl=gcl_norm,
            kin=kin_norm,
            div=div_norm,
            step=step_norm,
            damping=opts.damping,
        )

        threshold = max(atol_col, rtol_col * max(norm(xf_guess, Inf), one(T)))
        if norm(R_col, Inf) <= threshold || step_norm <= threshold
            commit!(rep, xf_guess, dt)
            _set_slab_graph!(rep, rep.xf, t + dt, zero(T))
            return (; x=x_new, sys=sys, history=history, terms=terms, converged=true)
        end

        xf_guess .-= step
    end

    commit!(rep, xf_guess, dt)
    _set_slab_graph!(rep, rep.xf, t + dt, zero(T))
    return (; x=x_new, sys=sys_last, history=history, terms=nothing, converged=false)
end
