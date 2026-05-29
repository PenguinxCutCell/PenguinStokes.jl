struct CoupledFrontTrackingOptions{T}
    max_iter::Int
    tol::T
    reltol::T
    damping::T
    scheme::Symbol
    drive_phase::Int
    step_clip::T
    smooth::Int
    backtrack::T
    max_backtracks::Int
    redistribute_markers::Bool
    velocity_predictor::Bool
    lm_lambda0::T
    lm_lambda_factor::T
    lm_lambda_min::T
    lm_lambda_max::T
    lm_reg_curv::T
    lm_reg_mass::T
    lm_fd_eps::T
    lm_max_reject::Int
end

function CoupledFrontTrackingOptions(;
    max_iter::Integer=6,
    tol::Real=1e-9,
    reltol::Real=1e-7,
    damping::Real=0.7,
    scheme::Symbol=:BE,
    drive_phase::Integer=1,
    step_clip::Real=0.35,
    smooth::Integer=2,
    backtrack::Real=0.5,
    max_backtracks::Integer=4,
    redistribute_markers::Bool=false,
    velocity_predictor::Bool=true,
    lm_lambda0::Real=1e-2,
    lm_lambda_factor::Real=10.0,
    lm_lambda_min::Real=1e-8,
    lm_lambda_max::Real=1e8,
    lm_reg_curv::Real=0.0,
    lm_reg_mass::Real=0.0,
    lm_fd_eps::Real=1e-3,
    lm_max_reject::Integer=4,
)
    drive_phase in (1, 2) || throw(ArgumentError("drive_phase must be 1 or 2"))
    T = promote_type(typeof(float(tol)), typeof(float(reltol)), typeof(float(damping)), typeof(float(step_clip)))
    return CoupledFrontTrackingOptions{T}(
        Int(max_iter),
        convert(T, tol),
        convert(T, reltol),
        convert(T, damping),
        scheme,
        Int(drive_phase),
        convert(T, step_clip),
        Int(smooth),
        convert(T, backtrack),
        Int(max_backtracks),
        redistribute_markers,
        velocity_predictor,
        convert(T, lm_lambda0),
        convert(T, lm_lambda_factor),
        convert(T, lm_lambda_min),
        convert(T, lm_lambda_max),
        convert(T, lm_reg_curv),
        convert(T, lm_reg_mass),
        convert(T, lm_fd_eps),
        Int(lm_max_reject),
    )
end

struct CoupledMovingStokesProblem{T,M,IR}
    model::M
    rep::IR
    options::CoupledFrontTrackingOptions{T}
end

function CoupledMovingStokesProblem(
    model::M,
    rep::IR;
    options::CoupledFrontTrackingOptions=CoupledFrontTrackingOptions(),
) where {T,M<:MovingStokesModelTwoPhase{2,T},IR<:FrontTrackingRep{T}}
    model.body === rep.body ||
        throw(ArgumentError("construct the moving model with body=rep.body so front-tracking can update geometry in-place"))
    opts = CoupledFrontTrackingOptions{T}(
        options.max_iter,
        convert(T, options.tol),
        convert(T, options.reltol),
        convert(T, options.damping),
        options.scheme,
        options.drive_phase,
        convert(T, options.step_clip),
        options.smooth,
        convert(T, options.backtrack),
        options.max_backtracks,
        options.redistribute_markers,
        options.velocity_predictor,
        convert(T, options.lm_lambda0),
        convert(T, options.lm_lambda_factor),
        convert(T, options.lm_lambda_min),
        convert(T, options.lm_lambda_max),
        convert(T, options.lm_reg_curv),
        convert(T, options.lm_reg_mass),
        convert(T, options.lm_fd_eps),
        options.lm_max_reject,
    )
    return CoupledMovingStokesProblem{T,M,IR}(model, rep, opts)
end

function _ft_vertex_lengths(pts::Vector{SVector{2,T}}) where {T}
    n = length(pts)
    out = Vector{T}(undef, n)
    @inbounds for i in 1:n
        im = mod1(i - 1, n)
        ip = mod1(i + 1, n)
        out[i] = (norm(pts[i] - pts[im]) + norm(pts[ip] - pts[i])) / 2
    end
    return out
end

function _ft_nearest_marker(x::SVector{2,T}, pts::Vector{SVector{2,T}}) where {T}
    best_i = 1
    best_d = typemax(T)
    @inbounds for i in eachindex(pts)
        d = sum(abs2, x - pts[i])
        if d < best_d
            best_d = d
            best_i = i
        end
    end
    return best_i
end

function _ft_smooth_periodic!(a::Vector{T}, passes::Int) where {T}
    n = length(a)
    tmp = similar(a)
    for _ in 1:passes
        @inbounds for i in 1:n
            tmp[i] = (a[mod1(i - 1, n)] + 2a[i] + a[mod1(i + 1, n)]) / 4
        end
        copyto!(a, tmp)
    end
    return a
end

function _ft_residual_displacement(
    R::AbstractVector{T},
    grid::CartesianGrid{2,T},
    pts::Vector{SVector{2,T}},
    opts::CoupledFrontTrackingOptions{T},
) where {T}
    accum = zeros(T, length(pts))
    xs = grid1d(grid)
    lin = LinearIndices(grid.n)
    @inbounds for I in CartesianIndices(grid.n)
        r = R[lin[I]]
        iszero(r) && continue
        isfinite(r) || continue
        x = SVector{2,T}(xs[1][I[1]], xs[2][I[2]])
        j = _ft_nearest_marker(x, pts)
        accum[j] += r
    end
    lengths = _ft_vertex_lengths(pts)
    disp = similar(accum)
    @inbounds for i in eachindex(disp)
        disp[i] = accum[i] / max(lengths[i], eps(T))
    end
    _ft_smooth_periodic!(disp, opts.smooth)
    hmin = minimum(meshsize(grid))
    clip = opts.step_clip * hmin
    @inbounds for i in eachindex(disp)
        disp[i] = clamp(disp[i], -clip, clip)
    end
    return disp
end

function _ft_apply_normal_step(
    pts::Vector{SVector{2,T}},
    normals::Vector,
    disp::Vector{T},
    λ::T,
) where {T}
    out = Vector{SVector{2,T}}(undef, length(pts))
    @inbounds for i in eachindex(pts)
        n = SVector{2,T}(normals[i])
        out[i] = pts[i] + λ * disp[i] * n
    end
    return out
end

function _ft_nearest_grid_value(grid::CartesianGrid{2,T}, values::AbstractVector, x::SVector{2,T}) where {T}
    xs = grid1d(grid)
    i = clamp(searchsortedfirst(xs[1], x[1]), firstindex(xs[1]), lastindex(xs[1]))
    j = clamp(searchsortedfirst(xs[2], x[2]), firstindex(xs[2]), lastindex(xs[2]))
    if i > firstindex(xs[1]) && abs(xs[1][i - 1] - x[1]) < abs(xs[1][i] - x[1])
        i -= 1
    end
    if j > firstindex(xs[2]) && abs(xs[2][j - 1] - x[2]) < abs(xs[2][j] - x[2])
        j -= 1
    end
    return convert(T, values[LinearIndices(grid.n)[i, j]])
end

function _ft_velocity_normal_displacement(
    model::MovingStokesModelTwoPhase{2,T},
    x_new::AbstractVector,
    pts::Vector{SVector{2,T}},
    normals::Vector,
    dt::T,
    phase::Int,
) where {T}
    layout_u = phase == 1 ? model.layout.uomega1 : model.layout.uomega2
    disp = zeros(T, length(pts))
    @inbounds for i in eachindex(pts)
        u = SVector{2,T}(
            _ft_nearest_grid_value(model.gridu[1], view(x_new, layout_u[1]), pts[i]),
            _ft_nearest_grid_value(model.gridu[2], view(x_new, layout_u[2]), pts[i]),
        )
        disp[i] = dt * dot(u, SVector{2,T}(normals[i]))
    end
    return disp
end

function _ft_predict_points(rep::FrontTrackingRep{T}, pts0::Vector{SVector{2,T}}, dt::T) where {T}
    normals = rep.state.geom.vertex_normals
    out = Vector{SVector{2,T}}(undef, length(pts0))
    @inbounds for i in eachindex(pts0)
        out[i] = pts0[i] + dt * rep.last_normal_speed[i] * SVector{2,T}(normals[i])
    end
    return out
end

function _ft_solve_and_residual(
    model::MovingStokesModelTwoPhase{2,T},
    rep::FrontTrackingRep{T},
    x_prev::AbstractVector,
    Vn,
    pts0::Vector{SVector{2,T}},
    pts1::Vector{SVector{2,T}},
    t::T,
    dt::T,
    opts::CoupledFrontTrackingOptions{T};
    method::Symbol=:direct,
    kwargs...,
) where {T}
    update_front_levelset!(rep.body, pts0, pts1, t, dt)
    sys = solve_unsteady_moving!(model, x_prev; t=t, dt=dt, scheme=opts.scheme, method=method, kwargs...)
    terms, R, R_gcl, R_kin, R_div = _gcl_step_terms(model, sys.x, Vn, sys.A, opts.drive_phase)
    phase_terms = opts.drive_phase == 1 ? terms.phase1 : terms.phase2
    return (; sys, terms, phase_terms, R, norm=norm(R, Inf),
        gcl=norm(R_gcl, Inf), kin=norm(R_kin, Inf), div=norm(R_div, Inf))
end

"""
    step_coupled_fronttracking!(prob, x_prev; t, dt, method=:direct, kwargs...)

Advance a 2D closed front for a moving two-phase Stokes model using the
conservative pressure-cell GCL residual. Dispatches on `rep.coupling`:

- `:explicit`         — single predictor + one Stokes solve (lagged motion).
- `:ft_redistribute`  — damped per-cell residual redistribution with backtracking.
- `:ft_lm`            — Levenberg–Marquardt with FD volume Jacobian.
"""
function step_coupled_fronttracking!(
    prob::CoupledMovingStokesProblem{T},
    x_prev::AbstractVector;
    t::T=zero(T),
    dt::T,
    method::Symbol=:direct,
    kwargs...,
) where {T}
    rep = prob.rep
    if rep.coupling === :ft_redistribute
        return _step_ft_redistribute!(prob, x_prev; t=t, dt=dt, method=method, kwargs...)
    elseif rep.coupling === :explicit
        return _step_ft_explicit!(prob, x_prev; t=t, dt=dt, method=method, kwargs...)
    elseif rep.coupling === :ft_lm
        return _step_ft_lm!(prob, x_prev; t=t, dt=dt, method=method, kwargs...)
    else
        throw(ArgumentError("unsupported front coupling: $(rep.coupling)"))
    end
end

function _step_ft_redistribute!(
    prob::CoupledMovingStokesProblem{T},
    x_prev::AbstractVector;
    t::T,
    dt::T,
    method::Symbol=:direct,
    kwargs...,
) where {T}
    model = prob.model
    rep = prob.rep
    opts = prob.options

    pts0 = front_points(rep)
    Vn = _start_pressure_volume(model, t)
    pts_guess = _ft_predict_points(rep, pts0, dt)
    history = StokesGHFCouplingHistory(T)

    best = nothing
    threshold_ref = one(T)

    for k in 1:opts.max_iter
        _set_front_points!(rep, pts_guess)
        cur = _ft_solve_and_residual(model, rep, x_prev, Vn, pts0, pts_guess, t, dt, opts; method=method, kwargs...)
        threshold_ref = max(threshold_ref, norm(cur.phase_terms.ΔV, Inf), norm(pts_guess[1], Inf))

        if cur.norm <= max(opts.tol, opts.reltol * threshold_ref)
            best = cur
            break
        end

        normals = rep.state.geom.vertex_normals
        disp = opts.damping .* _ft_residual_displacement(cur.R, model.gridp, pts_guess, opts)
        if opts.velocity_predictor
            disp .+= opts.damping .* _ft_velocity_normal_displacement(
                model,
                cur.sys.x,
                pts_guess,
                normals,
                dt,
                opts.drive_phase,
            )
            hmin = minimum(meshsize(model.gridp))
            clip = opts.step_clip * hmin
            @inbounds for i in eachindex(disp)
                disp[i] = clamp(disp[i], -clip, clip)
            end
        end
        step_norm = norm(disp, Inf)
        _push_history!(
            history;
            iter=k,
            gcl=convert(T, cur.gcl),
            kin=convert(T, cur.kin),
            div=convert(T, cur.div),
            step=step_norm,
            damping=opts.damping,
        )

        if step_norm <= max(opts.tol, opts.reltol * threshold_ref)
            best = cur
            break
        end

        accepted = false
        λ = one(T)
        best_trial = cur
        best_pts = pts_guess
        for _bt in 0:opts.max_backtracks
            trial_pts = _ft_apply_normal_step(pts_guess, normals, disp, λ)
            _set_front_points!(rep, trial_pts)
            trial = _ft_solve_and_residual(model, rep, x_prev, Vn, pts0, trial_pts, t, dt, opts; method=method, kwargs...)
            if trial.norm < best_trial.norm
                best_trial = trial
                best_pts = trial_pts
            end
            if trial.norm <= cur.norm
                pts_guess = trial_pts
                best = trial
                accepted = true
                break
            end
            λ *= opts.backtrack
        end

        if !accepted
            pts_guess = best_pts
            best = best_trial
            best_trial.norm < cur.norm || break
        end
    end

    _set_front_points!(rep, pts_guess; t=t + dt)
    opts.redistribute_markers && FrontTrackingMethods.redistribute!(rep.state, FrontTrackingMethods.CurveEqualArcRedistributor())
    FrontTrackingMethods.refresh_geometry!(rep.state)
    update_front_levelset!(rep.body, front_points(rep), front_points(rep), t + dt, zero(T))

    pts_final = front_points(rep)
    normals = rep.state.geom.vertex_normals
    resize!(rep.last_normal_speed, length(pts_final))
    @inbounds for i in eachindex(pts_final)
        rep.last_normal_speed[i] = dot(pts_final[i] - pts0[i], SVector{2,T}(normals[i])) / dt
    end

    converged = best !== nothing && best.norm <= max(opts.tol, opts.reltol * threshold_ref)
    return (; x=Vector{T}(best.sys.x), sys=best.sys, history, terms=best.terms, residual=best.R, converged)
end

# -----------------------------------------------------------------------------
# :explicit — single Stokes solve with one velocity-based marker update
# -----------------------------------------------------------------------------

function _ft_finalize_step!(
    rep::FrontTrackingRep{T},
    pts0::Vector{SVector{2,T}},
    pts_final::Vector{SVector{2,T}},
    t::T,
    dt::T,
    redistribute_markers::Bool,
) where {T}
    _set_front_points!(rep, pts_final; t=t + dt)
    redistribute_markers && FrontTrackingMethods.redistribute!(
        rep.state, FrontTrackingMethods.CurveEqualArcRedistributor(),
    )
    FrontTrackingMethods.refresh_geometry!(rep.state)
    update_front_levelset!(rep.body, front_points(rep), front_points(rep), t + dt, zero(T))

    pts_now = front_points(rep)
    normals = rep.state.geom.vertex_normals
    resize!(rep.last_normal_speed, length(pts_now))
    @inbounds for i in eachindex(pts_now)
        rep.last_normal_speed[i] = dot(pts_now[i] - pts0[i], SVector{2,T}(normals[i])) / dt
    end
    return pts_now
end

function _step_ft_explicit!(
    prob::CoupledMovingStokesProblem{T},
    x_prev::AbstractVector;
    t::T,
    dt::T,
    method::Symbol=:direct,
    kwargs...,
) where {T}
    model = prob.model
    rep = prob.rep
    opts = prob.options

    pts0 = front_points(rep)
    Vn = _start_pressure_volume(model, t)
    pts_pred = _ft_predict_points(rep, pts0, dt)
    history = StokesGHFCouplingHistory(T)

    _set_front_points!(rep, pts_pred)
    cur = _ft_solve_and_residual(model, rep, x_prev, Vn, pts0, pts_pred, t, dt, opts; method=method, kwargs...)

    # Marker update purely from the solved interface-trace velocity along normals
    normals = rep.state.geom.vertex_normals
    disp = _ft_velocity_normal_displacement(
        model, cur.sys.x, pts_pred, normals, dt, opts.drive_phase,
    )
    hmin = minimum(meshsize(model.gridp))
    clip = opts.step_clip * hmin
    @inbounds for i in eachindex(disp)
        disp[i] = clamp(opts.damping * disp[i], -clip, clip)
    end
    pts_new = _ft_apply_normal_step(pts_pred, normals, disp, one(T))

    _push_history!(
        history;
        iter=1,
        gcl=convert(T, cur.gcl),
        kin=convert(T, cur.kin),
        div=convert(T, cur.div),
        step=convert(T, norm(disp, Inf)),
        damping=opts.damping,
    )

    _ft_finalize_step!(rep, pts0, pts_new, t, dt, opts.redistribute_markers)
    converged = cur.norm <= opts.tol
    return (; x=Vector{T}(cur.sys.x), sys=cur.sys, history, terms=cur.terms, residual=cur.R, converged)
end

# -----------------------------------------------------------------------------
# :ft_lm — Levenberg–Marquardt with finite-difference volume Jacobian
# -----------------------------------------------------------------------------

function _ft_active_pressure_cells(R::AbstractVector{T}) where {T}
    idx = Int[]
    @inbounds for i in eachindex(R)
        (isfinite(R[i]) && !iszero(R[i])) && push!(idx, i)
    end
    return idx
end

function _ft_build_volume_jacobian(
    model::MovingStokesModelTwoPhase{2,T},
    rep::FrontTrackingRep{T},
    x_prev::AbstractVector,
    pts0::Vector{SVector{2,T}},
    pts::Vector{SVector{2,T}},
    normals,
    Vn,
    Rbase_a::Vector{T},
    t::T,
    dt::T,
    opts::CoupledFrontTrackingOptions{T},
    active::Vector{Int};
    method::Symbol=:direct,
    kwargs...,
) where {T}
    nm = length(pts)
    na = length(active)
    J = Matrix{T}(undef, na, nm)
    h = minimum(meshsize(model.gridp))
    ε = max(opts.lm_fd_eps * h, sqrt(eps(T)) * h)
    for m in 1:nm
        pts_eps = copy(pts)
        n_m = SVector{2,T}(normals[m])
        pts_eps[m] = pts[m] + ε * n_m
        update_front_levelset!(rep.body, pts0, pts_eps, t, dt)
        sys = solve_unsteady_moving!(model, x_prev; t=t, dt=dt, scheme=opts.scheme, method=method, kwargs...)
        _, R_eps, _, _, _ = _gcl_step_terms(model, sys.x, Vn, sys.A, opts.drive_phase)
        @inbounds for (k, i) in pairs(active)
            J[k, m] = (R_eps[i] - Rbase_a[k]) / ε
        end
    end
    update_front_levelset!(rep.body, pts0, pts, t, dt)
    return J
end

function _ft_lm_solve_step(
    J::Matrix{T},
    r::Vector{T},
    λ::T,
    opts::CoupledFrontTrackingOptions{T},
    nm::Int,
) where {T}
    H = J' * J
    g = J' * r
    D = Diagonal(max.(diag(H), eps(T)))
    # Mild Tikhonov regularization on displacement magnitude (lm_reg_mass)
    # and a discrete-Laplacian penalty on neighboring displacements (lm_reg_curv).
    A = H + λ * D
    if !iszero(opts.lm_reg_mass)
        A += opts.lm_reg_mass * I
    end
    if !iszero(opts.lm_reg_curv) && nm >= 3
        L = spzeros(T, nm, nm)
        for i in 1:nm
            im = mod1(i - 1, nm)
            ip = mod1(i + 1, nm)
            L[i, i] += 2
            L[i, im] -= 1
            L[i, ip] -= 1
        end
        A += opts.lm_reg_curv * Matrix(L' * L)
    end
    try
        return -(A \ g)
    catch
        return zeros(T, nm)
    end
end

function _step_ft_lm!(
    prob::CoupledMovingStokesProblem{T},
    x_prev::AbstractVector;
    t::T,
    dt::T,
    method::Symbol=:direct,
    kwargs...,
) where {T}
    model = prob.model
    rep = prob.rep
    opts = prob.options

    pts0 = front_points(rep)
    Vn = _start_pressure_volume(model, t)
    pts_guess = _ft_predict_points(rep, pts0, dt)
    history = StokesGHFCouplingHistory(T)

    λ = opts.lm_lambda0
    best = nothing
    threshold_ref = one(T)
    rejects = 0

    for k in 1:opts.max_iter
        _set_front_points!(rep, pts_guess)
        cur = _ft_solve_and_residual(model, rep, x_prev, Vn, pts0, pts_guess, t, dt, opts; method=method, kwargs...)
        threshold_ref = max(threshold_ref, norm(cur.phase_terms.ΔV, Inf), norm(pts_guess[1], Inf))

        if cur.norm <= max(opts.tol, opts.reltol * threshold_ref)
            best = cur
            _push_history!(history; iter=k, gcl=convert(T, cur.gcl), kin=convert(T, cur.kin),
                div=convert(T, cur.div), step=zero(T), damping=opts.damping)
            break
        end

        active = _ft_active_pressure_cells(cur.R)
        if isempty(active)
            best = cur
            break
        end

        normals = rep.state.geom.vertex_normals
        Ra = T[cur.R[i] for i in active]
        J = _ft_build_volume_jacobian(
            model, rep, x_prev, pts0, pts_guess, normals, Vn, Ra, t, dt, opts, active;
            method=method, kwargs...,
        )

        δ = _ft_lm_solve_step(J, Ra, λ, opts, length(pts_guess))
        # smooth and clip displacement, matching :ft_redistribute hygiene
        _ft_smooth_periodic!(δ, opts.smooth)
        hmin = minimum(meshsize(model.gridp))
        clip = opts.step_clip * hmin
        @inbounds for i in eachindex(δ)
            δ[i] = opts.damping * clamp(δ[i], -clip, clip)
        end
        step_norm = norm(δ, Inf)

        # Accept/reject by residual norm
        trial_pts = _ft_apply_normal_step(pts_guess, normals, δ, one(T))
        _set_front_points!(rep, trial_pts)
        trial = _ft_solve_and_residual(model, rep, x_prev, Vn, pts0, trial_pts, t, dt, opts; method=method, kwargs...)

        _push_history!(history; iter=k, gcl=convert(T, trial.gcl), kin=convert(T, trial.kin),
            div=convert(T, trial.div), step=step_norm, damping=λ)

        if trial.norm < cur.norm
            pts_guess = trial_pts
            best = trial
            λ = max(opts.lm_lambda_min, λ / opts.lm_lambda_factor)
            rejects = 0
            if trial.norm <= max(opts.tol, opts.reltol * threshold_ref) ||
               step_norm <= max(opts.tol, opts.reltol * threshold_ref)
                break
            end
        else
            λ = min(opts.lm_lambda_max, λ * opts.lm_lambda_factor)
            rejects += 1
            best === nothing && (best = cur)
            rejects >= opts.lm_max_reject && break
        end
    end

    best === nothing && (best = _ft_solve_and_residual(model, rep, x_prev, Vn, pts0, pts_guess, t, dt, opts; method=method, kwargs...))
    _ft_finalize_step!(rep, pts0, pts_guess, t, dt, opts.redistribute_markers)
    converged = best.norm <= max(opts.tol, opts.reltol * threshold_ref)
    return (; x=Vector{T}(best.sys.x), sys=best.sys, history, terms=best.terms, residual=best.R, converged)
end
