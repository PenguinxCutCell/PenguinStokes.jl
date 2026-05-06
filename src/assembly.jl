function _stokes_blocks(model::StokesModelMono{N,T}) where {N,T}
    nt = model.cap_p.ntotal

    grad_full = model.op_p.G + model.op_p.H
    size(grad_full, 1) == N * nt ||
        throw(ArgumentError("pressure gradient rows ($(size(grad_full, 1))) must equal N*nt ($(N * nt))"))

    visc_omega = ntuple(d -> begin
        opu = model.op_u[d]
        model.mu * (opu.G' * (opu.Winv * opu.G))
    end, N)

    visc_gamma = ntuple(d -> begin
        opu = model.op_u[d]
        model.mu * (opu.G' * (opu.Winv * opu.H))
    end, N)

    grad = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gd = sparse(-grad_full[rows, :])
        if !model.periodic[d]
            capd = model.cap_u[d]
            li = LinearIndices(capd.nnodes)
            @inbounds for I in CartesianIndices(capd.nnodes)
                if I[d] != 1
                    continue
                end
                physical = true
                for k in 1:N
                    if I[k] == capd.nnodes[k]
                        physical = false
                        break
                    end
                end
                physical || continue
                II = CartesianIndex(ntuple(k -> (k == d ? I[k] + 1 : I[k]), N))
                i = li[I]
                j = li[II]
                Aface = capd.buf.A[d][i]
                if !isfinite(Aface) || iszero(Aface)
                    continue
                end
                gd[i, i] = gd[i, i] + 2 * Aface
                gd[i, j] = gd[i, j] - Aface
            end
        end
        gd
    end, N)

    div_omega = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gp = model.op_p.G[rows, :]
        hp = model.op_p.H[rows, :]
        -(gp' + hp')
    end, N)

    div_gamma = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        hp = model.op_p.H[rows, :]
        sparse(hp')
    end, N)

    mass = ntuple(d -> model.rho .* model.cap_u[d].buf.V, N)

    return (; nt, visc_omega, visc_gamma, grad, div_omega, div_gamma, mass)
end

function _assemble_core!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::StokesModelMono{N,T}, blocks, t::T) where {N,T}
    nt = blocks.nt
    layout = model.layout

    @inbounds for d in 1:N
        _insert_block!(A, layout.uomega[d], layout.uomega[d], blocks.visc_omega[d])
        _insert_block!(A, layout.uomega[d], layout.ugamma[d], blocks.visc_gamma[d])
        _insert_block!(A, layout.uomega[d], layout.pomega, blocks.grad[d])

        tie = spdiagm(0 => ones(T, nt))
        _insert_block!(A, layout.ugamma[d], layout.ugamma[d], tie)

        force_vec = _force_values(model, d, t)
        _insert_vec!(b, layout.uomega[d], model.cap_u[d].V * force_vec)

        cut_vec = _cut_values(model.cap_u[d], model.bc_cut[d], t)
        _insert_vec!(b, layout.ugamma[d], cut_vec)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega, layout.uomega[d], blocks.div_omega[d])
        _insert_block!(A, layout.pomega, layout.ugamma[d], blocks.div_gamma[d])
    end

    return A, b
end

function _stokes_phase_blocks(
    cap_p::AssembledCapacity{N,T},
    op_p::DiffusionOps{N,T},
    cap_u::NTuple{N,AssembledCapacity{N,T}},
    op_u::NTuple{N,DiffusionOps{N,T}},
    mu::T,
    rho::T,
    periodic::NTuple{N,Bool},
) where {N,T}
    nt = cap_p.ntotal

    grad_full = op_p.G + op_p.H
    size(grad_full, 1) == N * nt ||
        throw(ArgumentError("pressure gradient rows ($(size(grad_full, 1))) must equal N*nt ($(N * nt))"))

    visc_omega = ntuple(d -> begin
        opud = op_u[d]
        mu * (opud.G' * (opud.Winv * opud.G))
    end, N)

    visc_gamma = ntuple(d -> begin
        opud = op_u[d]
        mu * (opud.G' * (opud.Winv * opud.H))
    end, N)

    grad = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gd = sparse(-grad_full[rows, :])
        if !periodic[d]
            capd = cap_u[d]
            li = LinearIndices(capd.nnodes)
            @inbounds for I in CartesianIndices(capd.nnodes)
                if I[d] != 1
                    continue
                end
                physical = true
                for k in 1:N
                    if I[k] == capd.nnodes[k]
                        physical = false
                        break
                    end
                end
                physical || continue
                II = CartesianIndex(ntuple(k -> (k == d ? I[k] + 1 : I[k]), N))
                i = li[I]
                j = li[II]
                Aface = capd.buf.A[d][i]
                if !isfinite(Aface) || iszero(Aface)
                    continue
                end
                gd[i, i] = gd[i, i] + 2 * Aface
                gd[i, j] = gd[i, j] - Aface
            end
        end
        gd
    end, N)

    div_omega = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gp = op_p.G[rows, :]
        hp = op_p.H[rows, :]
        -(gp' + hp')
    end, N)

    div_gamma = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        hp = op_p.H[rows, :]
        sparse(hp')
    end, N)

    deriv_omega = ntuple(β -> begin
        gω = op_u[β].Winv * op_u[β].G
        ntuple(b -> begin
            rows = ((b - 1) * nt + 1):(b * nt)
            sparse(gω[rows, :])
        end, N)
    end, N)

    deriv_gamma = ntuple(β -> begin
        gγ = op_u[β].Winv * op_u[β].H
        ntuple(b -> begin
            rows = ((b - 1) * nt + 1):(b * nt)
            sparse(gγ[rows, :])
        end, N)
    end, N)

    gamma = zeros(T, nt)
    normals = ntuple(_ -> zeros(T, nt), N)
    @inbounds for i in 1:nt
        Γi = cap_p.buf.Γ[i]
        if !(isfinite(Γi) && Γi > zero(T))
            continue
        end
        gamma[i] = Γi
        nγ = cap_p.n_γ[i]
        for d in 1:N
            nd = nγ[d]
            normals[d][i] = isfinite(nd) ? nd : zero(T)
        end
    end

    mass = ntuple(d -> rho .* cap_u[d].buf.V, N)

    return (
        nt=nt,
        visc_omega=visc_omega,
        visc_gamma=visc_gamma,
        grad=grad,
        div_omega=div_omega,
        div_gamma=div_gamma,
        deriv_omega=deriv_omega,
        deriv_gamma=deriv_gamma,
        gamma=gamma,
        normals=normals,
        mass=mass,
    )
end

function _stokes_blocks(model::StokesModelTwoPhase{N,T}) where {N,T}
    phase1 = _stokes_phase_blocks(
        model.cap_p1,
        model.op_p1,
        model.cap_u1,
        model.op_u1,
        model.mu1,
        model.rho1,
        model.periodic,
    )
    phase2 = _stokes_phase_blocks(
        model.cap_p2,
        model.op_p2,
        model.cap_u2,
        model.op_u2,
        model.mu2,
        model.rho2,
        model.periodic,
    )
    return (; nt=model.cap_p1.ntotal, phase1, phase2)
end

function _interface_condition_values(
    model::StokesModelTwoPhase{N,T},
    d::Int,
    gamma::AbstractVector{T},
    normals::NTuple{N,AbstractVector{T}},
    t::T,
) where {N,T}
    nt = length(gamma)
    α1 = ones(T, nt)
    α2 = ones(T, nt)
    β1 = ones(T, nt)
    β2 = ones(T, nt)
    rhs_jump = zeros(T, nt)
    rhs_trac = zeros(T, nt)

    ic = isnothing(model.bc_interface) ? nothing : model.bc_interface[d]
    sbc = isnothing(ic) ? nothing : ic.scalar
    fbc = isnothing(ic) ? nothing : ic.flux
    if !(sbc === nothing) && !(sbc isa ScalarJump)
        throw(ArgumentError("two-phase Stokes interface scalar condition for component $d must be ScalarJump or nothing"))
    end
    if !(fbc === nothing) && !(fbc isa FluxJump)
        throw(ArgumentError("two-phase Stokes interface flux condition for component $d must be FluxJump or nothing"))
    end

    @inbounds for i in 1:nt
        Γi = gamma[i]
        if !(isfinite(Γi) && Γi > zero(T))
            continue
        end
        xi = model.cap_p1.C_γ[i]
        if sbc === nothing
            rhs_jump[i] = _interface_force_component(model.interface_jump, d, xi, t)
        else
            sb = sbc::ScalarJump
            α1[i] = convert(T, eval_bc(sb.α₁, xi, t))
            α2[i] = convert(T, eval_bc(sb.α₂, xi, t))
            rhs_jump[i] = convert(T, eval_bc(sb.value, xi, t))
        end
        if fbc === nothing
            nγ = SVector{N,T}(ntuple(k -> normals[k][i], N))
            fγ = _interface_force_vector_with_normal(model.interface_force, xi, nγ, t)
            rhs_trac[i] = Γi * _consistent_interface_force_component(fγ, nγ, d)
        else
            fb = fbc::FluxJump
            β1[i] = convert(T, eval_bc(fb.β₁, xi, t))
            β2[i] = convert(T, eval_bc(fb.β₂, xi, t))
            rhs_trac[i] = Γi * convert(T, eval_bc(fb.value, xi, t))
        end
    end
    return α1, α2, β1, β2, rhs_jump, rhs_trac
end

function _interface_condition_values(
    model::MovingStokesModelTwoPhase{N,T},
    cap_p1::AssembledCapacity{N,T},
    d::Int,
    gamma::AbstractVector{T},
    normals::NTuple{N,AbstractVector{T}},
    t::T,
) where {N,T}
    nt = length(gamma)
    α1 = ones(T, nt)
    α2 = ones(T, nt)
    β1 = ones(T, nt)
    β2 = ones(T, nt)
    rhs_jump = zeros(T, nt)
    rhs_trac = zeros(T, nt)

    ic = isnothing(model.bc_interface) ? nothing : model.bc_interface[d]
    sbc = isnothing(ic) ? nothing : ic.scalar
    fbc = isnothing(ic) ? nothing : ic.flux
    if !(sbc === nothing) && !(sbc isa ScalarJump)
        throw(ArgumentError("two-phase Stokes interface scalar condition for component $d must be ScalarJump or nothing"))
    end
    if !(fbc === nothing) && !(fbc isa FluxJump)
        throw(ArgumentError("two-phase Stokes interface flux condition for component $d must be FluxJump or nothing"))
    end

    @inbounds for i in 1:nt
        Γi = gamma[i]
        if !(isfinite(Γi) && Γi > zero(T))
            continue
        end
        xi = cap_p1.C_γ[i]
        if sbc === nothing
            rhs_jump[i] = _interface_force_component(model.interface_jump, d, xi, t)
        else
            sb = sbc::ScalarJump
            α1[i] = convert(T, eval_bc(sb.α₁, xi, t))
            α2[i] = convert(T, eval_bc(sb.α₂, xi, t))
            rhs_jump[i] = convert(T, eval_bc(sb.value, xi, t))
        end
        if fbc === nothing
            nγ = SVector{N,T}(ntuple(k -> normals[k][i], N))
            fγ = _interface_force_vector_with_normal(model.interface_force, xi, nγ, t)
            rhs_trac[i] = Γi * _consistent_interface_force_component(fγ, nγ, d)
        else
            fb = fbc::FluxJump
            β1[i] = convert(T, eval_bc(fb.β₁, xi, t))
            β2[i] = convert(T, eval_bc(fb.β₂, xi, t))
            rhs_trac[i] = Γi * convert(T, eval_bc(fb.value, xi, t))
        end
    end
    return α1, α2, β1, β2, rhs_jump, rhs_trac
end

function _assemble_interface_traction_rows!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::StokesModelTwoPhase{N,T},
    blocks,
    t::T,
) where {N,T}
    layout = model.layout
    phase1 = blocks.phase1
    phase2 = blocks.phase2

    @inbounds for α in 1:N
        rows_trac = layout.ugamma1[α]
        rows_jump = layout.ugamma2[α]
        α1, α2, β1, β2, rhs_jump, rhs_trac = _interface_condition_values(model, α, phase1.gamma, phase1.normals, t)

        Tp1 = spdiagm(0 => -(β1 .* phase1.gamma .* phase1.normals[α]))
        Tp2 = spdiagm(0 => -(β2 .* phase2.gamma .* phase2.normals[α]))
        _insert_block!(A, rows_trac, layout.pomega1, Tp1)
        _insert_block!(A, rows_trac, layout.pomega2, Tp2)

        for β in 1:N
            w1 = model.mu1 .* (β1 .* phase1.gamma .* phase1.normals[β])
            D1 = spdiagm(0 => w1)
            _insert_block!(A, rows_trac, layout.uomega1[α], sparse(D1 * phase1.deriv_omega[α][β]))
            _insert_block!(A, rows_trac, layout.uomega1[β], sparse(D1 * phase1.deriv_omega[β][α]))
            _insert_block!(A, rows_trac, layout.ugamma1[α], sparse(D1 * phase1.deriv_gamma[α][β]))
            _insert_block!(A, rows_trac, layout.ugamma1[β], sparse(D1 * phase1.deriv_gamma[β][α]))

            w2 = model.mu2 .* (β2 .* phase2.gamma .* phase2.normals[β])
            D2 = spdiagm(0 => w2)
            _insert_block!(A, rows_trac, layout.uomega2[α], sparse(D2 * phase2.deriv_omega[α][β]))
            _insert_block!(A, rows_trac, layout.uomega2[β], sparse(D2 * phase2.deriv_omega[β][α]))
            _insert_block!(A, rows_trac, layout.ugamma2[α], sparse(D2 * phase2.deriv_gamma[α][β]))
            _insert_block!(A, rows_trac, layout.ugamma2[β], sparse(D2 * phase2.deriv_gamma[β][α]))
        end

        _insert_vec!(b, rows_trac, rhs_trac)

        _insert_block!(A, rows_jump, layout.ugamma1[α], spdiagm(0 => α1))
        _insert_block!(A, rows_jump, layout.ugamma2[α], spdiagm(0 => -α2))
        _insert_vec!(b, rows_jump, rhs_jump)
    end

    return A, b
end

function _assemble_interface_traction_rows!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::MovingStokesModelTwoPhase{N,T},
    blocks,
    t::T,
) where {N,T}
    isnothing(model.cap_p1_end) && throw(ArgumentError("moving two-phase model pressure end-capacity cache is not built"))
    cap_p1_end = something(model.cap_p1_end)

    layout = model.layout
    phase1 = blocks.phase1
    phase2 = blocks.phase2

    @inbounds for α in 1:N
        rows_trac = layout.ugamma1[α]
        rows_jump = layout.ugamma2[α]
        α1, α2, β1, β2, rhs_jump, rhs_trac = _interface_condition_values(
            model,
            cap_p1_end,
            α,
            phase1.gamma,
            phase1.normals,
            t,
        )

        Tp1 = spdiagm(0 => -(β1 .* phase1.gamma .* phase1.normals[α]))
        Tp2 = spdiagm(0 => -(β2 .* phase2.gamma .* phase2.normals[α]))
        _insert_block!(A, rows_trac, layout.pomega1, Tp1)
        _insert_block!(A, rows_trac, layout.pomega2, Tp2)

        for β in 1:N
            w1 = model.mu1 .* (β1 .* phase1.gamma .* phase1.normals[β])
            D1 = spdiagm(0 => w1)
            _insert_block!(A, rows_trac, layout.uomega1[α], sparse(D1 * phase1.deriv_omega[α][β]))
            _insert_block!(A, rows_trac, layout.uomega1[β], sparse(D1 * phase1.deriv_omega[β][α]))
            _insert_block!(A, rows_trac, layout.ugamma1[α], sparse(D1 * phase1.deriv_gamma[α][β]))
            _insert_block!(A, rows_trac, layout.ugamma1[β], sparse(D1 * phase1.deriv_gamma[β][α]))

            w2 = model.mu2 .* (β2 .* phase2.gamma .* phase2.normals[β])
            D2 = spdiagm(0 => w2)
            _insert_block!(A, rows_trac, layout.uomega2[α], sparse(D2 * phase2.deriv_omega[α][β]))
            _insert_block!(A, rows_trac, layout.uomega2[β], sparse(D2 * phase2.deriv_omega[β][α]))
            _insert_block!(A, rows_trac, layout.ugamma2[α], sparse(D2 * phase2.deriv_gamma[α][β]))
            _insert_block!(A, rows_trac, layout.ugamma2[β], sparse(D2 * phase2.deriv_gamma[β][α]))
        end

        _insert_vec!(b, rows_trac, rhs_trac)

        _insert_block!(A, rows_jump, layout.ugamma1[α], spdiagm(0 => α1))
        _insert_block!(A, rows_jump, layout.ugamma2[α], spdiagm(0 => -α2))
        _insert_vec!(b, rows_jump, rhs_jump)
    end

    return A, b
end

function _apply_auxiliary_trace_rows!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::MovingStokesModelTwoPhase{N,T},
) where {N,T}
    isnothing(model.cap_u1_end) && throw(ArgumentError("moving two-phase model phase-1 velocity end-capacity cache is not built"))
    isnothing(model.cap_u2_end) && throw(ArgumentError("moving two-phase model phase-2 velocity end-capacity cache is not built"))
    isnothing(model.cap_p1_end) && throw(ArgumentError("moving two-phase model phase-1 pressure end-capacity cache is not built"))

    cap_u1_end = something(model.cap_u1_end)
    cap_u2_end = something(model.cap_u2_end)
    cap_p1_end = something(model.cap_p1_end)
    _, agamma_p = _cell_activity_masks(cap_p1_end)
    layout = model.layout

    @inbounds for d in 1:N
        aomega1, agamma1 = _cell_activity_masks(cap_u1_end[d])
        aomega2, agamma2 = _cell_activity_masks(cap_u2_end[d])
        for i in 1:layout.nt
            (agamma1[i] || agamma2[i]) || continue
            agamma_p[i] && continue

            cols = Int[layout.ugamma1[d][i]]
            vals = T[one(T)]
            v1 = aomega1[i] ? cap_u1_end[d].buf.V[i] : zero(T)
            v2 = aomega2[i] ? cap_u2_end[d].buf.V[i] : zero(T)
            wsum = ((isfinite(v1) && v1 > zero(T)) ? v1 : zero(T)) +
                   ((isfinite(v2) && v2 > zero(T)) ? v2 : zero(T))
            wsum > zero(T) || continue

            if isfinite(v1) && v1 > zero(T)
                push!(cols, layout.uomega1[d][i])
                push!(vals, -v1 / wsum)
            end
            if isfinite(v2) && v2 > zero(T)
                push!(cols, layout.uomega2[d][i])
                push!(vals, -v2 / wsum)
            end

            _set_sparse_row!(A, b, layout.ugamma1[d][i], cols, vals, zero(T))
        end
    end

    return A, b
end

function _assemble_core!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::StokesModelTwoPhase{N,T}, blocks, t::T) where {N,T}
    layout = model.layout
    phase1 = blocks.phase1
    phase2 = blocks.phase2

    @inbounds for d in 1:N
        _insert_block!(A, layout.uomega1[d], layout.uomega1[d], phase1.visc_omega[d])
        _insert_block!(A, layout.uomega1[d], layout.ugamma1[d], phase1.visc_gamma[d])
        _insert_block!(A, layout.uomega1[d], layout.pomega1, phase1.grad[d])
        f1 = _force_values(model, 1, d, t)
        _insert_vec!(b, layout.uomega1[d], model.cap_u1[d].V * f1)

        _insert_block!(A, layout.uomega2[d], layout.uomega2[d], phase2.visc_omega[d])
        _insert_block!(A, layout.uomega2[d], layout.ugamma2[d], phase2.visc_gamma[d])
        _insert_block!(A, layout.uomega2[d], layout.pomega2, phase2.grad[d])
        f2 = _force_values(model, 2, d, t)
        _insert_vec!(b, layout.uomega2[d], model.cap_u2[d].V * f2)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega1, layout.uomega1[d], phase1.div_omega[d])
        _insert_block!(A, layout.pomega1, layout.ugamma1[d], phase1.div_gamma[d])
        _insert_block!(A, layout.pomega2, layout.uomega2[d], phase2.div_omega[d])
        _insert_block!(A, layout.pomega2, layout.ugamma2[d], phase2.div_gamma[d])
    end

    _assemble_interface_traction_rows!(A, b, model, blocks, t)
    return A, b
end

"""
    assemble_steady!(sys, model, t=0)

Assemble steady Stokes linear system into `sys` for `StokesModelMono` or
`StokesModelTwoPhase` at time `t`.

Mutates `sys.A` and `sys.b` in place.
"""
function assemble_steady!(sys::LinearSystem{T}, model::StokesModelTwoPhase{N,T}, t::T=zero(T)) where {N,T}
    blocks = _stokes_blocks(model)
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)

    _assemble_core!(A, b, model, blocks, t)
    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t)
    _apply_symmetry_box_bc!(A, b, model, t, traction_locked_rows)
    _apply_velocity_box_bc!(A, b, model, t; locked_rows=traction_locked_rows)
    _apply_pressure_box_bc!(A, b, model, t)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

"""
    assemble_unsteady!(sys, model, x_prev, t, dt; scheme=:BE)

Assemble one unsteady theta-step system for `StokesModelMono` or
`StokesModelTwoPhase`.

`scheme` supports `:BE`, `:CN`, or numeric `theta ∈ [0,1]`.
"""
function assemble_unsteady!(
    sys::LinearSystem{T},
    model::StokesModelTwoPhase{N,T},
    x_prev::AbstractVector,
    t::T,
    dt::T;
    scheme=:BE,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    theta = _theta_from_scheme(T, scheme)
    t_next = t + dt

    blocks = _stokes_blocks(model)
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)
    layout = model.layout
    xfull_prev = _expand_prev_state(model, x_prev)

    phase1 = blocks.phase1
    phase2 = blocks.phase2

    @inbounds for d in 1:N
        mdt1 = phase1.mass[d] ./ dt
        block11 = theta .* phase1.visc_omega[d] + spdiagm(0 => mdt1)
        block1g = theta .* phase1.visc_gamma[d]
        _insert_block!(A, layout.uomega1[d], layout.uomega1[d], block11)
        _insert_block!(A, layout.uomega1[d], layout.ugamma1[d], block1g)
        _insert_block!(A, layout.uomega1[d], layout.pomega1, phase1.grad[d])

        u1_prev = Vector{T}(xfull_prev[layout.uomega1[d]])
        ug1_prev = Vector{T}(xfull_prev[layout.ugamma1[d]])
        rhs1 = mdt1 .* u1_prev
        if theta != one(T)
            rhs1 .-= (one(T) - theta) .* (phase1.visc_omega[d] * u1_prev + phase1.visc_gamma[d] * ug1_prev)
        end
        f1_prev = _force_values(model, 1, d, t)
        f1_next = _force_values(model, 1, d, t_next)
        f1_theta = theta .* f1_next .+ (one(T) - theta) .* f1_prev
        rhs1 .+= model.cap_u1[d].V * f1_theta
        _insert_vec!(b, layout.uomega1[d], rhs1)

        mdt2 = phase2.mass[d] ./ dt
        block22 = theta .* phase2.visc_omega[d] + spdiagm(0 => mdt2)
        block2g = theta .* phase2.visc_gamma[d]
        _insert_block!(A, layout.uomega2[d], layout.uomega2[d], block22)
        _insert_block!(A, layout.uomega2[d], layout.ugamma2[d], block2g)
        _insert_block!(A, layout.uomega2[d], layout.pomega2, phase2.grad[d])

        u2_prev = Vector{T}(xfull_prev[layout.uomega2[d]])
        ug2_prev = Vector{T}(xfull_prev[layout.ugamma2[d]])
        rhs2 = mdt2 .* u2_prev
        if theta != one(T)
            rhs2 .-= (one(T) - theta) .* (phase2.visc_omega[d] * u2_prev + phase2.visc_gamma[d] * ug2_prev)
        end
        f2_prev = _force_values(model, 2, d, t)
        f2_next = _force_values(model, 2, d, t_next)
        f2_theta = theta .* f2_next .+ (one(T) - theta) .* f2_prev
        rhs2 .+= model.cap_u2[d].V * f2_theta
        _insert_vec!(b, layout.uomega2[d], rhs2)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega1, layout.uomega1[d], phase1.div_omega[d])
        _insert_block!(A, layout.pomega1, layout.ugamma1[d], phase1.div_gamma[d])
        _insert_block!(A, layout.pomega2, layout.uomega2[d], phase2.div_omega[d])
        _insert_block!(A, layout.pomega2, layout.ugamma2[d], phase2.div_gamma[d])
    end

    _assemble_interface_traction_rows!(A, b, model, blocks, t_next)
    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t_next)
    _apply_symmetry_box_bc!(A, b, model, t_next, traction_locked_rows)
    _apply_velocity_box_bc!(A, b, model, t_next; locked_rows=traction_locked_rows)
    _apply_pressure_box_bc!(A, b, model, t_next)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

function assemble_steady!(sys::LinearSystem{T}, model::StokesModelMono{N,T}, t::T=zero(T)) where {N,T}
    blocks = _stokes_blocks(model)
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)

    _assemble_core!(A, b, model, blocks, t)
    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t)
    _apply_symmetry_box_bc!(A, b, model, t, traction_locked_rows)
    _apply_velocity_box_bc!(A, b, model, t; locked_rows=traction_locked_rows)
    _apply_pressure_box_bc!(A, b, model, t)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

function assemble_unsteady!(
    sys::LinearSystem{T},
    model::StokesModelMono{N,T},
    x_prev::AbstractVector,
    t::T,
    dt::T;
    scheme=:BE,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    theta = _theta_from_scheme(T, scheme)
    t_next = t + dt

    blocks = _stokes_blocks(model)
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)

    layout = model.layout
    nt = blocks.nt
    xfull_prev = _expand_prev_state(model, x_prev)

    @inbounds for d in 1:N
        mdt = blocks.mass[d] ./ dt
        block_oo = theta * blocks.visc_omega[d] + spdiagm(0 => mdt)
        block_og = theta * blocks.visc_gamma[d]

        _insert_block!(A, layout.uomega[d], layout.uomega[d], block_oo)
        _insert_block!(A, layout.uomega[d], layout.ugamma[d], block_og)
        _insert_block!(A, layout.uomega[d], layout.pomega, blocks.grad[d])

        tie = spdiagm(0 => ones(T, nt))
        _insert_block!(A, layout.ugamma[d], layout.ugamma[d], tie)

        uo_prev = Vector{T}(xfull_prev[layout.uomega[d]])
        ug_prev = Vector{T}(xfull_prev[layout.ugamma[d]])

        rhs = mdt .* uo_prev
        if theta != one(T)
            rhs .-= (one(T) - theta) .* (blocks.visc_omega[d] * uo_prev + blocks.visc_gamma[d] * ug_prev)
        end

        f_prev = _force_values(model, d, t)
        f_next = _force_values(model, d, t_next)
        ftheta = theta .* f_next .+ (one(T) - theta) .* f_prev
        rhs .+= model.cap_u[d].V * ftheta
        _insert_vec!(b, layout.uomega[d], rhs)

        cut_next = _cut_values(model.cap_u[d], model.bc_cut[d], t_next)
        _insert_vec!(b, layout.ugamma[d], cut_next)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega, layout.uomega[d], blocks.div_omega[d])
        _insert_block!(A, layout.pomega, layout.ugamma[d], blocks.div_gamma[d])
    end

    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t_next)
    _apply_symmetry_box_bc!(A, b, model, t_next, traction_locked_rows)
    _apply_velocity_box_bc!(A, b, model, t_next; locked_rows=traction_locked_rows)
    _apply_pressure_box_bc!(A, b, model, t_next)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

function _apply_inactive_moving_velocity_extension!(
    b::Vector{T},
    model::MovingStokesModelMono{N,T},
    active_rows::BitVector,
    t::T,
) where {N,T}
    isnothing(model.cap_u_end) && throw(ArgumentError("moving model velocity end-capacity cache is not built"))
    cap_u_end = something(model.cap_u_end)
    layout = model.layout

    @inbounds for d in 1:N
        bc_cut = model.bc_cut_u[d]
        bc_cut isa Dirichlet ||
            throw(ArgumentError("cut-cell velocity condition currently supports Dirichlet only"))
        capd = cap_u_end[d]
        for i in 1:capd.ntotal
            row_omega = layout.uomega[d][i]
            xω = capd.C_ω[i]
            if !active_rows[row_omega] && all(isfinite, xω)
                b[row_omega] = convert(T, eval_bc(bc_cut.value, xω, t))
            end

            row_gamma = layout.ugamma[d][i]
            if !active_rows[row_gamma]
                xγ = capd.C_γ[i]
                if all(isfinite, xγ)
                    b[row_gamma] = convert(T, eval_bc(bc_cut.value, xγ, t))
                elseif all(isfinite, xω)
                    b[row_gamma] = convert(T, eval_bc(bc_cut.value, xω, t))
                end
            end
        end
    end

    return b
end

"""
    assemble_unsteady_moving!(sys, model, x_prev, t, dt; scheme=:CN)

Assemble one unsteady moving-boundary theta-step system for
`MovingStokesModelMono`, using slab geometry over `[t, t+dt]` and end-time box
BC/gauge application.
"""
function assemble_unsteady_moving!(
    sys::LinearSystem{T},
    model::MovingStokesModelMono{N,T},
    x_prev::AbstractVector,
    t::T,
    dt::T;
    scheme=:CN,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    theta = _theta_from_scheme(T, scheme)
    psip, psim = _psi_functions(scheme)
    t_next = t + dt

    _build_moving_slab!(model, t, dt)

    op_p_slab = something(model.op_p_slab)
    op_p_end = something(model.op_p_end)
    cap_u_slab = something(model.cap_u_slab)
    op_u_slab = something(model.op_u_slab)
    cap_u_end = something(model.cap_u_end)
    Vun = something(model.Vun)
    Vun1 = something(model.Vun1)

    nt = model.layout.nt
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)
    layout = model.layout
    xfull_prev = _expand_prev_state(model, x_prev)

    grad_full = op_p_slab.G + op_p_slab.H
    size(grad_full, 1) == N * nt ||
        throw(ArgumentError("pressure gradient rows ($(size(grad_full, 1))) must equal N*nt ($(N * nt))"))

    grad = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gd = sparse(-grad_full[rows, :])
        if !model.periodic[d]
            capd = cap_u_end[d]
            li = LinearIndices(capd.nnodes)
            @inbounds for I in CartesianIndices(capd.nnodes)
                if I[d] != 1
                    continue
                end
                physical = true
                for k in 1:N
                    if I[k] == capd.nnodes[k]
                        physical = false
                        break
                    end
                end
                physical || continue
                II = CartesianIndex(ntuple(k -> (k == d ? I[k] + 1 : I[k]), N))
                i = li[I]
                j = li[II]
                Aface = capd.buf.A[d][i]
                if !isfinite(Aface) || iszero(Aface)
                    continue
                end
                gd[i, i] = gd[i, i] + 2 * Aface
                gd[i, j] = gd[i, j] - Aface
            end
        end
        gd
    end, N)

    div_omega = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gp = op_p_slab.G[rows, :]
        hp = op_p_slab.H[rows, :]
        -(gp' + hp')
    end, N)

    div_gamma = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        hp = op_p_slab.H[rows, :]
        sparse(hp')
    end, N)

    @inbounds for d in 1:N
        opud = op_u_slab[d]
        K = model.mu * (opud.G' * (opud.Winv * opud.G))
        C = model.mu * (opud.G' * (opud.Winv * opud.H))
        M1 = spdiagm(0 => model.rho .* Vun1[d])
        M0 = spdiagm(0 => model.rho .* Vun[d])
        Ψp = spdiagm(0 => T[psip(Vun[d][i], Vun1[d][i]) for i in 1:nt])
        Ψm = spdiagm(0 => T[psim(Vun[d][i], Vun1[d][i]) for i in 1:nt])

        # Ψp/Ψm already encode the selected temporal scheme for moving-slab terms.
        # Do not apply θ a second time to K/C contributions.
        A_oo = M1 + (K * Ψp)
        A_og = -(M1 - M0) + (C * Ψp)
        _insert_block!(A, layout.uomega[d], layout.uomega[d], A_oo)
        _insert_block!(A, layout.uomega[d], layout.ugamma[d], A_og)
        _insert_block!(A, layout.uomega[d], layout.pomega, grad[d])

        tie = spdiagm(0 => ones(T, nt))
        _insert_block!(A, layout.ugamma[d], layout.ugamma[d], tie)

        uω_prev = Vector{T}(xfull_prev[layout.uomega[d]])
        uγ_prev = Vector{T}(xfull_prev[layout.ugamma[d]])
        rhs = (M0 - (K * Ψm)) * uω_prev
        rhs .-= (C * Ψm) * uγ_prev

        f_prev = _force_values(model, cap_u_end[d], d, t)
        f_next = _force_values(model, cap_u_end[d], d, t_next)
        rhs .+= theta .* (cap_u_slab[d].V * f_next) .+ (one(T) - theta) .* (cap_u_slab[d].V * f_prev)
        _insert_vec!(b, layout.uomega[d], rhs)

        cut_next = _cut_values(cap_u_end[d], model.bc_cut_u[d], t_next)
        _insert_vec!(b, layout.ugamma[d], cut_next)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega, layout.uomega[d], div_omega[d])
        _insert_block!(A, layout.pomega, layout.ugamma[d], div_gamma[d])
    end

    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t_next)
    _apply_symmetry_box_bc!(A, b, model, t_next, traction_locked_rows)
    _apply_velocity_box_bc!(A, b, model, t_next; locked_rows=traction_locked_rows)
    _apply_pressure_box_bc!(A, b, model, t_next)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)
    _apply_inactive_moving_velocity_extension!(b, model, active_rows, t_next)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

"""
    assemble_unsteady_moving!(sys, model::MovingStokesModelTwoPhase, x_prev, t, dt; scheme=:CN)

Assemble one unsteady moving-interface theta-step system for
`MovingStokesModelTwoPhase`, using slab geometry over `[t, t+dt]` and end-time
box BC/gauge application.
"""
function assemble_unsteady_moving!(
    sys::LinearSystem{T},
    model::MovingStokesModelTwoPhase{N,T},
    x_prev::AbstractVector,
    t::T,
    dt::T;
    scheme=:CN,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    theta = _theta_from_scheme(T, scheme)
    psip, psim = _psi_functions(scheme)
    t_next = t + dt

    _build_moving_slab!(model, t, dt)

    cap_p1_slab = something(model.cap_p1_slab)
    op_p1_slab = something(model.op_p1_slab)
    cap_p1_end = something(model.cap_p1_end)
    op_p1_end = something(model.op_p1_end)
    cap_u1_slab = something(model.cap_u1_slab)
    op_u1_slab = something(model.op_u1_slab)
    cap_u1_end = something(model.cap_u1_end)
    op_u1_end = something(model.op_u1_end)
    Vu1n = something(model.Vu1n)
    Vu1n1 = something(model.Vu1n1)

    cap_p2_slab = something(model.cap_p2_slab)
    op_p2_slab = something(model.op_p2_slab)
    cap_p2_end = something(model.cap_p2_end)
    op_p2_end = something(model.op_p2_end)
    cap_u2_slab = something(model.cap_u2_slab)
    op_u2_slab = something(model.op_u2_slab)
    cap_u2_end = something(model.cap_u2_end)
    op_u2_end = something(model.op_u2_end)
    Vu2n = something(model.Vu2n)
    Vu2n1 = something(model.Vu2n1)

    nt = model.layout.nt
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)
    layout = model.layout
    xfull_prev = _expand_prev_state(model, x_prev)

    grad_full1 = op_p1_slab.G + op_p1_slab.H
    size(grad_full1, 1) == N * nt ||
        throw(ArgumentError("phase-1 pressure gradient rows ($(size(grad_full1, 1))) must equal N*nt ($(N * nt))"))
    grad_full2 = op_p2_slab.G + op_p2_slab.H
    size(grad_full2, 1) == N * nt ||
        throw(ArgumentError("phase-2 pressure gradient rows ($(size(grad_full2, 1))) must equal N*nt ($(N * nt))"))

    grad1 = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gd = sparse(-grad_full1[rows, :])
        if !model.periodic[d]
            capd = cap_u1_end[d]
            li = LinearIndices(capd.nnodes)
            @inbounds for I in CartesianIndices(capd.nnodes)
                if I[d] != 1
                    continue
                end
                physical = true
                for k in 1:N
                    if I[k] == capd.nnodes[k]
                        physical = false
                        break
                    end
                end
                physical || continue
                II = CartesianIndex(ntuple(k -> (k == d ? I[k] + 1 : I[k]), N))
                i = li[I]
                j = li[II]
                Aface = capd.buf.A[d][i]
                if !isfinite(Aface) || iszero(Aface)
                    continue
                end
                gd[i, i] = gd[i, i] + 2 * Aface
                gd[i, j] = gd[i, j] - Aface
            end
        end
        gd
    end, N)

    grad2 = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gd = sparse(-grad_full2[rows, :])
        if !model.periodic[d]
            capd = cap_u2_end[d]
            li = LinearIndices(capd.nnodes)
            @inbounds for I in CartesianIndices(capd.nnodes)
                if I[d] != 1
                    continue
                end
                physical = true
                for k in 1:N
                    if I[k] == capd.nnodes[k]
                        physical = false
                        break
                    end
                end
                physical || continue
                II = CartesianIndex(ntuple(k -> (k == d ? I[k] + 1 : I[k]), N))
                i = li[I]
                j = li[II]
                Aface = capd.buf.A[d][i]
                if !isfinite(Aface) || iszero(Aface)
                    continue
                end
                gd[i, i] = gd[i, i] + 2 * Aface
                gd[i, j] = gd[i, j] - Aface
            end
        end
        gd
    end, N)

    div_omega1 = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gp = op_p1_slab.G[rows, :]
        hp = op_p1_slab.H[rows, :]
        -(gp' + hp')
    end, N)
    div_gamma1 = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        hp = op_p1_slab.H[rows, :]
        sparse(hp')
    end, N)

    div_omega2 = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gp = op_p2_slab.G[rows, :]
        hp = op_p2_slab.H[rows, :]
        -(gp' + hp')
    end, N)
    div_gamma2 = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        hp = op_p2_slab.H[rows, :]
        sparse(hp')
    end, N)

    @inbounds for d in 1:N
        opu1d = op_u1_slab[d]
        K1 = model.mu1 * (opu1d.G' * (opu1d.Winv * opu1d.G))
        C1 = model.mu1 * (opu1d.G' * (opu1d.Winv * opu1d.H))
        M11 = spdiagm(0 => model.rho1 .* Vu1n1[d])
        M10 = spdiagm(0 => model.rho1 .* Vu1n[d])
        Ψ1p = spdiagm(0 => T[psip(Vu1n[d][i], Vu1n1[d][i]) for i in 1:nt])
        Ψ1m = spdiagm(0 => T[psim(Vu1n[d][i], Vu1n1[d][i]) for i in 1:nt])

        A11 = M11 + (K1 * Ψ1p)
        A1g = -(M11 - M10) + (C1 * Ψ1p)
        _insert_block!(A, layout.uomega1[d], layout.uomega1[d], A11)
        _insert_block!(A, layout.uomega1[d], layout.ugamma1[d], A1g)
        _insert_block!(A, layout.uomega1[d], layout.pomega1, grad1[d])

        uω1_prev = Vector{T}(xfull_prev[layout.uomega1[d]])
        uγ1_prev = Vector{T}(xfull_prev[layout.ugamma1[d]])
        rhs1 = (M10 - (K1 * Ψ1m)) * uω1_prev
        rhs1 .-= (C1 * Ψ1m) * uγ1_prev
        f1_prev = _force_values(model, cap_u1_end[d], 1, d, t)
        f1_next = _force_values(model, cap_u1_end[d], 1, d, t_next)
        rhs1 .+= theta .* (cap_u1_slab[d].V * f1_next) .+ (one(T) - theta) .* (cap_u1_slab[d].V * f1_prev)
        _insert_vec!(b, layout.uomega1[d], rhs1)

        opu2d = op_u2_slab[d]
        K2 = model.mu2 * (opu2d.G' * (opu2d.Winv * opu2d.G))
        C2 = model.mu2 * (opu2d.G' * (opu2d.Winv * opu2d.H))
        M21 = spdiagm(0 => model.rho2 .* Vu2n1[d])
        M20 = spdiagm(0 => model.rho2 .* Vu2n[d])
        Ψ2p = spdiagm(0 => T[psip(Vu2n[d][i], Vu2n1[d][i]) for i in 1:nt])
        Ψ2m = spdiagm(0 => T[psim(Vu2n[d][i], Vu2n1[d][i]) for i in 1:nt])

        A22 = M21 + (K2 * Ψ2p)
        A2g = -(M21 - M20) + (C2 * Ψ2p)
        _insert_block!(A, layout.uomega2[d], layout.uomega2[d], A22)
        _insert_block!(A, layout.uomega2[d], layout.ugamma2[d], A2g)
        _insert_block!(A, layout.uomega2[d], layout.pomega2, grad2[d])

        uω2_prev = Vector{T}(xfull_prev[layout.uomega2[d]])
        uγ2_prev = Vector{T}(xfull_prev[layout.ugamma2[d]])
        rhs2 = (M20 - (K2 * Ψ2m)) * uω2_prev
        rhs2 .-= (C2 * Ψ2m) * uγ2_prev
        f2_prev = _force_values(model, cap_u2_end[d], 2, d, t)
        f2_next = _force_values(model, cap_u2_end[d], 2, d, t_next)
        rhs2 .+= theta .* (cap_u2_slab[d].V * f2_next) .+ (one(T) - theta) .* (cap_u2_slab[d].V * f2_prev)
        _insert_vec!(b, layout.uomega2[d], rhs2)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega1, layout.uomega1[d], div_omega1[d])
        _insert_block!(A, layout.pomega1, layout.ugamma1[d], div_gamma1[d])
        _insert_block!(A, layout.pomega2, layout.uomega2[d], div_omega2[d])
        _insert_block!(A, layout.pomega2, layout.ugamma2[d], div_gamma2[d])
    end

    blocks_iface = (
        nt=nt,
        phase1=_stokes_phase_blocks(cap_p1_end, op_p1_end, cap_u1_end, op_u1_end, model.mu1, model.rho1, model.periodic),
        phase2=_stokes_phase_blocks(cap_p2_end, op_p2_end, cap_u2_end, op_u2_end, model.mu2, model.rho2, model.periodic),
    )
    _assemble_interface_traction_rows!(A, b, model, blocks_iface, t_next)
    _apply_auxiliary_trace_rows!(A, b, model)

    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t_next)
    _apply_symmetry_box_bc!(A, b, model, t_next, traction_locked_rows)
    _apply_velocity_box_bc!(A, b, model, t_next; locked_rows=traction_locked_rows)
    _apply_pressure_box_bc!(A, b, model, t_next)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

"""
    solve_steady!(model; t=0, method=:direct, kwargs...)

Assemble and solve a steady Stokes system for `StokesModelMono` or
`StokesModelTwoPhase`. Returns a `LinearSystem` with solution in `sys.x`.
"""
function solve_steady!(
    model::StokesModelMono{N,T};
    t::T=zero(T),
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_steady!(sys, model, t)
    solve!(sys; method=method, kwargs...)
    return sys
end

function solve_steady!(
    model::StokesModelTwoPhase{N,T};
    t::T=zero(T),
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_steady!(sys, model, t)
    solve!(sys; method=method, kwargs...)
    return sys
end

"""
    solve_unsteady!(model, x_prev; t=0, dt, scheme=:BE, method=:direct, kwargs...)

Assemble and solve one unsteady theta-step for `StokesModelMono` or
`StokesModelTwoPhase`.
"""
function solve_unsteady!(
    model::StokesModelMono{N,T},
    x_prev::AbstractVector;
    t::T=zero(T),
    dt::T,
    scheme=:BE,
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_unsteady!(sys, model, x_prev, t, dt; scheme=scheme)
    solve!(sys; method=method, kwargs...)
    return sys
end

function solve_unsteady!(
    model::StokesModelTwoPhase{N,T},
    x_prev::AbstractVector;
    t::T=zero(T),
    dt::T,
    scheme=:BE,
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_unsteady!(sys, model, x_prev, t, dt; scheme=scheme)
    solve!(sys; method=method, kwargs...)
    return sys
end

"""
    solve_unsteady_moving!(model, x_prev; t=0, dt, scheme=:CN, method=:direct, kwargs...)

Assemble and solve one unsteady moving-boundary theta-step for
`MovingStokesModelMono`.
"""
function solve_unsteady_moving!(
    model::MovingStokesModelMono{N,T},
    x_prev::AbstractVector;
    t::T=zero(T),
    dt::T,
    scheme=:CN,
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_unsteady_moving!(sys, model, x_prev, t, dt; scheme=scheme)
    solve!(sys; method=method, kwargs...)
    return sys
end

"""
    solve_unsteady_moving!(model::MovingStokesModelTwoPhase, x_prev; t=0, dt, scheme=:CN, method=:direct, kwargs...)

Assemble and solve one unsteady moving-interface theta-step for
`MovingStokesModelTwoPhase`.
"""
function solve_unsteady_moving!(
    model::MovingStokesModelTwoPhase{N,T},
    x_prev::AbstractVector;
    t::T=zero(T),
    dt::T,
    scheme=:CN,
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_unsteady_moving!(sys, model, x_prev, t, dt; scheme=scheme)
    solve!(sys; method=method, kwargs...)
    return sys
end

@inline function _is_zero_identity_row(
    A::SparseMatrixCSC{T,Int},
    b::AbstractVector{T},
    row::Int;
    atol::T=sqrt(eps(T)),
) where {T}
    idx, vals = findnz(A[row, :])
    if length(idx) != 1
        return false
    end
    return idx[1] == row &&
        abs(vals[1] - one(T)) <= atol &&
        abs(b[row]) <= atol
end
