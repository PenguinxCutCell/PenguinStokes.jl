using Test
using LinearAlgebra: norm, dot
using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs
using PenguinStokes

# ── helpers ──────────────────────────────────────────────────────────────────

function make_circle_state(X::NTuple{2}, V::NTuple{2}=(0.0, 0.0))
    return RigidBodyState(X, V)
end

function make_circle_params(R::Float64)
    shape = Circle(R)
    m = 1.0
    rho_body = 1.0
    return RigidBodyParams2D(m, rho_body, shape, SVector(0.0, 0.0); buoyancy=false)
end

# ── 1. Wall gap for circle ────────────────────────────────────────────────────

@testset "Wall gap for circle" begin
    R = 0.1
    params = make_circle_params(R)
    bottom_wall = PlanarWallContact((0.0, 1.0), (0.0, 0.0); name=:bottom)

    state1 = make_circle_state((0.5, 0.2))
    g1 = PenguinStokes._wall_gap(state1, params, bottom_wall)
    @test g1 ≈ 0.1  atol=1e-14

    state2 = make_circle_state((0.5, 0.05))
    g2 = PenguinStokes._wall_gap(state2, params, bottom_wall)
    @test g2 ≈ -0.05  atol=1e-14
end

# ── 2. Wall contact force ────────────────────────────────────────────────────

@testset "Wall contact force magnitude" begin
    R = 0.1
    params = make_circle_params(R)
    bottom_wall = PlanarWallContact((0.0, 1.0), (0.0, 0.0); name=:bottom)
    model = NormalSpringDashpotContact(stiffness=1000.0, damping=10.0, gap_tol=0.0)

    # Center at y=0.09, V=(0,-1): penetration δ=0.01, vn=-1 → Fn = 1000*0.01 - 10*(-1) = 20
    state = make_circle_state((0.5, 0.09), (0.0, -1.0))
    Fw, g = PenguinStokes.wall_contact_force(state, params, bottom_wall, model)
    @test g ≈ -0.01  atol=1e-12
    @test Fw[1] ≈ 0.0   atol=1e-12
    @test Fw[2] ≈ 20.0  atol=1e-10
end

# ── 3. No adhesion ────────────────────────────────────────────────────────────

@testset "No adhesion (separating fast)" begin
    R = 0.1
    params = make_circle_params(R)
    bottom_wall = PlanarWallContact((0.0, 1.0), (0.0, 0.0); name=:bottom)
    model = NormalSpringDashpotContact(stiffness=1000.0, damping=10.0, gap_tol=0.0)

    # Moving away fast → Fn should be clamped to 0 (no attraction)
    state = make_circle_state((0.5, 0.09), (0.0, 100.0))
    Fw, _ = PenguinStokes.wall_contact_force(state, params, bottom_wall, model)
    @test Fw[2] >= 0.0   # never pulls toward wall
    @test Fw[2] == 0.0   # fast separation kills the spring force
end

# ── 4. Projection ─────────────────────────────────────────────────────────────

@testset "Wall projection removes penetration" begin
    R = 0.1
    proj_tol = 1e-6
    params = make_circle_params(R)
    bottom_wall = PlanarWallContact((0.0, 1.0), (0.0, 0.0); name=:bottom)
    box = BoxContact{2,Float64}([bottom_wall])
    model = NormalSpringDashpotContact(stiffness=1.0, damping=0.0, gap_tol=0.0,
                                       projection_tol=proj_tol, enable_projection=true)

    state = make_circle_state((0.5, 0.05))  # penetrating: y=0.05, R=0.1 → g=-0.05
    apply_contact_projection!(state, params, (box,), model)
    X = state.X
    g_after = X[2] - R
    @test g_after >= proj_tol - 1e-14
end

# ── 5. Pairwise gap ───────────────────────────────────────────────────────────

@testset "Pairwise gap" begin
    R = 0.1
    params = make_circle_params(R)

    s1 = make_circle_state((0.0, 0.0))
    s2 = make_circle_state((0.25, 0.0))
    d1 = norm(s2.X - s1.X)
    g1 = d1 - 2R
    @test g1 ≈ 0.05  atol=1e-14

    s3 = make_circle_state((0.0, 0.0))
    s4 = make_circle_state((0.15, 0.0))
    d2 = norm(s4.X - s3.X)
    g2 = d2 - 2R
    @test g2 ≈ -0.05  atol=1e-14
end

# ── 6. Pairwise contact force symmetry ───────────────────────────────────────

@testset "Pairwise contact force symmetry (Fi + Fj = 0)" begin
    R = 0.1
    p1 = make_circle_params(R)
    p2 = make_circle_params(R)

    # Overlapping, approaching
    s1 = make_circle_state((0.0, 0.0), (1.0, 0.0))
    s2 = make_circle_state((0.15, 0.0), (-1.0, 0.0))

    model = NormalSpringDashpotContact(stiffness=1000.0, damping=10.0, gap_tol=0.0)
    pc = PairwiseParticleContact(nothing)
    pf = pairwise_contact_forces([s1, s2], [p1, p2], pc, model)

    @test norm(pf[1].force + pf[2].force) < 1e-12
    # Force on body 1 should point left (away from body 2)
    @test pf[1].force[1] < 0
    @test pf[2].force[1] > 0
end

# ── 7. Pairwise projection ────────────────────────────────────────────────────

@testset "Pairwise projection separates overlapping pair" begin
    R = 0.1
    proj_tol = 1e-6
    p1 = make_circle_params(R)
    p2 = make_circle_params(R)

    s1 = make_circle_state((0.0, 0.0))
    s2 = make_circle_state((0.15, 0.0))  # overlap: dist=0.15 < 2R=0.2

    model = NormalSpringDashpotContact(stiffness=1.0, damping=0.0, gap_tol=0.0,
                                       projection_tol=proj_tol, enable_projection=true)
    pc = PairwiseParticleContact(nothing)
    apply_contact_projection!([s1, s2], [p1, p2], (pc,), model)

    dist_after = norm(s2.X - s1.X)
    @test dist_after >= 2R + proj_tol - 1e-14
end

# ── 8. FSI compatibility (single body) ───────────────────────────────────────

@testset "step_fsi! with contact_model=nothing is backward-compatible" begin
    T = Float64
    n = 9
    R = 0.12
    grid = CartesianGrid((zero(T), zero(T)), (one(T), one(T)), (n, n))
    bc_per = BorderConditions(; left=Periodic(), right=Periodic(),
                               bottom=Periodic(), top=Periodic())
    shape = Circle(R)
    center = SVector{2,T}(0.5, 0.5)
    U = SVector{2,T}(0.1, 0.0)
    body_func(x, y, t) = shape.R - hypot(x - center[1], y - center[2])
    model = MovingStokesModelMono(
        grid, body_func, one(T), one(T);
        bc_u=(bc_per, bc_per),
        force=(zero(T), zero(T)),
        bc_cut_u=(Dirichlet(U[1]), Dirichlet(U[2])),
    )
    state = RigidBodyState(Tuple(center), Tuple(U))
    params = RigidBodyParams(one(T), one(T), shape, SVector(0.0, 0.0); buoyancy=false)
    fsi = StokesFSIProblem(model, state, params)

    # Without contact
    out1 = step_fsi!(fsi; t=zero(T), dt=T(0.02), contact_model=nothing)
    @test haskey(out1, :contact)  # contact field always present
    @test out1.contact.active == false
    @test out1.contact.ncontacts == 0

    # With contact model but no contact (particle far from walls in periodic domain)
    box = box_contacts_from_grid(grid)
    contact_model = NormalSpringDashpotContact(stiffness=1e4, damping=10.0)
    out2 = step_fsi!(fsi; t=T(0.02), dt=T(0.02),
                     contact_model=contact_model, contact_constraints=(box,))
    @test haskey(out2, :contact)
end

# ── 9. Multi-body compatibility ───────────────────────────────────────────────

@testset "step_multi_fsi! returns contact diagnostics" begin
    T = Float64
    n = 9
    R = 0.08
    grid = CartesianGrid((zero(T), zero(T)), (one(T), one(T)), (n, n))
    bc_per = BorderConditions(; left=Periodic(), right=Periodic(),
                               bottom=Periodic(), top=Periodic())

    shape = Circle(R)
    c1 = SVector{2,T}(0.3, 0.5)
    c2 = SVector{2,T}(0.7, 0.5)
    U1 = SVector{2,T}(0.05, 0.0)
    U2 = SVector{2,T}(-0.05, 0.0)

    body_func(x, y, t) = max(
        shape.R - hypot(x - 0.3, y - 0.5),
        shape.R - hypot(x - 0.7, y - 0.5),
    )

    model = MovingStokesModelMono(
        grid, body_func, one(T), one(T);
        bc_u=(bc_per, bc_per),
        force=(zero(T), zero(T)),
        bc_cut_u=(Dirichlet(zero(T)), Dirichlet(zero(T))),
    )

    state1 = RigidBodyState(Tuple(c1), Tuple(U1))
    state2 = RigidBodyState(Tuple(c2), Tuple(U2))
    params1 = RigidBodyParams(one(T), one(T), shape, SVector(0.0, 0.0); buoyancy=false)
    params2 = RigidBodyParams(one(T), one(T), shape, SVector(0.0, 0.0); buoyancy=false)

    fsi = MultiBodyFSIProblem(model, [state1, state2], [params1, params2], [shape, shape])

    contact_model = NormalSpringDashpotContact(stiffness=1e4, damping=10.0)
    contact_constraints = (
        box_contacts_from_grid(grid),
        PairwiseParticleContact(nothing),
    )

    out = step_multi_fsi!(fsi; t=zero(T), dt=T(0.02),
                          contact_model=contact_model,
                          contact_constraints=contact_constraints)
    @test haskey(out, :contact)
    @test haskey(out.contact, :active)
    @test haskey(out.contact, :ncontacts)
    @test haskey(out.contact, :min_gap)
    @test haskey(out.contact, :forces)
    @test length(out.contact.forces) == 2
end
