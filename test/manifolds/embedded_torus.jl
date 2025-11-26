using Manifolds

using OrdinaryDiffEq
using DiffEqCallbacks
using Test
using Manifolds: TFVector
using RecursiveArrayTools
using BoundaryValueDiffEq
using Einsum

@testset "Torus in ℝ³" begin
    M = Manifolds.EmbeddedTorus(3, 2)
    A = Manifolds.DefaultTorusAtlas()
    @test !is_flat(M)

    p0x = [0.5, -1.2]
    X_p0x = [-1.2, 0.4]
    Y_p0x = [-0.2, -0.3]
    p = [Manifolds._torus_param(M, p0x...)...]
    @test p ≈ [1.7230709564189848, -4.431999755591838, 0.958851077208406]
    i_p0x = Manifolds.get_chart_index(M, A, p)
    @test [i_p0x...] ≈ p0x
    B = induced_basis(M, A, i_p0x)
    X = get_vector(M, p, X_p0x, B)
    Y = get_vector(M, p, Y_p0x, B)
    @test get_coordinates(M, p, X, B) ≈ X_p0x
    Γ_X_Y = [-0.20602262287496229, -0.11547890480178581]
    @test affine_connection(M, A, i_p0x, p0x, X_p0x, Y_p0x) ≈ Γ_X_Y
    Z_p0x = similar(X_p0x)
    Manifolds.levi_civita_affine_connection!(M, Z_p0x, A, i_p0x, p0x, X_p0x, Y_p0x)
    @test Z_p0x ≈ Γ_X_Y
    @test christoffel_symbols_second(M, A, i_p0x, p0x) ≈ [0.0 0.0; 0.0 -0.37637884969165114;;; 0.0 1.5668024839360153; -0.37637884969165114 0.0]
    RR = [0.0 0.0; 0.0 0.2648148288193222;;; 0.0 0.0; -0.2648148288193222 0.0;;;; 0.0 -1.1023800405286386; 0.0 0.0;;; 1.1023800405286386 0.0; 0.0 0.0]
    @test riemann_tensor(M, A, i_p0x, p0x) ≈ RR
    W_p0x = similar(X_p0x)
    @einsum W_p0x[i] = RR[i, j, k, l] * X_p0x[j] * Y_p0x[k] * X_p0x[l]
    @test riemann_tensor(M, A, i_p0x, p0x, X_p0x, Y_p0x, X_p0x) ≈ W_p0x

    Ric = ricci_tensor(M, A, i_p0x, p0x)
    Ric_ref = zeros(2, 2)
    @einsum Ric_ref[i, j] = RR[k, i, k, j]
    @test Ric ≈ Ric_ref
    @test ricci_curvature(M, A, i_p0x, p0x) ≈ sum(inverse_local_metric(M, A, i_p0x, p0x) .* Ric)
    @testset "generic, default implementation" begin
        Z = similar(X)
        invoke(
            Manifolds.get_vector_induced_basis!,
            Tuple{AbstractManifold, Any, Any, Any, InducedBasis{ℝ, Manifolds.TangentSpaceType, <:AbstractAtlas}},
            M, Z, p, X_p0x, B
        )
        @test Z ≈ X
        invoke(
            Manifolds.get_coordinates_induced_basis!,
            Tuple{AbstractManifold, Any, Any, Any, InducedBasis{ℝ, Manifolds.TangentSpaceType, <:AbstractAtlas}},
            M, Z_p0x, p, X, B
        )
        @test Z_p0x ≈ X_p0x
    end

    @test norm(X) ≈ norm(M, A, (0.0, 0.0), p0x, X_p0x)

    @test Manifolds.aspect_ratio(M) == 3 / 2
    @test check_point(M, p) === nothing
    @test check_point(M, [0.0, 0.0, 0.0]) isa DomainError

    @test check_vector(M, p, X) === nothing
    @test check_vector(M, p, [1, 2, 3]) isa DomainError

    @test get_embedding(M) === Euclidean(3)
    @test Manifolds.inverse_chart_injectivity_radius(M, A, i_p0x) === π
    @test gaussian_curvature(M, p) ≈ 0.09227677052701619
    @test Manifolds.normal_vector(M, p) ≈
        [0.3179988464944819, -0.8179412488450798, 0.479425538604203]

    p_exp = Manifolds.solve_chart_parallel_transport_ode(
        M,
        [0.0, 0.0],
        X_p0x,
        A,
        i_p0x,
        [1.0, 2.0];
        final_time = 3.0,
    )
    pexp_3 = p_exp(3.0)
    @test isapprox(
        pexp_3[1],
        [2.701765894057119, 2.668437820810143, -1.8341712552932237];
        atol = 1.0e-5,
    )
    @test isapprox(
        pexp_3[2],
        [-0.41778834843865575, 2.935021992911625, 0.7673987137187901];
        atol = 1.0e-5,
    )
    @test isapprox(
        pexp_3[3],
        [7.661627684089519, 4.629037950515605, 3.7839234533367194];
        atol = 1.0e-4,
    )
    @test_throws DomainError p_exp(10.0)
    @test_throws DomainError p_exp(-1.0)
    @test p_exp([3.0]) == [pexp_3]

    p_exp = Manifolds.solve_chart_exp_ode(M, [0.0, 0.0], X_p0x, A, i_p0x; final_time = 3.0)
    pexp_3 = p_exp(3.0)
    @test isapprox(
        pexp_3[1],
        [2.701765894057119, 2.668437820810143, -1.8341712552932237],
        rtol = 1.0e-5,
    )
    @test isapprox(
        pexp_3[2],
        [-0.41778834843865575, 2.935021992911625, 0.7673987137187901],
        rtol = 1.0e-5,
    )
    @test_throws DomainError p_exp(10.0)
    @test_throws DomainError p_exp(-1.0)

    p_exp_switch = Manifolds.solve_chart_exp_ode(
        M,
        [0.0, 0.0],
        X_p0x,
        A,
        i_p0x;
        final_time = 3.0,
        check_chart_switch_kwargs = (; ϵ = 0.3),
    )
    p_exp_switch_3 = p_exp_switch(3.0)
    @test isapprox(
        p_exp_switch_3[1],
        [2.701765894057119, 2.668437820810143, -1.8341712552932237],
        rtol = 1.0e-5,
    )
    @test length(p_exp_switch.sols) < length(p_exp.sols)

    Manifolds.transition_map_diff(M, A, i_p0x, [0.0, 0.0], X_p0x, (-1.0, -0.3))

    a2 = [-0.5, 0.3]
    sol_log = Manifolds.solve_chart_log_bvp(M, p0x, a2, A, (0, 0))
    @test sol_log(0.0)[1:2] ≈ p0x
    @test sol_log(1.0)[1:2] ≈ a2 atol = 1.0e-7
    # a test randomly failed here on Julia 1.6 once for no clear reason?
    # so I bumped tolerance considerably
    bvp_atol = VERSION < v"1.7" ? 2.0e-3 : 1.0e-15
    @test isapprox(
        norm(M, A, (0, 0), p0x, sol_log(0.0)[3:4]),
        Manifolds.estimate_distance_from_bvp(M, p0x, a2, A, (0, 0));
        atol = bvp_atol,
    )

    @test Manifolds.IntegratorTerminatorNearChartBoundary().check_chart_switch_kwargs ===
        NamedTuple()
end
