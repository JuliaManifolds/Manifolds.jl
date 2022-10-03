using Manifolds

using OrdinaryDiffEq
using DiffEqCallbacks
using Test
using Manifolds: TFVector
using RecursiveArrayTools
using BoundaryValueDiffEq

@testset "Torus in ℝ³" begin
    M = Manifolds.EmbeddedTorus(3, 2)
    A = Manifolds.DefaultTorusAtlas()

    p0x = [0.5, -1.2]
    X_p0x = [-1.2, 0.4]
    p = [Manifolds._torus_param(M, p0x...)...]
    @test p ≈ [1.7230709564189848, -4.431999755591838, 0.958851077208406]
    i_p0x = Manifolds.get_chart_index(M, A, p)
    @test [i_p0x...] ≈ p0x
    B = induced_basis(M, A, i_p0x)
    X = get_vector(M, p, X_p0x, B)
    @test get_coordinates(M, p, X, B) ≈ X_p0x

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
        final_time=3.0,
    )
    pexp_3 = p_exp(3.0)
    @test pexp_3[1] ≈ [2.701765894057119, 2.668437820810143, -1.8341712552932237]
    @test pexp_3[2] ≈ [-0.41778834843865575, 2.935021992911625, 0.7673987137187901]
    @test pexp_3[3] ≈ [7.661627684089519, 4.629037950515605, 3.7839234533367194]

    p_exp = Manifolds.solve_chart_exp_ode(M, [0.0, 0.0], X_p0x, A, i_p0x; final_time=3.0)
    pexp_3 = p_exp(3.0)
    @test isapprox(
        pexp_3[1],
        [2.701765894057119, 2.668437820810143, -1.8341712552932237],
        rtol=1e-5,
    )
    @test isapprox(
        pexp_3[2],
        [-0.41778834843865575, 2.935021992911625, 0.7673987137187901],
        rtol=1e-5,
    )

    p_exp_switch = Manifolds.solve_chart_exp_ode(
        M,
        [0.0, 0.0],
        X_p0x,
        A,
        i_p0x;
        final_time=3.0,
        check_chart_switch_kwargs=(; ϵ=0.3),
    )
    p_exp_switch_3 = p_exp_switch(3.0)
    @test isapprox(
        p_exp_switch_3[1],
        [2.701765894057119, 2.668437820810143, -1.8341712552932237],
        rtol=1e-5,
    )
    @test length(p_exp_switch.sols) < length(p_exp.sols)

    Manifolds.transition_map_diff(M, A, i_p0x, [0.0, 0.0], X_p0x, (-1.0, -0.3))

    a2 = [-0.5, 0.3]
    sol_log = Manifolds.solve_chart_log_bvp(M, p0x, a2, A, (0, 0))
    @test sol_log(0.0)[1:2] ≈ p0x
    @test sol_log(1.0)[1:2] ≈ a2
    @test norm(M, A, (0, 0), p0x, sol_log(0.0)[3:4]) ≈
          Manifolds.estimate_distance_from_bvp(M, p0x, a2, A, (0, 0))
end
