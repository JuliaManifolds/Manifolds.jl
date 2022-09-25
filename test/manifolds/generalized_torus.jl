using Manifolds

using OrdinaryDiffEq
using DiffEqCallbacks
using Test
using Manifolds: TFVector
using RecursiveArrayTools

@testset "Torus in ℝ³" begin
    M = Manifolds.TorusInR3(3, 2)
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

    @test norm(X) ≈ norm(M, [0.0, 0.0], TFVector(X_p0x, B))

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
end
