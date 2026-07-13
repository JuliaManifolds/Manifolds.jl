using Manifolds, Test
using ManifoldDiff
using DiffEqCallbacks, OrdinaryDiffEq, RecursiveArrayTools


@testset "Atlases" begin
    M = Sphere(2)
    A = Manifolds.StereographicAtlas()
    a = [0.2, -0.3]
    i = :north
    p = get_point(M, A, i, a)
    B = induced_basis(M, A, i)
    Xc = [0.3, 0.2]
    Yc = [-0.2, 0.25]
    X = get_vector(M, p, Xc, B)
    Y = get_vector(M, p, Yc, B)

    solution = Manifolds.solve_chart_jacobi_field(
        M, a, Xc, A, i, Yc, zeros(2); final_time = 1.0
    )
    p_final, _, Y_final, _ = solution(1.0)

    expected = zero_vector(M, p_final)
    ManifoldDiff.jacobi_field!(
        M,
        expected,
        p,
        exp(M, p, X),
        1.0,
        Y,
        ManifoldDiff.βdifferential_exp_basepoint,
    )
    @test p_final ≈ exp(M, p, X) atol = 1.0e-8
    @test Y_final ≈ expected atol = 1.0e-8
end