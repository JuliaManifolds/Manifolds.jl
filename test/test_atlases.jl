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

    @test Manifolds.solve_chart_volume_density(M, a, Xc, A, i) ≈
        volume_density(M, p, X) atol = 1.0e-8

    solution = Manifolds.solve_chart_differential_exp_basepoint(
        M, a, Xc, A, i, Yc; final_time = 1.0
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

    solution = Manifolds.solve_chart_differential_exp_argument(
        M, a, Xc, A, i, Yc; final_time = 1.0
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
        ManifoldDiff.βdifferential_exp_argument,
    )
    @test p_final ≈ exp(M, p, X) atol = 1.0e-8
    @test Y_final ≈ expected atol = 1.0e-8

    solution = Manifolds.solve_chart_differential_log_basepoint(
        M, a, Xc, A, i, Yc; final_time = 1.0
    )
    _, _, _, dY_initial = solution(0.0)

    expected = zero_vector(M, p)
    ManifoldDiff.jacobi_field!(
        M,
        expected,
        p,
        exp(M, p, X),
        0.0,
        Y,
        ManifoldDiff.βdifferential_log_basepoint,
    )
    @test dY_initial ≈ expected atol = 1.0e-8

    q = exp(M, p, X)
    Bq = induced_basis(M, A, i)
    Yq = get_vector(M, q, Yc, Bq)
    solution = Manifolds.solve_chart_differential_log_argument(
        M, a, Xc, A, i, Yc; final_time = 1.0
    )
    _, _, _, dY_initial = solution(0.0)

    expected = zero_vector(M, p)
    ManifoldDiff.jacobi_field!(
        M,
        expected,
        q,
        p,
        1.0,
        Yq,
        ManifoldDiff.βdifferential_log_argument,
    )
    @test dY_initial ≈ expected atol = 1.0e-8
end
