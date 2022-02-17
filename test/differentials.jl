using Manifolds
using Manifolds:
    FlatExpDiffArgumentMethod,
    FiniteDifferenceLogDiffArgumentMethod,
    exp_diff_argument,
    log_diff_argument,
    retract_diff_argument,
    retract_diff_argument!,
    inverse_retract_diff_argument,
    inverse_retract_diff_argument!
using Test
using LinearAlgebra

@testset "Differentials on SO(3)" begin
    G = SpecialOrthogonal(3)
    M = base_manifold(G)
    p = Matrix(I, 3, 3)

    ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
    pts = [exp(M, p, hat(M, p, ωi)) for ωi in ω]
    Xpts = [hat(M, p, [-1.0, 2.0, 0.5]), hat(M, p, [1.0, 0.0, 0.5])]

    @testset "differentials" begin
        lged0 = Manifolds.LieGroupExpDiffArgumentApprox(0)
        q2 = exp(G, pts[1], Xpts[2])
        @test isapprox(
            G,
            q2,
            retract_diff_argument(G, pts[1], Xpts[1], Xpts[2], lged0),
            Xpts[2],
        )
        lged20 = Manifolds.LieGroupExpDiffArgumentApprox(20)
        diff_ref = [
            0.0 -0.7482721017619345 -0.508151233069837
            0.7482721017619345 0.0 -0.10783358474129323
            0.508151233069837 0.10783358474129323 0.0
        ]
        @test isapprox(
            G,
            q2,
            retract_diff_argument(G, pts[1], Xpts[1], Xpts[2], lged20),
            diff_ref;
            atol=1e-12,
        )
    end
end

@testset "FiniteDifferenceLogDiffArgumentMethod" begin
    M = Sphere(2)
    lda_1_e_4 = FiniteDifferenceLogDiffArgumentMethod(M, 1e-4)
    p = [1.0, 0.0, 0.0]
    q = [0.0, sqrt(2) / 2, sqrt(2) / 2]
    X = [1.0, -2.0, 2.0]

    # computed using Manopt.differential_log_argument(M, p, q, X)
    diff_ref = [-5.131524956784507e-33, -3.84869943477634, 2.434485872403245]
    @test isapprox(
        M,
        q,
        inverse_retract_diff_argument(M, p, q, X, lda_1_e_4),
        diff_ref;
        atol=1e-7,
    )
end

@testset "Flat differentials" begin
    M = Euclidean(3)
    @test Manifolds.default_retract_diff_argument_method(M, ExponentialRetraction()) ===
          FlatExpDiffArgumentMethod()

    p = [1.0, -1.0, 2.0]
    q = [2.0, 1.0, 0.0]
    X1 = [0.0, 1.0, 2.0]
    X2 = [2.0, -2.0, 0.0]
    @test isapprox(M, p, retract_diff_argument(M, p, X1, X2), X2)
    @test isapprox(M, p, exp_diff_argument(M, p, X1, X2), X2)
    Y = similar(X1)
    retract_diff_argument!(M, Y, p, X1, X2)
    @test isapprox(M, p, Y, X2)

    @test isapprox(M, q, inverse_retract_diff_argument(M, p, q, X1), X1)
    @test isapprox(M, q, log_diff_argument(M, p, q, X1), X1)

    Y = similar(X1)
    inverse_retract_diff_argument!(M, Y, p, q, X1)
    @test isapprox(M, q, Y, X1)
end
