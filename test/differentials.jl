using Manifolds
using Manifolds: FlatExpDiffArgumentMethod, retract_diff_argument, retract_diff_argument!
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

@testset "Flat differentials" begin
    M = Euclidean(3)
    @test Manifolds.default_retract_diff_argument_method(M, ExponentialRetraction()) ===
          FlatExpDiffArgumentMethod()

    p = [1.0, -1.0, 2.0]
    X1 = [0.0, 1.0, 2.0]
    X2 = [2.0, -2.0, 0.0]
    @test isapprox(M, p, retract_diff_argument(M, p, X1, X2), X2)
    Y = similar(X1)
    retract_diff_argument!(M, Y, p, X1, X2)
    @test isapprox(M, p, Y, X2)
end
