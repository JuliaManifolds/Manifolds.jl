using Manifolds
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
            Manifolds.retract_diff_argument(G, pts[1], Xpts[1], Xpts[2], lged0),
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
            Manifolds.retract_diff_argument(G, pts[1], Xpts[1], Xpts[2], lged20),
            diff_ref;
            atol=1e-12,
        )
    end
end
