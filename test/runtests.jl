using ManifoldMuseum
using Test

"""
    test_manifold(m::Manifold, pts::AbstractVector)

Tests general properties of manifold `m`, given at least three different points
that lie on it (contained in `pts`).
"""
function test_manifold(M::Manifold, pts::AbstractVector)
    # log/exp
    length(pts) ≥ 3 || error("Not enough points (at least three expected)")
    isapprox(M, pts[1], pts[2]) && error("Points 1 and 2 are equal")
    isapprox(M, pts[1], pts[3]) && error("Points 1 and 3 are equal")

    tv1 = log(M, pts[1], pts[2])
    @test isapprox(M, pts[2], exp(M, pts[1], tv1))
    for x ∈ pts
        @test isapprox(M, zero_tangent_vector(M, x), log(M, pts[1], pts[1]))
    end
    zero_tangent_vector!(M, tv1, pts[1])
    @test isapprox(tv1, zero_tangent_vector(M, pts[1]))
    log!(M, tv1, pts[1], pts[2])
    @test norm(M, pts[1], tv1) ≈ sqrt(dot(M, pts[1], tv1, tv1))
end

@testset "Sphere" begin
    test_manifold(ManifoldMuseum.Sphere((3,)), [[1.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0],
                                                [0.0, 0.0, 1.0]])
end
